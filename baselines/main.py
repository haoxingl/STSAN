import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import time
import data_loader as dl
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
import tensorflow as tf
from tensorflow.keras import layers, models
import parameters_nyctaxi as param_taxi
import parameters_nycbike as param_bike

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)

dataset = 'taxi'
verbose = 0
num_slots_curr = 12
output_size = 2

saved_data = False

if dataset == 'taxi':
    d_loader = dl.data_loader(dataset)
    mask_threshold = 10 / param_taxi.flow_train_max
    param = param_taxi
elif dataset == 'bike':
    d_loader = dl.data_loader(dataset)
    mask_threshold = 10 / param_bike.flow_train_max
    param = param_bike

_, _, _, flow_inputs_currday_train, _, _, _, ys_train \
    = d_loader.generate_data('train', 0, 1, 1, num_slots_curr - 1, 0, 3, saved_data)

_, _, _, flow_inputs_currday_test, _, _, _, ys_test \
    = d_loader.generate_data('test', 0, 1, 1, num_slots_curr - 1, 0, 3, saved_data)

original_train = flow_inputs_currday_train[:, 3, 3, :, :]
original_test = flow_inputs_currday_test[:, 3, 3, :, :]


class CustomStopper(tf.keras.callbacks.EarlyStopping):
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', start_epoch=5):
        super().__init__(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)


def historical_average(flow, target):
    in_mask = np.greater(target[:, 0], mask_threshold)
    out_mask = np.greater(target[:, 1], mask_threshold)
    averaged_sum = np.squeeze(np.mean(flow, axis=-2))
    in_rmse = np.sqrt(np.mean(np.square(target[:, 0][in_mask] - averaged_sum[:, 0][in_mask])))
    out_rmse = np.sqrt(np.mean(np.square(target[:, 1][out_mask] - averaged_sum[:, 1][out_mask])))
    in_mae = np.mean(np.abs(target[:, 0][in_mask] - averaged_sum[:, 0][in_mask]))
    out_mae = np.mean(np.abs(target[:, 1][out_mask] - averaged_sum[:, 1][out_mask]))
    return in_rmse, out_rmse, in_mae, out_mae


def arima(flow, target):
    warnings.filterwarnings("ignore")
    in_mask = np.greater(target[:, 0], mask_threshold)
    out_mask = np.greater(target[:, 1], mask_threshold)
    result = np.zeros((flow.shape[0], flow.shape[-1]))
    for i in range(flow.shape[0]):
        if verbose:
            if (i + 1) % 10000 == 0:
                print("ARIMA: line {} of {}".format(i + 1, flow.shape[0]))
        for j in range(flow.shape[-1]):
            data = (flow[i, :, j]).tolist()
            try:
                model = SARIMAX(data,
                                order=(1, 1, 0),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                model_fit = model.fit(disp=False)
                result[i, j] = model_fit.predict(len(data), len(data))[0]
            except:
                result[i, j] = 0.0
                pass
    in_rmse = np.sqrt(np.mean(np.square(target[:, 0][in_mask] - result[:, 0][in_mask])))
    out_rmse = np.sqrt(np.mean(np.square(target[:, 1][out_mask] - result[:, 1][out_mask])))
    in_mae = np.mean(np.abs(target[:, 0][in_mask] - result[:, 0][in_mask]))
    out_mae = np.mean(np.abs(target[:, 1][out_mask] - result[:, 1][out_mask]))
    return in_rmse, out_rmse, in_mae, out_mae


def var(flow, target):
    warnings.filterwarnings("ignore")
    in_mask = np.greater(target[:, 0], mask_threshold)
    out_mask = np.greater(target[:, 1], mask_threshold)
    result = np.zeros((flow.shape[0], flow.shape[-1]))
    for i in range(flow.shape[0]):
        if verbose:
            if (i + 1) % 10000 == 0:
                print("VAR: line {} of {}".format(i + 1, flow.shape[0]))
        for j in range(flow.shape[-1]):
            data = list()
            for k in range(flow.shape[1] - 1):
                data.append([flow[i, k, j], flow[i, k + 1, j]])
            model = VAR(data)
            try:
                model_fit = model.fit()
                result[i, j] = model_fit.forecast(model_fit.y, steps=1)[0][1]
            except:
                result[i, j] = 0.0
                pass
    in_rmse = np.sqrt(np.mean(np.square(target[:, 0][in_mask] - result[:, 0][in_mask])))
    out_rmse = np.sqrt(np.mean(np.square(target[:, 1][out_mask] - result[:, 1][out_mask])))
    in_mae = np.mean(np.abs(target[:, 0][in_mask] - result[:, 0][in_mask]))
    out_mae = np.mean(np.abs(target[:, 1][out_mask] - result[:, 1][out_mask]))
    return in_rmse, out_rmse, in_mae, out_mae


def mlp(x_train, y_train, x_test, y_test):
    half_size = int(x_train.shape[2] *  x_train.shape[1] / 2)
    early_stop = CustomStopper(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min', start_epoch=5)

    model = models.Sequential(
        [
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(output_size, activation='tanh')
        ]
    )

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mse', 'mae'])

    model.fit(x_train, y_train, epochs=500, batch_size=64, validation_split=0.2, callbacks=[early_stop], verbose=verbose)

    y_pred = model.predict(x_test)

    in_mask = np.greater(y_test[:, 0], mask_threshold)
    out_mask = np.greater(y_test[:, 1], mask_threshold)
    in_rmse = np.sqrt(np.mean(np.square(y_test[:, 0][in_mask] - y_pred[:, 0][in_mask])))
    out_rmse = np.sqrt(np.mean(np.square(y_test[:, 1][out_mask] - y_pred[:, 1][out_mask])))
    in_mae = np.mean(np.abs(y_test[:, 0][in_mask] - y_pred[:, 0][in_mask]))
    out_mae = np.mean(np.abs(y_test[:, 1][out_mask] - y_pred[:, 1][out_mask]))
    return in_rmse, out_rmse, in_mae, out_mae


def lstm(x_train, y_train, x_test, y_test):
    early_stop = CustomStopper(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min', start_epoch=5)

    model = models.Sequential(
        [
            tf.keras.layers.LSTM(64, recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(output_size, activation='tanh')
        ]
    )

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mse', 'mae'])

    model.fit(x_train, y_train, epochs=500, batch_size=64, validation_split=0.2, callbacks=[early_stop], verbose=verbose)

    y_pred = model.predict(x_test)

    in_mask = np.greater(y_test[:, 0], mask_threshold)
    out_mask = np.greater(y_test[:, 1], mask_threshold)
    in_rmse = np.sqrt(np.mean(np.square(y_test[:, 0][in_mask] - y_pred[:, 0][in_mask])))
    out_rmse = np.sqrt(np.mean(np.square(y_test[:, 1][out_mask] - y_pred[:, 1][out_mask])))
    in_mae = np.mean(np.abs(y_test[:, 0][in_mask] - y_pred[:, 0][in_mask]))
    out_mae = np.mean(np.abs(y_test[:, 1][out_mask] - y_pred[:, 1][out_mask]))
    return in_rmse, out_rmse, in_mae, out_mae


def gru(x_train, y_train, x_test, y_test):
    early_stop = CustomStopper(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min', start_epoch=5)

    model = models.Sequential(
        [
            tf.keras.layers.GRU(64, recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(output_size, activation='tanh')
        ]
    )

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mse', 'mae'])

    model.fit(x_train, y_train, epochs=500, batch_size=64, validation_split=0.2, callbacks=[early_stop], verbose=verbose)

    y_pred = model.predict(x_test)

    in_mask = np.greater(y_test[:, 0], mask_threshold)
    out_mask = np.greater(y_test[:, 1], mask_threshold)
    in_rmse = np.sqrt(np.mean(np.square(y_test[:, 0][in_mask] - y_pred[:, 0][in_mask])))
    out_rmse = np.sqrt(np.mean(np.square(y_test[:, 1][out_mask] - y_pred[:, 1][out_mask])))
    in_mae = np.mean(np.abs(y_test[:, 0][in_mask] - y_pred[:, 0][in_mask]))
    out_mae = np.mean(np.abs(y_test[:, 1][out_mask] - y_pred[:, 1][out_mask]))
    return in_rmse, out_rmse, in_mae, out_mae


if __name__ == "__main__":
    for i in range(1):
        print("\nTest Round {}\n".format(i + 1))

        start = time.time()
        print("Calculating Historical Average RMSE, MAE results...")
        in_rmse, out_rmse, in_mae, out_mae = historical_average(original_test, ys_test)
        print("Historical average: IN_RMSE {}, OUT_RMSE {}, IN_MAE {}, OUT_MAE {}" \
              .format(in_rmse * param.flow_train_max, out_rmse * param.flow_train_max, in_mae * param.flow_train_max, out_mae * param.flow_train_max))
        print("Time used: {} seconds".format(time.time() - start))

        start = time.time()
        print("Calculating ARIMA RMSE, MAE results...")
        in_rmse, out_rmse, in_mae, out_mae = arima(original_test, ys_test)
        print("ARIMA: IN_RMSE {}, OUT_RMSE {}, IN_MAE {}, OUT_MAE {}" \
              .format(in_rmse * param.flow_train_max, out_rmse * param.flow_train_max, in_mae * param.flow_train_max, out_mae * param.flow_train_max))
        print("Time used: {} seconds".format(time.time() - start))

        start = time.time()
        print("Calculating VAR RMSE, MAE results...")
        in_rmse, out_rmse, in_mae, out_mae = var(original_test, ys_test)
        print("VAR: IN_RMSE {}, OUT_RMSE {}, IN_MAE {}, OUT_MAE {}" \
              .format(in_rmse * param.flow_train_max, out_rmse * param.flow_train_max, in_mae * param.flow_train_max, out_mae * param.flow_train_max))
        print("Time used: {} seconds".format(time.time() - start))

        try:
            start = time.time()
            print("Calculating MLP RMSE, MAE results...")
            in_rmse, out_rmse, in_mae, out_mae = mlp(original_train, ys_train, original_test, ys_test)
            print("MLP: IN_RMSE {}, OUT_RMSE {}, IN_MAE {}, OUT_MAE {}" \
                  .format(in_rmse * param.flow_train_max, out_rmse * param.flow_train_max,
                          in_mae * param.flow_train_max, out_mae * param.flow_train_max))
            print("Time used: {} seconds".format(time.time() - start))

            start = time.time()
            print("Calculating LSTM RMSE, MAE results...")
            in_rmse, out_rmse, in_mae, out_mae = lstm(original_train, ys_train, original_test, ys_test)
            print("LSTM: IN_RMSE {}, OUT_RMSE {}, IN_MAE {}, OUT_MAE {}" \
                  .format(in_rmse * param.flow_train_max, out_rmse * param.flow_train_max,
                          in_mae * param.flow_train_max, out_mae * param.flow_train_max))
            print("Time used: {} seconds".format(time.time() - start))

            start = time.time()
            print("Calculating GRU RMSE, MAE results...")
            in_rmse, out_rmse, in_mae, out_mae = gru(original_train, ys_train, original_test, ys_test)
            print("GRU: IN_RMSE {}, OUT_RMSE {}, IN_MAE {}, OUT_MAE {}" \
                  .format(in_rmse * param.flow_train_max, out_rmse * param.flow_train_max,
                          in_mae * param.flow_train_max, out_mae * param.flow_train_max))
            print("Time used: {} seconds".format(time.time() - start))
        except:
            pass

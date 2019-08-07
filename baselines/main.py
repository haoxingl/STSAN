import numpy as np
import time
import data_loader as dl
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
import tensorflow as tf
from tensorflow.keras import layers, models
import parameters_nyctaxi
import parameters_nycbike

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dataset = 'taxi'
verbose = 0
use_all_data = True     # True if want to use the 25 sequences of all 7 previous days,
                        # Flase to use only the data of previous 12 sequences in current day.

if dataset == 'taxi':
    d_loader = dl.data_loader(dataset)
    output_size = parameters_nyctaxi.output_size
    mask_threshold = 10 / parameters_nyctaxi.flow_train_max
elif dataset == 'bike':
    d_loader = dl.data_loader(dataset)
    output_size = parameters_nycbike.output_size
    mask_threshold = 10 / parameters_nycbike.flow_train_max

hist_flow_train, _, curr_flow_train, _, _, targets_train \
    = d_loader.generate_data()

hist_flow_test, _, curr_flow_test, _, _, targets_test \
    = d_loader.generate_data(datatype='test')

flow_train = np.concatenate([hist_flow_train, curr_flow_train], axis=1)

flow_test = np.concatenate([hist_flow_test, curr_flow_test], axis=1)

if use_all_data:
    original_train = flow_train
    original_test = flow_test
else:
    original_train = curr_flow_train
    original_test = curr_flow_test


class CustomStopper(tf.keras.callbacks.EarlyStopping):
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', start_epoch=5):
        super().__init__(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)


def historical_average(flow, target):
    half_size = int(flow.shape[2] / 2)
    mask = np.greater(target, mask_threshold)
    mask.astype(np.float32)
    target *= mask
    averaged_sum = np.squeeze(np.mean(flow, axis=1)) * mask
    in_rmse = np.sqrt(np.mean(np.square(target[:, :half_size] - averaged_sum[:, :half_size])))
    out_rmse = np.sqrt(np.mean(np.square(target[:, half_size:] - averaged_sum[:, half_size:])))
    in_mae = np.mean(np.abs(target[:, :half_size] - averaged_sum[:, :half_size]))
    out_mae = np.mean(np.abs(target[:, half_size:] - averaged_sum[:, half_size:]))
    return in_rmse, out_rmse, in_mae, out_mae


def arima(flow, target):
    warnings.filterwarnings("ignore")
    half_size = int(flow.shape[2] / 2)
    mask = np.greater(target, mask_threshold)
    mask.astype(np.float32)
    target *= mask
    result = np.zeros((flow.shape[0], flow.shape[2]))
    for i in range(flow.shape[0]):
        if verbose:
            print("ARIMA: line {} of {}".format(i, flow.shape[0]))
        for j in range(flow.shape[2]):
            data = (flow[i, :, j]).tolist()
            model = SARIMAX(data,
                            order=(1, 1, 0),
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            model_fit = model.fit(disp=False)
            result[i, j] = model_fit.predict(len(data), len(data))[0]
    result *= mask
    in_rmse = np.sqrt(np.mean(np.square(target[:, :half_size] - result[:, :half_size])))
    out_rmse = np.sqrt(np.mean(np.square(target[:, half_size:] - result[:, half_size:])))
    in_mae = np.mean(np.abs(target[:, :half_size] - result[:, :half_size]))
    out_mae = np.mean(np.abs(target[:, half_size:] - result[:, half_size:]))
    return in_rmse, out_rmse, in_mae, out_mae


def var(flow, target):
    warnings.filterwarnings("ignore")  # specify to ignore warning messages
    half_size = int(flow.shape[2] / 2)
    mask = np.greater(target, mask_threshold)
    mask.astype(np.float32)
    target *= mask
    result = np.zeros((flow.shape[0], flow.shape[2]))
    for i in range(flow.shape[0]):
        if verbose:
            print("VAR: line {} of {}".format(i, flow.shape[0]))
        for j in range(flow.shape[2]):
            data = list()
            for k in range(flow.shape[1] - 1):
                data.append([flow[i, k, j], flow[i, k + 1, j]])
            model = VAR(data)
            model_fit = model.fit()
            result[i, j] = model_fit.forecast(model_fit.y, steps=1)[0][1]
    result *= mask
    in_rmse = np.sqrt(np.mean(np.square(target[:, :half_size] - result[:, :half_size])))
    out_rmse = np.sqrt(np.mean(np.square(target[:, half_size:] - result[:, half_size:])))
    in_mae = np.mean(np.abs(target[:, :half_size] - result[:, :half_size]))
    out_mae = np.mean(np.abs(target[:, half_size:] - result[:, half_size:]))
    return in_rmse, out_rmse, in_mae, out_mae


def MLP(x_train, y_train, x_test, y_test):
    half_size = int(x_train.shape[2] / 2)
    early_stop = CustomStopper(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min', start_epoch=5)

    model = models.Sequential(
        [
            layers.Flatten(),
            layers.Dense(2*output_size, activation='relu'),
            layers.Dense(2*output_size, activation='relu'),
            layers.Dense(output_size)
        ]
    )

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mse', 'mae'])

    model.fit(x_train, y_train, epochs=500, batch_size=64, validation_split=0.2, callbacks=[early_stop], verbose=verbose)

    y_pred = model.predict(x_test)

    mask = np.greater(y_test, mask_threshold)
    mask.astype(np.float32)
    y_test *= mask
    y_pred *= mask
    in_rmse = np.sqrt(np.mean(np.square(y_test[:, :half_size] - y_pred[:, :half_size])))
    out_rmse = np.sqrt(np.mean(np.square(y_test[:, half_size:] - y_pred[:, half_size:])))
    in_mae = np.mean(np.abs(y_test[:, :half_size] - y_pred[:, :half_size]))
    out_mae = np.mean(np.abs(y_test[:, half_size:] - y_pred[:, half_size:]))
    return in_rmse, out_rmse, in_mae, out_mae


def lstm(x_train, y_train, x_test, y_test):
    half_size = int(x_train.shape[2] / 2)
    early_stop = CustomStopper(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min', start_epoch=5)

    model = models.Sequential(
        [
            tf.keras.layers.LSTM(64, recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(output_size)
        ]
    )

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mse', 'mae'])

    model.fit(x_train, y_train, epochs=500, batch_size=64, validation_split=0.2, callbacks=[early_stop], verbose=verbose)

    in_rmse = []
    out_rmse = []
    in_mae = []
    out_mae = []

    for i in range(12):
        y_pred = model.predict(x_test)
        x_test = np.concatenate([x_test, np.expand_dims(y_pred, axis=1)], axis=1)
        mask = np.greater(y_test[:, i, :], mask_threshold)
        mask.astype(np.float32)
        y_test[:, i, :] *= mask
        y_pred *= mask

        in_rmse.append(np.sqrt(np.mean(np.square(y_test[:, i, :half_size] - y_pred[:, :half_size]))))
        out_rmse.append(np.sqrt(np.mean(np.square(y_test[:, i, half_size:] - y_pred[:, half_size:]))))
        in_mae.append(np.mean(np.abs(y_test[:, i, :half_size] - y_pred[:, :half_size])))
        out_mae.append(np.mean(np.abs(y_test[:, i, half_size:] - y_pred[:, half_size:])))

    for i in range(12):
        print('Slot {} INFLOW_RMSE {:.8f} OUTFLOW_RMSE {:.8f} INFLOW_MAE {:.8f} OUTFLOW_MAE {:.8f}'.format(
            i + 1,
            in_rmse[i],
            out_rmse[i],
            in_mae[i],
            out_mae[i]))
    return in_rmse, out_rmse, in_mae, out_mae


def gru(x_train, y_train, x_test, y_test):
    half_size = int(x_train.shape[2] / 2)
    early_stop = CustomStopper(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min', start_epoch=5)

    model = models.Sequential(
        [
            tf.keras.layers.GRU(64, recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(output_size)
        ]
    )

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mse', 'mae'])

    model.fit(x_train, y_train, epochs=500, batch_size=64, validation_split=0.2, callbacks=[early_stop], verbose=verbose)

    in_rmse = []
    out_rmse = []
    in_mae = []
    out_mae = []

    for i in range(12):
        y_pred = model.predict(x_test)
        x_test = np.concatenate([x_test, np.expand_dims(y_pred, axis=1)], axis=1)
        mask = np.greater(y_test[:, i, :], mask_threshold)
        mask.astype(np.float32)
        y_test[:, i, :] *= mask
        y_pred *= mask

        in_rmse.append(np.sqrt(np.mean(np.square(y_test[:, i, :half_size] - y_pred[:, :half_size]))))
        out_rmse.append(np.sqrt(np.mean(np.square(y_test[:, i, half_size:] - y_pred[:, half_size:]))))
        in_mae.append(np.mean(np.abs(y_test[:, i, :half_size] - y_pred[:, :half_size])))
        out_mae.append(np.mean(np.abs(y_test[:, i, half_size:] - y_pred[:, half_size:])))

    for i in range(12):
        print('Slot {} INFLOW_RMSE {:.8f} OUTFLOW_RMSE {:.8f} INFLOW_MAE {:.8f} OUTFLOW_MAE {:.8f}'.format(
            i + 1,
            in_rmse[i],
            out_rmse[i],
            in_mae[i],
            out_mae[i]))
    return in_rmse, out_rmse, in_mae, out_mae


if __name__ == "__main__":
    for i in range(10):
        print("\nTest Round {}\n".format(i + 1))

        start = time.time()
        print("Calculating Historical Average RMSE, MAE results...")
        in_rmse, out_rmse, in_mae, out_mae = historical_average(original_test, targets_test[:, 0, :])
        print("Historical average: IN_RMSE {}, OUT_RMSE {}, IN_MAE {}, OUT_MAE {}" \
              .format(in_rmse, out_rmse, in_mae, out_mae))
        print("Time used: {} seconds".format(time.time() - start))

        start = time.time()
        print("Calculating ARIMA RMSE, MAE results...")
        in_rmse, out_rmse, in_mae, out_mae = arima(original_test, targets_test[:, 0, :])
        print("ARIMA: IN_RMSE {}, OUT_RMSE {}, IN_MAE {}, OUT_MAE {}" \
              .format(in_rmse, out_rmse, in_mae, out_mae))
        print("Time used: {} seconds".format(time.time() - start))

        start = time.time()
        print("Calculating VAR RMSE, MAE results...")
        in_rmse, out_rmse, in_mae, out_mae = var(original_test, targets_test[:, 0, :])
        print("VAR: IN_RMSE {}, OUT_RMSE {}, IN_MAE {}, OUT_MAE {}" \
              .format(in_rmse, out_rmse, in_mae, out_mae))
        print("Time used: {} seconds".format(time.time() - start))

        start = time.time()
        print("Calculating MLP RMSE, MAE results...")
        in_rmse, out_rmse, in_mae, out_mae = MLP(flow_train, targets_train[:, 0, :], flow_test, targets_test[:, 0, :])
        print("MLP: IN_RMSE {}, OUT_RMSE {}, IN_MAE {}, OUT_MAE {}" \
              .format(in_rmse, out_rmse, in_mae, out_mae))
        print("Time used: {} seconds".format(time.time() - start))

        start = time.time()
        print("Calculating LSTM RMSE, MAE results...")
        in_rmse, out_rmse, in_mae, out_mae = lstm(flow_train, targets_train[:, 0, :], flow_test, targets_test)
        print("Time used: {} seconds".format(time.time() - start))

        start = time.time()
        print("Calculating GRU RMSE, MAE results...")
        in_rmse, out_rmse, in_mae, out_mae = gru(flow_train, targets_train[:, 0, :], flow_test, targets_test)
        print("Time used: {} seconds".format(time.time() - start))

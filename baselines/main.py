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

dataset = 'taxi'
verbose = 0
use_all_data = True     # True if want to use the 7 sequences of all 7 previous days (like in STAM_EX),
                        # Flase to use only the data of previous 12 timeslot in current day.

if dataset == 'taxi':
    d_loader = dl.data_loader(dataset)
    mask_threshold = 10 / parameters_nyctaxi.flow_train_max
elif dataset == 'bike':
    d_loader = dl.data_loader(dataset)
    mask_threshold = 10 / parameters_nycbike.flow_train_max

hist_flow_train, hist_ex_train, curr_flow_train, curr_ex_train, next_exs_train, targets_train \
    = d_loader.generate_data()

hist_flow_test, hist_ex_test, curr_flow_test, curr_ex_test, next_exs_test, targets_test \
    = d_loader.generate_data(datatype='test')

flow_train = np.concatenate([hist_flow_train, curr_flow_train], axis=1)
ex_train = np.concatenate([hist_ex_train, curr_ex_train], axis=1)

flow_test = np.concatenate([hist_flow_test, curr_flow_test], axis=1)
ex_test = np.concatenate([hist_ex_test, curr_ex_test], axis=1)

if use_all_data:
    original_train = flow_train
    original_test = flow_test
else:
    original_train = curr_flow_train
    original_test = curr_flow_test

x_rnn_train = original_train.reshape(original_train.shape[0], original_train.shape[1], \
                                      2, int(original_train.shape[2] / 2)).transpose((0, 3, 1, 2)).reshape(-1,
                                                                                                            original_train.shape[
                                                                                                                1], 2)
y_rnn_train = targets_train.reshape(targets_train.shape[0], targets_train.shape[1], \
                                    2, int(targets_train.shape[2] / 2)).transpose((0, 3, 1, 2))[:, :, 0, :].reshape(-1,
                                                                                                                    2)

x_rnn_test = original_test.reshape(original_test.shape[0], original_test.shape[1], \
                                    2, int(original_test.shape[2] / 2)).transpose((0, 3, 1, 2)).reshape(-1,
                                                                                                         original_test.shape[
                                                                                                             1], 2)
y_rnn_test = targets_test.reshape(targets_test.shape[0], targets_test.shape[1], \
                                  2, int(targets_test.shape[2] / 2)).transpose((0, 3, 1, 2))[:, :, 0, :].reshape(-1,
                                                                                                                 2)


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
    early_stop = CustomStopper(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min', start_epoch=5)

    model = models.Sequential(
        [
            layers.Flatten(),
            layers.Dense(10, activation='relu'),
            layers.Dense(10, activation='relu'),
            layers.Dense(2, activation='relu')
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
    in_rmse = np.sqrt(np.mean(np.square(y_test[:, 0] - y_pred[:, 0])))
    out_rmse = np.sqrt(np.mean(np.square(y_test[:, 1] - y_pred[:, 1])))
    in_mae = np.mean(np.abs(y_test[:, 0] - y_pred[:, 0]))
    out_mae = np.mean(np.abs(y_test[:, 1] - y_pred[:, 1]))
    return in_rmse, out_rmse, in_mae, out_mae


def lstm(x_train, y_train, x_test, y_test):

    early_stop = CustomStopper(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min', start_epoch=5)

    model = models.Sequential(
        [
            tf.keras.layers.LSTM(64, recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(2, activation='relu')
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
    in_rmse = np.sqrt(np.mean(np.square(y_test[:, 0] - y_pred[:, 0])))
    out_rmse = np.sqrt(np.mean(np.square(y_test[:, 1] - y_pred[:, 1])))
    in_mae = np.mean(np.abs(y_test[:, 0] - y_pred[:, 0]))
    out_mae = np.mean(np.abs(y_test[:, 1] - y_pred[:, 1]))
    return in_rmse, out_rmse, in_mae, out_mae


def gru(x_train, y_train, x_test, y_test):

    early_stop = CustomStopper(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min', start_epoch=5)

    model = models.Sequential(
        [
            tf.keras.layers.GRU(64, recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(2, activation='relu')
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
    in_rmse = np.sqrt(np.mean(np.square(y_test[:, 0] - y_pred[:, 0])))
    out_rmse = np.sqrt(np.mean(np.square(y_test[:, 1] - y_pred[:, 1])))
    in_mae = np.mean(np.abs(y_test[:, 0] - y_pred[:, 0]))
    out_mae = np.mean(np.abs(y_test[:, 1] - y_pred[:, 1]))
    return in_rmse, out_rmse, in_mae, out_mae


if __name__ == "__main__":
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
    in_rmse, out_rmse, in_mae, out_mae = MLP(x_rnn_train, y_rnn_train, x_rnn_test, y_rnn_test)
    print("MLP: IN_RMSE {}, OUT_RMSE {}, IN_MAE {}, OUT_MAE {}" \
          .format(in_rmse, out_rmse, in_mae, out_mae))
    print("Time used: {} seconds".format(time.time() - start))

    start = time.time()
    print("Calculating LSTM RMSE, MAE results...")
    in_rmse, out_rmse, in_mae, out_mae = lstm(x_rnn_train, y_rnn_train, x_rnn_test, y_rnn_test)
    print("LSTM: IN_RMSE {}, OUT_RMSE {}, IN_MAE {}, OUT_MAE {}" \
          .format(in_rmse, out_rmse, in_mae, out_mae))
    print("Time used: {} seconds".format(time.time() - start))

    start = time.time()
    print("Calculating GRU RMSE, MAE results...")
    in_rmse, out_rmse, in_mae, out_mae = gru(x_rnn_train, y_rnn_train, x_rnn_test, y_rnn_test)
    print("GRU: IN_RMSE {}, OUT_RMSE {}, IN_MAE {}, OUT_MAE {}" \
          .format(in_rmse, out_rmse, in_mae, out_mae))
    print("Time used: {} seconds".format(time.time() - start))

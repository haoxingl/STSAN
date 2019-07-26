from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import time
import argparse

import parameters_nyctaxi
import parameters_nycbike

from utils import create_masks, CustomSchedule, load_dataset
from model import AMEX

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--dataset', type=str, default='taxi', help='default is taxi, bike for bike dataset')

args = arg_parser.parse_args()

loss_object = tf.keras.losses.MeanSquaredError()


def loss_function(real, pred, threshold=None):
    loss_ = loss_object(real, pred)
    if threshold:
        mask = tf.math.greater(real, threshold)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

    return tf.reduce_mean(loss_)


def main():
    if args.dataset == 'taxi':
        output_size = parameters_nyctaxi.output_size
        flow_max = parameters_nyctaxi.flow_train_max
        total_slot = int(parameters_nyctaxi.timeslot_total)
    elif args.dataset == 'bike':
        output_size = parameters_nycbike.output_size
        flow_max = parameters_nycbike.flow_train_max
        total_slot = int(parameters_nycbike.timeslot_total)
    else:
        raise Exception("Dataset should be taxi or bike")

    """ Model hyperparameters """
    num_layers = 2  # 4
    d_model = 64  # 128
    dff = 128  # 512
    dff_ex = 16  # 16
    num_heads = 4  # 4
    dropout_rate = 0.1
    half_size = int(output_size / 2)
    conv_embedding = False

    """ Training settings"""
    save_ckpt = True
    BATCH_SIZE = 64
    MAX_EPOCHS = 500

    """ Data hyperparameters """
    TARGET_SIZE = 5  # number of future slots to predict
    predicted_weather = True  # use weather prediction
    hist_day_num = 7
    hist_day_seq_len = 7
    curr_day_seq_len = 12

    """ Training flags"""
    BEST_MSE = 1.0

    train_dataset, test_dataset = load_dataset(args.dataset, predicted_weather, BATCH_SIZE, TARGET_SIZE + 1, hist_day_num,
                                               hist_day_seq_len, curr_day_seq_len, total_slot, conv_embedding)

    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_mse = tf.keras.metrics.MeanSquaredError(name='train_mse')
    train_rmse = tf.keras.metrics.RootMeanSquaredError(name='train_rmse')
    train_mae = tf.keras.metrics.MeanAbsoluteError(name='train_mae')

    amex = AMEX(num_layers, d_model, num_heads, dff, output_size, dff_ex=dff_ex, total_slot=total_slot,
                rate=dropout_rate,
                conv_embedding=conv_embedding)

    if save_ckpt:
        checkpoint_path = "./checkpoints/amex"

        ckpt = tf.train.Checkpoint(AMEX=amex,
                                   optimizer=optimizer)

        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

    @tf.function
    def train_step(hist_flow, hist_ex, curr_flow, curr_ex, next_exs, tar, threshold=None):
        next_ex_inp = next_exs[:, :-1, :]
        tar_inp = tar[:, :-1, :]
        tar_real = tar[:, 1:, :]

        look_ahead_mask = create_masks(tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = amex(hist_flow,
                                  hist_ex,
                                  curr_flow,
                                  curr_ex,
                                  next_ex_inp,
                                  tar_inp,
                                  training=True,
                                  look_ahead_mask=look_ahead_mask)
            loss = loss_function(tar_real, predictions, threshold)

        gradients = tape.gradient(loss, amex.trainable_variables)
        optimizer.apply_gradients(zip(gradients, amex.trainable_variables))

        train_loss(loss)
        if threshold:
            mask = tf.math.greater(tar_real, threshold)
            mask = tf.cast(mask, dtype=loss.dtype)
            train_mse(tar_real * mask, predictions * mask)
            train_rmse(tar_real * mask, predictions * mask)
            train_mae(tar_real * mask, predictions * mask)
        train_mse(tar_real, predictions)
        train_rmse(tar_real, predictions)
        train_mae(tar_real, predictions)

    """ Start training... """
    for epoch in range(MAX_EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_mse.reset_states()
        train_rmse.reset_states()
        train_mae.reset_states()

        for (batch, (inp, tar)) in enumerate(train_dataset):
            hist_flow = inp['hist_flow']
            hist_ex = inp['hist_ex']
            curr_flow = inp['curr_flow']
            curr_ex = inp['curr_ex']
            next_exs = inp['next_exs']

            train_step(hist_flow, hist_ex, curr_flow, curr_ex, next_exs, tar)

            if batch % 10 == 0:
                print('Epoch {} Batch {} Loss {:.8f} MSE {:.8f} sumed_RMSE {:.8f} sumed_MAE {:.8f}'.format(epoch + 1,
                                                                                                           batch,
                                                                                                           train_loss.result(),
                                                                                                           train_mse.result(),
                                                                                                           train_rmse.result(),
                                                                                                           train_mae.result()))

        print(
            'Epoch {} Loss {:.8f} MSE {:.8f} sumed_RMSE {:.8f} sumed_MAE {:.8f}'.format(epoch + 1,
                                                                                        train_loss.result(),
                                                                                        train_mse.result(),
                                                                                        train_rmse.result(),
                                                                                        train_mae.result()))
        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

        if (epoch + 1) % 5 == 0 and train_mse.result() < BEST_MSE:
            BEST_MSE = train_mse.result()
            last_loss = train_loss.result()
            last_mse = train_mse.result()
            last_rmse = train_rmse.result()
            last_ape = train_mae.result()

        if epoch + 1 > 40 and (epoch + 1) % 5 == 0 and train_mse.result() > BEST_MSE * 1.01:
            print("Early stoping...")
            if save_ckpt:
                ckpt.restore(ckpt_manager.latest_checkpoint)
                print('Checkpoint restored!!')
            print('Train final result: Loss {:.8f} MSE {:.8f} sumed_RMSE {:.8f} sumed_MAE {:.8f}'.format(last_loss,
                                                                                                         last_mse,
                                                                                                         last_rmse,
                                                                                                         last_ape))
            break

        if save_ckpt and (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

    threshold = 10 / flow_max

    in_rmse = [tf.keras.metrics.RootMeanSquaredError() for _ in range(TARGET_SIZE)]
    out_rmse = [tf.keras.metrics.RootMeanSquaredError() for _ in range(TARGET_SIZE)]
    in_mae = [tf.keras.metrics.MeanAbsoluteError() for _ in range(TARGET_SIZE)]
    out_mae = [tf.keras.metrics.MeanAbsoluteError() for _ in range(TARGET_SIZE)]

    for (batch, (inp, tar)) in enumerate(test_dataset):
        hist_flow = inp['hist_flow']
        hist_ex = inp['hist_ex']
        curr_flow = inp['curr_flow']
        curr_ex = inp['curr_ex']
        next_exs = inp['next_exs']

        dec_input = tar[:, 0, :]
        output = tf.expand_dims(dec_input, 1)
        real = tar[:, 1:, :]

        for i in range(TARGET_SIZE):
            next_ex_inp = next_exs[:, :i + 1, :]

            look_ahead_mask = create_masks(output)

            predictions, attention_weights = amex(hist_flow,
                                                  hist_ex,
                                                  curr_flow,
                                                  curr_ex,
                                                  next_ex_inp,
                                                  output,
                                                  training=False,
                                                  look_ahead_mask=look_ahead_mask)

            predictions = predictions[:, -1:, :]

            output = tf.concat([output, predictions], axis=1)

        pred = output[:, 1:, :]
        mask = tf.math.greater(real, threshold)
        mask = tf.cast(mask, dtype=pred.dtype)
        pred_masked = pred * mask
        real_masked = real * mask
        for i in range(TARGET_SIZE):
            in_rmse[i](real_masked[:, i, :half_size], pred_masked[:, i, :half_size])
            out_rmse[i](real_masked[:, i, half_size:], pred_masked[:, i, half_size:])
            in_mae[i](real_masked[:, i, :half_size], pred_masked[:, i, :half_size])
            out_mae[i](real_masked[:, i, half_size:], pred_masked[:, i, half_size:])

    print('Test result:')
    for i in range(TARGET_SIZE):
        print('Slot {} INFLOW_RMSE {:.8f} OUTFLOW_RMSE {:.8f} INFLOW_MAE {:.8f} OUTFLOW_MAE {:.8f}'.format(
            i + 1,
            in_rmse[i].result(),
            out_rmse[i].result(),
            in_mae[i].result(),
            out_mae[i].result()))


if __name__ == "__main__":
    main()

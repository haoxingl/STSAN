from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import time
import argparse

import parameters_nyctaxi
import parameters_nycbike

from utils import create_masks, create_masks_3d, CustomSchedule, load_dataset, evaluate, early_stop_helper
from model import Transformer_Ex

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    num_layers = 1  # 4
    d_model = 64  # 128
    dff = 256  # 512
    dff_ex = 256  # 16
    num_heads = 4  # 4
    dropout_rate = 0.1
    half_size = int(output_size / 2)
    conv_embedding = True
    conv_layers = 1
    print("num_layers: {}, d_model: {}, dff: {}, num_heads: {}, conv_embedding, conv_layers: {}".format(num_layers,
                                                                                                        d_model, dff,
                                                                                                        num_heads,
                                                                                                        conv_embedding,
                                                                                                        conv_layers))

    """ Training settings"""
    save_ckpt = True
    BATCH_SIZE = 16
    MAX_EPOCHS = 500
    verbose_train = 1
    test_period = 1
    earlystop_epoch = 30
    earlystop_patience = 10
    earlystop_threshold = 1.0
    print("BATCH_SIZE: {}, es_epoch: {}, patience: {}".format(BATCH_SIZE, earlystop_epoch, earlystop_patience))

    """ Data hyperparameters """
    TARGET_SIZE = 12  # number of future slots to predict
    predicted_weather = True  # use weather prediction
    hist_day_num = 7
    hist_day_seq_len = 25
    curr_day_seq_len = 12
    print("TARGET_SIZE: {}, hist_day_num: {}, hist_day_seq_len: {}, curr_day_seq_len: {}".format(TARGET_SIZE,
                                                                                                 hist_day_num,
                                                                                                 hist_day_seq_len,
                                                                                                 curr_day_seq_len))

    train_dataset, val_dataset, test_dataset = load_dataset(args.dataset, predicted_weather, BATCH_SIZE,
                                                            TARGET_SIZE + 1,
                                                            hist_day_num,
                                                            hist_day_seq_len, curr_day_seq_len, total_slot,
                                                            conv_embedding)

    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    train_loss = tf.keras.metrics.Mean()
    train_mse = tf.keras.metrics.MeanSquaredError()
    train_rmse = tf.keras.metrics.RootMeanSquaredError()
    train_mae = tf.keras.metrics.MeanAbsoluteError()

    transformer_ex = Transformer_Ex(num_layers, d_model, num_heads, dff, output_size,
                                    seq_len=hist_day_num * hist_day_seq_len + curr_day_seq_len, seq_len_tar=TARGET_SIZE,
                                    dff_ex=dff_ex, total_slot=total_slot,
                                    rate=dropout_rate, conv_embedding=conv_embedding, conv_layers=conv_layers)

    if save_ckpt:
        checkpoint_path = "./checkpoints/transformer_ex_conv__{}_{}_{}_{}_{}".format(num_layers, d_model, num_heads,
                                                                                 conv_layers, BATCH_SIZE)

        ckpt = tf.train.Checkpoint(Transformer_Ex=transformer_ex,
                                   optimizer=optimizer)

        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=earlystop_patience)

    @tf.function
    def train_step(hist_flow, hist_ex, curr_flow, curr_ex, next_exs, tar, conv_embedding=False):
        next_ex_inp = next_exs[:, :-1, :]
        if not conv_embedding:
            tar_inp = tar[:, :-1, :]
            tar_real = tar[:, 1:, :]
        else:
            tar_inp = tar[:, :, :, :-1, :]
            tar_real = tar[:, :, :, 1:, :]

        if not conv_embedding:
            look_ahead_mask = create_masks(tar_inp)
        else:
            look_ahead_mask = create_masks_3d(tar_inp)

        with tf.GradientTape() as tape:
            predictions = transformer_ex(hist_flow,
                                         hist_ex,
                                         curr_flow,
                                         curr_ex,
                                         next_ex_inp,
                                         tar_inp,
                                         training=True,
                                         look_ahead_mask=look_ahead_mask)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer_ex.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer_ex.trainable_variables))

        train_loss(loss)
        train_mse(tar_real, predictions)
        train_rmse(tar_real, predictions)
        train_mae(tar_real, predictions)

    """ Start training... """
    earlystop_flag = False
    earlystop_helper = early_stop_helper(earlystop_patience, test_period, earlystop_epoch, earlystop_threshold)
    for epoch in range(MAX_EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_mse.reset_states()
        train_rmse.reset_states()
        train_mae.reset_states()

        for (batch, (inp, tar)) in enumerate(train_dataset):
            if not conv_embedding:
                hist_flow = inp['hist_flow']
                hist_ex = inp['hist_ex']
                curr_flow = inp['curr_flow']
                curr_ex = inp['curr_ex']
                next_exs = inp['next_exs']
            else:
                hist_flow = inp['flow']
                hist_ex = inp['ex']
                curr_flow = inp['flow']
                curr_ex = inp['ex']
                next_exs = inp['next_exs']

            train_step(hist_flow, hist_ex, curr_flow, curr_ex, next_exs, tar, conv_embedding)

            if (batch + 1) % 10 == 0 and verbose_train and conv_embedding:
                print('Epoch {} Batch {} Loss {:.8f} MSE {:.8f} sumed_RMSE {:.8f} sumed_MAE {:.8f}'.format(epoch + 1,
                                                                                                           batch + 1,
                                                                                                           train_loss.result(),
                                                                                                           train_mse.result(),
                                                                                                           train_rmse.result(),
                                                                                                           train_mae.result()))

        if verbose_train:
            print(
                'Epoch {} Loss {:.8f} MSE {:.8f} sumed_RMSE {:.8f} sumed_MAE {:.8f}'.format(epoch + 1,
                                                                                            train_loss.result(),
                                                                                            train_mse.result(),
                                                                                            train_rmse.result(),
                                                                                            train_mae.result()))
            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

        if epoch >= earlystop_epoch and (epoch + 1) % test_period == 0:
            print("Validation Result: ")
            in_rmse, out_rmse = evaluate(transformer_ex, val_dataset, TARGET_SIZE, flow_max, half_size, verbose=1,
                                         conv_embedding=conv_embedding)
            earlystop_flag = earlystop_helper.check(in_rmse, out_rmse, epoch)

        if earlystop_flag:
            print("Early stoping...")
            if save_ckpt:
                ckpt.restore(ckpt_manager.checkpoints[int(- earlystop_patience / test_period)])
                print('Checkpoint restored!!')
            break

        if epoch >= earlystop_epoch and save_ckpt and (epoch + 1) % test_period == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}\n'.format(epoch + 1, ckpt_save_path))

    print("Final Test Result: ")
    _, _ = evaluate(transformer_ex, test_dataset, TARGET_SIZE, flow_max, half_size, conv_embedding=conv_embedding)


if __name__ == "__main__":
    main()

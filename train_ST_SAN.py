from __future__ import absolute_import, division, print_function, unicode_literals

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
gpu_id = "0, 1, 2, 3, 4, 5, 6, 7"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

import time
import argparse

import parameters_nyctaxi
import parameters_nycbike

from utils import CustomSchedule, load_dataset, early_stop_helper
from model import Stream_T, ST_SAN

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--dataset', type=str, default='taxi', help='default is taxi, bike for bike dataset')

args = arg_parser.parse_args()


def main():
    if args.dataset == 'taxi':
        flow_max = parameters_nyctaxi.flow_train_max
        total_slot = int(parameters_nyctaxi.time_interval_total)
    elif args.dataset == 'bike':
        flow_max = parameters_nycbike.flow_train_max
        total_slot = int(parameters_nycbike.time_interval_total)
    else:
        raise Exception("Dataset should be taxi or bike")

    direct_test = False

    """ Model hyperparameters """
    num_layers = 4  # number of encoder and decoder layers
    d_model = 64  # dimension of representation space
    dff = 128  # dimension of feed forward network
    d_final = 256  # dimension of the dense layer in the masked fusion layer
    num_heads = 8  # number of attention head
    dropout_rate = 0.1
    cnn_layers = 3
    cnn_filters = 64
    print("num_layers: {}, d_model: {}, dff: {}, num_heads: {}, cnn_layers: {}, cnn_filters: {}" \
          .format(num_layers, d_model, dff, num_heads, cnn_layers, cnn_filters))

    """ Training settings"""
    load_saved_data = True
    save_ckpt = True
    BATCH_SIZE = 128
    MAX_EPOCHS = 500
    verbose_train = 1
    test_period = 1
    earlystop_epoch_stream_f = 10
    earlystop_epoch = 20
    earlystop_patience_stream_f = 10
    earlystop_patience = 20
    earlystop_threshold = 1.0
    start_from_ckpt = None
    lr_exp = 1
    warmup_steps = 4000
    print("BATCH_SIZE: {}, es_epoch: {}, patience: {}".format(BATCH_SIZE, earlystop_epoch, earlystop_patience))

    """ Data hyperparameters """
    num_weeks_hist = 0
    num_days_hist = 7
    num_intervals_hist = 3
    num_intervals_currday = 1
    num_intervals_before_predict = 1
    num_intervals_enc = num_days_hist * num_intervals_hist + num_intervals_currday
    local_block_len = 3
    print(
        "num_weeks_hist: {}, num_days_hist: {}, num_intervals_hist: {}, num_intervals_currday: {}, num_intervals_before_predict: {}" \
        .format(num_weeks_hist, num_days_hist, num_intervals_hist, num_intervals_currday, num_intervals_before_predict))

    def result_writer(str):
        with open("results/ST-SAN.txt", 'a+') as file:
            file.write(str)

    """ mirror strategy of distributed training """
    strategy = tf.distribute.MirroredStrategy()
    print('Number of GPU devices: {}'.format(strategy.num_replicas_in_sync))

    GLOBAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync

    train_dataset, val_dataset, test_dataset = \
        load_dataset(args.dataset,
                     load_saved_data,
                     GLOBAL_BATCH_SIZE,
                     num_weeks_hist,
                     num_days_hist,
                     num_intervals_hist,
                     num_intervals_currday,
                     num_intervals_before_predict,
                     local_block_len)

    train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    val_dataset = strategy.experimental_distribute_dataset(val_dataset)
    test_dataset = strategy.experimental_distribute_dataset(test_dataset)

    with strategy.scope():

        loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        def loss_function(real, pred):
            loss_ = loss_object(real, pred)
            return tf.nn.compute_average_loss(loss_, global_batch_size=GLOBAL_BATCH_SIZE)

        train_in_rmse = tf.keras.metrics.RootMeanSquaredError(name='train_in_rmse')
        train_out_rmse = tf.keras.metrics.RootMeanSquaredError(name='train_out_rmse')
        train_in_mae = tf.keras.metrics.MeanAbsoluteError(name='train_in_mae')
        train_out_mae = tf.keras.metrics.MeanAbsoluteError(name='train_out_mae')

        in_rmse = tf.keras.metrics.RootMeanSquaredError(name='in_rmse')
        out_rmse = tf.keras.metrics.RootMeanSquaredError(name='out_rmse')
        in_mae = tf.keras.metrics.MeanAbsoluteError(name='in_mae')
        out_mae = tf.keras.metrics.MeanAbsoluteError(name='out_mae')

        learning_rate = CustomSchedule(d_model, lr_exp, warmup_steps)

        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        stream_t_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        stream_t = Stream_T(num_layers,
                            d_model,
                            num_heads,
                            dff,
                            cnn_layers,
                            cnn_filters,
                            4,
                            num_intervals_enc,
                            dropout_rate)

        print('Loading checkpoints...')
        stream_t_checkpoint_path = "./checkpoints/stream_t"

        stream_t_ckpt = tf.train.Checkpoint(Stream_T=stream_t, optimizer=stream_t_optimizer)

        stream_t_ckpt_manager = tf.train.CheckpointManager(stream_t_ckpt, stream_t_checkpoint_path,
                                                           max_to_keep=(
                                                                       earlystop_patience_stream_f + earlystop_epoch_stream_f))

        stream_t_ckpt.restore(stream_t_ckpt_manager.checkpoints[int(-1 - earlystop_patience_stream_f / test_period)])

        print('Checkpoints restored...')

        st_san = ST_SAN(stream_t, num_layers, d_model, num_heads, dff, cnn_layers, cnn_filters, num_intervals_enc,
                        d_final, dropout_rate)

        last_epoch = -1

        if save_ckpt:
            checkpoint_path = "./checkpoints/ST-SAN"

            ckpt = tf.train.Checkpoint(ST_SAN=st_san, optimizer=optimizer)

            ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path,
                                                      max_to_keep=(earlystop_patience + earlystop_epoch))

            ckpt_rec_flag = False

            if ckpt_manager.latest_checkpoint:
                ckpt_rec_flag = True
                if start_from_ckpt:
                    ckpt.restore(ckpt_manager.checkpoints[start_from_ckpt - 1])
                    last_epoch = start_from_ckpt
                elif len(ckpt_manager.checkpoints) >= earlystop_epoch + earlystop_patience:
                    ckpt.restore(ckpt_manager.checkpoints[int(-1 - earlystop_patience / test_period)])
                    last_epoch = len(ckpt_manager.checkpoints) - earlystop_patience + 1
                elif len(ckpt_manager.checkpoints) > earlystop_epoch:
                    ckpt.restore(ckpt_manager.checkpoints[earlystop_epoch])
                    last_epoch = earlystop_epoch + 1
                else:
                    ckpt.restore(ckpt_manager.checkpoints[len(ckpt_manager.checkpoints) - 1])
                    last_epoch = len(ckpt_manager.checkpoints)
                print('Latest checkpoint restored!! At epoch {}'.format(last_epoch))

        def train_step(inp, tar):
            flow_hist = inp["flow_hist"]
            trans_hist = inp["trans_hist"]
            ex_hist = inp["ex_hist"]
            flow_currday = inp["flow_currday"]
            trans_currday = inp["trans_currday"]
            ex_currday = inp["ex_currday"]

            ys = tar["ys"]

            with tf.GradientTape() as tape:
                predictions, _ = st_san(flow_hist,
                                        trans_hist,
                                        ex_hist,
                                        flow_currday,
                                        trans_currday,
                                        ex_currday,
                                        training=True)
                loss = loss_function(ys, predictions)

            gradients = tape.gradient(loss, st_san.trainable_variables)
            optimizer.apply_gradients(zip(gradients, st_san.trainable_variables))

            train_in_rmse(ys[:, 0], predictions[:, 0])
            train_out_rmse(ys[:, 1], predictions[:, 1])
            train_in_mae(ys[:, 0], predictions[:, 0])
            train_out_mae(ys[:, 1], predictions[:, 1])

        def test_step(inp, tar, threshold):
            flow_hist = inp["flow_hist"]
            trans_hist = inp["trans_hist"]
            ex_hist = inp["ex_hist"]
            flow_currday = inp["flow_currday"]
            trans_currday = inp["trans_currday"]
            ex_currday = inp["ex_currday"]

            ys = tar["ys"]

            predictions, _ = st_san(flow_hist, trans_hist, ex_hist, flow_currday, trans_currday, ex_currday,
                                    training=False)

            real_in = ys[:, 0]
            real_out = ys[:, 1]
            pred_in = predictions[:, 0]
            pred_out = predictions[:, 1]
            mask_in = tf.where(tf.math.greater(real_in, threshold))
            mask_out = tf.where(tf.math.greater(real_out, threshold))
            masked_real_in = tf.gather_nd(real_in, mask_in)
            masked_real_out = tf.gather_nd(real_out, mask_out)
            masked_pred_in = tf.gather_nd(pred_in, mask_in)
            masked_pred_out = tf.gather_nd(pred_out, mask_out)
            in_rmse(masked_real_in, masked_pred_in)
            out_rmse(masked_real_out, masked_pred_out)
            in_mae(masked_real_in, masked_pred_in)
            out_mae(masked_real_out, masked_pred_out)

        @tf.function
        def distributed_test_step(inp, tar, threshold):
            strategy.experimental_run_v2(test_step, args=(inp, tar, threshold,))

        def evaluate(test_dataset, flow_max, epoch, verbose=1):
            threshold = 10 / flow_max

            in_rmse.reset_states()
            out_rmse.reset_states()
            in_mae.reset_states()
            out_mae.reset_states()

            for (batch, (inp, tar)) in enumerate(test_dataset):

                distributed_test_step(inp, tar, threshold)

                if verbose and (batch + 1) % 100 == 0:
                    print(
                        'Epoch {} ValBatch {} INFLOW_RMSE {:.6f} OUTFLOW_RMSE {:.6f} INFLOW_MAE {:.6f} OUTFLOW_MAE {:.6f}'.format(
                            epoch + 1,
                            batch + 1,
                            in_rmse.result(),
                            out_rmse.result(),
                            in_mae.result(),
                            out_mae.result()))

            if verbose:
                template = 'Epoch {} Total: INFLOW_RMSE {:.6f} OUTFLOW_RMSE {:.6f} INFLOW_MAE {:.6f} OUTFLOW_MAE {:.6f}\n'.format(
                    epoch + 1,
                    in_rmse.result(),
                    out_rmse.result(),
                    in_mae.result(),
                    out_mae.result())
                result_writer(template)
                print(template)

            return in_rmse.result(), out_rmse.result()

        @tf.function
        def distributed_train_step(inp, tar):
            strategy.experimental_run_v2(train_step, args=(inp, tar,))

        if direct_test:
            print("Final Test Result: ")
            _, _ = evaluate(test_dataset, flow_max, -2)

        """ Start training... """
        if not direct_test:
            earlystop_flag = False
            skip_flag = False
            earlystop_helper = early_stop_helper(earlystop_patience, test_period, earlystop_epoch, earlystop_threshold)
            for epoch in range(MAX_EPOCHS):

                if ckpt_rec_flag and (epoch + 1) < last_epoch:
                    skip_flag = True
                    continue

                start = time.time()

                train_in_rmse.reset_states()
                train_out_rmse.reset_states()
                train_in_mae.reset_states()
                train_out_mae.reset_states()

                for (batch, (inp, tar)) in enumerate(train_dataset):
                    if skip_flag:
                        break

                    distributed_train_step(inp, tar)

                    if (batch + 1) % 100 == 0 and verbose_train:
                        print('Epoch {} Batch {} in_RMSE {:.6f} out_RMSE {:.6f} in_MAE {:.6f} out_MAE {:.6f}'.format(
                            epoch + 1,
                            batch + 1,
                            train_in_rmse.result(),
                            train_out_rmse.result(),
                            train_in_mae.result(),
                            train_out_mae.result()))

                if not skip_flag and verbose_train:
                    template = 'Epoch {} in_RMSE {:.6f} out_RMSE {:.6f} in_MAE {:.6f} out_MAE {:.6f}'.format(
                        epoch + 1,
                        train_in_rmse.result(),
                        train_out_rmse.result(),
                        train_in_mae.result(),
                        train_out_mae.result())
                    print(template)
                    result_writer(template + '\n')
                    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

                if (epoch + 1) > earlystop_epoch and (epoch + 1) % test_period == 0:
                    print("Validation Result: ")
                    in_rmse_value, out_rmse_value = evaluate(val_dataset, flow_max, epoch)
                    earlystop_flag = earlystop_helper.check(in_rmse_value, out_rmse_value, epoch)
                    print("Best epoch {}\n".format(earlystop_helper.get_bestepoch()))
                    result_writer("Best epoch {}\n".format(earlystop_helper.get_bestepoch()))

                if not skip_flag and save_ckpt and epoch % test_period == 0:
                    ckpt_save_path = ckpt_manager.save()
                    print('Saving checkpoint for epoch {} at {}\n'.format(epoch + 1, ckpt_save_path))

                if not skip_flag and earlystop_flag:
                    print("Early stoping...")
                    if save_ckpt:
                        ckpt.restore(ckpt_manager.checkpoints[int(-1 - earlystop_patience / test_period)])
                        print('Checkpoint restored!! At epoch {}'.format(
                            int(epoch + 1 - earlystop_patience / test_period)))
                    break

                skip_flag = False

            print("Final Test Result: ")
            _, _ = evaluate(test_dataset, flow_max, epoch)


if __name__ == "__main__":
    main()

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
from model import Stream_T

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--dataset', type=str, default='taxi', help='default is taxi, bike for bike dataset')

args = arg_parser.parse_args()


def main(model_index):
    if args.dataset == 'taxi':
        flow_max = parameters_nyctaxi.flow_train_max
    elif args.dataset == 'bike':
        flow_max = parameters_nycbike.flow_train_max
    else:
        raise Exception("Dataset should be taxi or bike")

    direct_test = False

    """ Model hyperparameters """
    num_layers = 4
    d_model = 64
    dff = 128
    num_heads = 8
    dropout_rate = 0.1
    cnn_layers = 3
    cnn_filters = 64
    print("num_layers: {}, d_model: {}, dff: {}, num_heads: {}, cnn_layers: {}, cnn_filters: {}" \
          .format(num_layers, d_model, dff, num_heads, cnn_layers, cnn_filters))

    """ Training settings"""
    load_saved_data = False
    save_ckpt = True
    BATCH_SIZE = 128
    MAX_EPOCHS = 500
    verbose_train = 1
    test_period = 1
    earlystop_epoch = 20
    earlystop_patience = 10
    earlystop_threshold = 1.0
    last_reshuffle_epoch = 0
    reshuffle_epochs = earlystop_patience
    reshuffle_cnt = 0
    start_from_ckpt = None
    lr_exp = 1
    warmup_steps = 4000
    print("BATCH_SIZE: {}, es_epoch: {}, patience: {}".format(BATCH_SIZE, earlystop_epoch, earlystop_patience))

    """ Data hyperparameters """
    num_weeks_hist = 0
    num_days_hist = 7
    num_intervals_hist = 3
    num_intervals_curr = 1
    num_intervals_before_predict = 1
    num_intervals_enc = (num_weeks_hist + num_days_hist) * num_intervals_hist + num_intervals_curr
    local_block_len = 3
    print(
        "num_weeks_hist: {}, num_days_hist: {}, num_intervals_hist: {}, num_intervals_curr: {}, num_intervals_before_predict: {}" \
            .format(num_weeks_hist, num_days_hist, num_intervals_hist, num_intervals_curr,
                    num_intervals_before_predict))

    def result_writer(str):
        with open("results/stream_t_{}.txt".format(model_index), 'a+') as file:
            file.write(str)

    """ use mirrored strategy for distributed training """
    strategy = tf.distribute.MirroredStrategy()
    print('Number of GPU devices: {}'.format(strategy.num_replicas_in_sync))

    GLOBAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync

    def get_datasets():
        train_dataset, val_dataset, test_dataset = \
            load_dataset(args.dataset,
                         load_saved_data,
                         GLOBAL_BATCH_SIZE,
                         num_weeks_hist,
                         num_days_hist,
                         num_intervals_hist,
                         num_intervals_curr,
                         num_intervals_before_predict,
                         local_block_len)

        train_dataset = strategy.experimental_distribute_dataset(train_dataset)
        val_dataset = strategy.experimental_distribute_dataset(val_dataset)
        test_dataset = strategy.experimental_distribute_dataset(test_dataset)

        return train_dataset, val_dataset, test_dataset

    train_dataset, val_dataset, test_dataset = get_datasets()

    with strategy.scope():

        loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        def loss_function(real, pred):
            loss_ = loss_object(real, pred)
            return tf.nn.compute_average_loss(loss_, global_batch_size=GLOBAL_BATCH_SIZE)

        train_rmse_1 = tf.keras.metrics.RootMeanSquaredError()
        train_rmse_2 = tf.keras.metrics.RootMeanSquaredError()
        train_rmse_3 = tf.keras.metrics.RootMeanSquaredError()
        train_rmse_4 = tf.keras.metrics.RootMeanSquaredError()

        test_rmse_1 = tf.keras.metrics.RootMeanSquaredError()
        test_rmse_2 = tf.keras.metrics.RootMeanSquaredError()
        test_rmse_3 = tf.keras.metrics.RootMeanSquaredError()
        test_rmse_4 = tf.keras.metrics.RootMeanSquaredError()

        learning_rate = CustomSchedule(d_model, lr_exp, warmup_steps)

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

        last_epoch = -1

        if save_ckpt:
            checkpoint_path = "./checkpoints/stream_t_{}".format(model_index)

            ckpt = tf.train.Checkpoint(Stream_T=stream_t, stream_t_optimizer=stream_t_optimizer)

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
            x_hist = inp["trans_hist"]
            ex_hist = inp["ex_hist"]
            x_curr = inp["trans_curr"]
            ex_curr = inp["ex_curr"]

            ys_transitions = tar["ys_transitions"]

            with tf.GradientTape() as tape:
                predictions, _ = stream_t(x_hist,
                                          ex_hist,
                                          x_curr,
                                          ex_curr,
                                          training=True)
                loss = loss_function(ys_transitions, predictions)

            gradients = tape.gradient(loss, stream_t.trainable_variables)
            stream_t_optimizer.apply_gradients(zip(gradients, stream_t.trainable_variables))

            train_rmse_1(ys_transitions[:, :, :, 0], predictions[:, :, :, 0])
            train_rmse_2(ys_transitions[:, :, :, 1], predictions[:, :, :, 1])
            train_rmse_3(ys_transitions[:, :, :, 2], predictions[:, :, :, 2])
            train_rmse_4(ys_transitions[:, :, :, 3], predictions[:, :, :, 3])

        def test_step(inp, tar, threshold):
            x_hist = inp["trans_hist"]
            ex_hist = inp["ex_hist"]
            x_curr = inp["trans_curr"]
            ex_curr = inp["ex_curr"]

            ys_transitions = tar["ys_transitions"]

            predictions, _ = stream_t(x_hist, ex_hist, x_curr, ex_curr, training=False)

            """ here we filter out all nodes where their real flows are less than 10 """
            real_1 = ys_transitions[:, :, :, 0]
            real_2 = ys_transitions[:, :, :, 1]
            real_3 = ys_transitions[:, :, :, 2]
            real_4 = ys_transitions[:, :, :, 3]
            pred_1 = predictions[:, :, :, 0]
            pred_2 = predictions[:, :, :, 1]
            pred_3 = predictions[:, :, :, 2]
            pred_4 = predictions[:, :, :, 3]
            mask_1 = tf.where(tf.math.greater(real_1, threshold))
            mask_2 = tf.where(tf.math.greater(real_2, threshold))
            mask_3 = tf.where(tf.math.greater(real_3, threshold))
            mask_4 = tf.where(tf.math.greater(real_4, threshold))
            masked_real_1 = tf.gather_nd(real_1, mask_1)
            masked_real_2 = tf.gather_nd(real_2, mask_2)
            masked_real_3 = tf.gather_nd(real_3, mask_3)
            masked_real_4 = tf.gather_nd(real_4, mask_4)
            masked_pred_1 = tf.gather_nd(pred_1, mask_1)
            masked_pred_2 = tf.gather_nd(pred_2, mask_2)
            masked_pred_3 = tf.gather_nd(pred_3, mask_3)
            masked_pred_4 = tf.gather_nd(pred_4, mask_4)
            test_rmse_1(masked_real_1, masked_pred_1)
            test_rmse_2(masked_real_2, masked_pred_2)
            test_rmse_3(masked_real_3, masked_pred_3)
            test_rmse_4(masked_real_4, masked_pred_4)

        @tf.function
        def distributed_test_step(inp, tar, threshold):
            strategy.experimental_run_v2(test_step, args=(inp, tar, threshold,))

        def evaluate(eval_dataset, flow_max, epoch, verbose=1, testing=False):
            threshold = 10 / flow_max

            test_rmse_1.reset_states()
            test_rmse_2.reset_states()
            test_rmse_3.reset_states()
            test_rmse_4.reset_states()

            for (batch, (inp, tar)) in enumerate(eval_dataset):

                distributed_test_step(inp, tar, threshold)

                if verbose and (batch + 1) % 100 == 0:
                    if not testing:
                        print(
                            'Epoch {} Batch {} RMSE_1 {:.6f} RMSE_2 {:.6f} RMSE_3 {:.6f} RMSE_4 {:.6f}'.format(
                                epoch + 1, batch + 1, test_rmse_1.result(), test_rmse_2.result(), test_rmse_3.result(), test_rmse_4.result()))
                    else:
                        print(
                            'Testing: Batch {} RMSE_1 {:.6f} RMSE_2 {:.6f} RMSE_3 {:.6f} RMSE_4 {:.6f}'.format(
                                batch + 1, test_rmse_1.result(), test_rmse_2.result(),
                                test_rmse_3.result(), test_rmse_4.result()))

            if verbose:
                if not testing:
                    template = 'Epoch {}: RMSE_1 {:.6f} RMSE_2 {:.6f} RMSE_3 {:.6f} RMSE_4 {:.6f}\n'.format(
                        epoch + 1, test_rmse_1.result(), test_rmse_2.result(), test_rmse_3.result(), test_rmse_4.result())
                    result_writer(template)
                    print(template)
                else:
                    template = 'Final results: RMSE_1 {:.6f} RMSE_2 {:.6f} RMSE_3 {:.6f} RMSE_4 {:.6f}\n'.format(
                        test_rmse_1.result(), test_rmse_2.result(), test_rmse_3.result(), test_rmse_4.result())
                    result_writer(template)
                    print(template)

            return test_rmse_1.result(), test_rmse_2.result(), test_rmse_3.result(), test_rmse_4.result()

        @tf.function
        def distributed_train_step(inp, tar):
            strategy.experimental_run_v2(train_step, args=(inp, tar,))

        if direct_test:
            print("Final Test Result: ")
            _, _, _, _ = evaluate(test_dataset, flow_max, -2, testing=True)

        """ Start training... """
        if not direct_test:
            earlystop_flag = False
            skip_flag = False
            earlystop_helper = early_stop_helper(earlystop_patience, test_period, earlystop_epoch, earlystop_threshold, in_weight=0.5, out_weight=0.5)
            for epoch in range(MAX_EPOCHS):

                if reshuffle_cnt < 2 and (epoch - last_reshuffle_epoch) == reshuffle_epochs:
                    train_dataset, val_dataset, test_dataset = get_datasets()

                    last_reshuffle_epoch = epoch
                    reshuffle_epochs = int(reshuffle_epochs * 1.2)
                    reshuffle_cnt += 1

                if ckpt_rec_flag and (epoch + 1) < last_epoch:
                    skip_flag = True
                    continue

                start = time.time()

                train_rmse_1.reset_states()
                train_rmse_2.reset_states()
                train_rmse_3.reset_states()
                train_rmse_4.reset_states()

                for (batch, (inp, tar)) in enumerate(train_dataset):
                    if skip_flag:
                        break

                    distributed_train_step(inp, tar)

                    if (batch + 1) % 100 == 0 and verbose_train:
                        print('Epoch {} Batch {} RMSE_1 {:.6f} RMSE_2 {:.6f} RMSE_3 {:.6f} RMSE_4 {:.6f}'.format(
                            epoch + 1,
                            batch + 1,
                            train_rmse_1.result(),
                            train_rmse_2.result(),
                            train_rmse_3.result(),
                            train_rmse_4.result()))

                if not skip_flag and verbose_train:
                    template = 'Epoch {} RMSE_1 {:.6f} RMSE_2 {:.6f} RMSE_3 {:.6f} RMSE_4 {:.6f}'.format(
                        epoch + 1,
                        train_rmse_1.result(),
                        train_rmse_2.result(),
                        train_rmse_3.result(),
                        train_rmse_4.result())
                    print(template)
                    result_writer(template + '\n')

                if (epoch + 1) > earlystop_epoch and (epoch + 1) % test_period == 0:
                    print("Validation Result: ")
                    rmse_value_1, rmse_value_2, rmse_value_3, rmse_value_4 = evaluate(val_dataset, flow_max, epoch)
                    earlystop_flag = earlystop_helper.check(rmse_value_1 + rmse_value_2, rmse_value_3 + rmse_value_4,
                                                            epoch)
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

                print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

            print("Testing:")
            result_writer("Testing:")
            _, _, _, _ = evaluate(test_dataset, flow_max, epoch, testing=True)


if __name__ == "__main__":
    main('0')

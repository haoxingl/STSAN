from __future__ import absolute_import, division, print_function, unicode_literals

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

import parameters_nyctaxi
import parameters_nycbike

from utils.CustomSchedule import CustomSchedule
from utils.EarlystopHelper import EarlystopHelper
from utils.ReshuffleHelper import ReshuffleHelper
from models import Stream_T, ST_SAN
from utils.utils import DatasetGenerator, write_result
from utils.Metrics import MAE

""" use mirrored strategy for distributed training """
strategy = tf.distribute.MirroredStrategy()
print('Number of GPU devices: {}'.format(strategy.num_replicas_in_sync))


class ModelTrainer:
    def __init__(self, model_index, args):
        assert args.dataset == 'taxi' or args.dataset == 'bike'
        self.model_index = model_index
        self.args = args
        self.GLOBAL_BATCH_SIZE = args.BATCH_SIZE * strategy.num_replicas_in_sync
        self.stream_t = None
        self.st_san = None
        self.dataset_generator = DatasetGenerator(args.dataset,
                                                  self.GLOBAL_BATCH_SIZE,
                                                  args.num_weeks_hist,
                                                  args.num_days_hist,
                                                  args.num_intervals_hist,
                                                  args.num_intervals_curr,
                                                  args.num_intervals_before_predict,
                                                  args.local_block_len)

        if args.dataset == 'taxi':
            self.trans_max = parameters_nyctaxi.trans_train_max
            self.flow_max = parameters_nyctaxi.flow_train_max
            self.earlystop_patiences_1 = [5, args.earlystop_patience_stream_t]
            self.earlystop_patiences_2 = [5, args.earlystop_patience_stsan]
            self.earlystop_thres_1 = 0.02
            self.earlystop_thres_2 = 0.01
            self.reshuffle_thres_stream_t = [0.8, 1.3]
            self.reshuffle_thres_stsan = [0.8, 1.3, 1.7]
        else:
            self.trans_max = parameters_nycbike.trans_train_max
            self.flow_max = parameters_nycbike.flow_train_max
            self.earlystop_patiences_1 = [5, args.earlystop_patience_stream_t]
            self.earlystop_patiences_2 = [5, args.earlystop_patience_stsan]
            self.earlystop_thres_1 = 0.02
            self.earlystop_thres_2 = 0.01
            self.reshuffle_thres_stream_t = [0.8, 1.3]
            self.reshuffle_thres_stsan = [0.8, 1.3, 1.7]

    def train_stream_t(self):

        result_output_path = "results/stream_t/{}.txt".format(self.model_index)

        train_dataset, val_dataset = self.dataset_generator.load_dataset('train', self.args.load_saved_data, strategy)
        test_dataset = self.dataset_generator.load_dataset('test', self.args.load_saved_data, strategy)

        with strategy.scope():

            loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

            def loss_function(real, pred):
                loss_ = loss_object(real, pred)
                return tf.nn.compute_average_loss(loss_, global_batch_size=self.GLOBAL_BATCH_SIZE)

            train_rmse_in_trans_1 = tf.keras.metrics.RootMeanSquaredError(dtype=tf.float32)
            train_rmse_in_trans_2 = tf.keras.metrics.RootMeanSquaredError(dtype=tf.float32)
            train_rmse_out_trans_1 = tf.keras.metrics.RootMeanSquaredError(dtype=tf.float32)
            train_rmse_out_trans_2 = tf.keras.metrics.RootMeanSquaredError(dtype=tf.float32)

            test_rmse_in_trans_1 = tf.keras.metrics.RootMeanSquaredError(dtype=tf.float32)
            test_rmse_in_trans_2 = tf.keras.metrics.RootMeanSquaredError(dtype=tf.float32)
            test_rmse_out_trans_1 = tf.keras.metrics.RootMeanSquaredError(dtype=tf.float32)
            test_rmse_out_trans_2 = tf.keras.metrics.RootMeanSquaredError(dtype=tf.float32)

            learning_rate = CustomSchedule(self.args.d_model, self.args.warmup_steps)

            optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

            self.stream_t = Stream_T(self.args.num_layers,
                                     self.args.d_model,
                                     self.args.num_heads,
                                     self.args.dff,
                                     self.args.cnn_layers,
                                     self.args.cnn_filters,
                                     4,
                                     self.args.num_intervals_enc,
                                     self.args.dropout_rate)

            checkpoint_path = "./checkpoints/stream_t/{}".format(self.model_index)

            ckpt = tf.train.Checkpoint(Stream_T=self.stream_t, optimizer=optimizer)

            ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path,
                                                      max_to_keep=(self.args.earlystop_patience_stream_t + 1))

            if ckpt_manager.latest_checkpoint:
                if len(ckpt_manager.checkpoints) <= self.args.earlystop_patience_stream_t:
                    ckpt.restore(ckpt_manager.latest_checkpoint)
                else:
                    ckpt.restore(ckpt_manager.checkpoints[0])
                print('Latest checkpoint restored!!')

            def train_step(stream_t, inp, tar):
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
                optimizer.apply_gradients(zip(gradients, stream_t.trainable_variables))

                train_rmse_in_trans_1(ys_transitions[:, :, :, 0], predictions[:, :, :, 0])
                train_rmse_in_trans_2(ys_transitions[:, :, :, 1], predictions[:, :, :, 1])
                train_rmse_out_trans_1(ys_transitions[:, :, :, 2], predictions[:, :, :, 2])
                train_rmse_out_trans_2(ys_transitions[:, :, :, 3], predictions[:, :, :, 3])

                return loss

            def test_step(stream_t, inp, tar, threshold):
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
                test_rmse_in_trans_1(masked_real_1, masked_pred_1)
                test_rmse_in_trans_2(masked_real_2, masked_pred_2)
                test_rmse_out_trans_1(masked_real_3, masked_pred_3)
                test_rmse_out_trans_2(masked_real_4, masked_pred_4)

            @tf.function(experimental_relax_shapes=True)
            def distributed_test_step(stream_t, inp, tar, threshold):
                return strategy.experimental_run_v2(test_step, args=(stream_t, inp, tar, threshold,))

            def evaluate(stream_t, eval_dataset, trans_max, epoch, verbose=1, testing=False):
                threshold = 10 / trans_max

                test_rmse_in_trans_1.reset_states()
                test_rmse_in_trans_2.reset_states()
                test_rmse_out_trans_1.reset_states()
                test_rmse_out_trans_2.reset_states()

                for (batch, (inp, tar)) in enumerate(eval_dataset):

                    distributed_test_step(stream_t, inp, tar, threshold)

                    if verbose and (batch + 1) % 100 == 0:
                        if not testing:
                            print(
                                'Epoch {} Batch {} RMSE_IN_1 {:.6f} RMSE_IN_2 {:.6f} RMSE_OUT_1 {:.6f} RMSE_OUT_2 {:.6f}'.format(
                                    epoch + 1,
                                    batch + 1,
                                    test_rmse_in_trans_1.result(),
                                    test_rmse_in_trans_2.result(),
                                    test_rmse_out_trans_1.result(),
                                    test_rmse_out_trans_2.result()))
                        else:
                            print(
                                'Testing: Batch {} RMSE_IN_1 {:.2f}({:.6f}) RMSE_IN_2 {:.2f}({:.6f}) RMSE_OUT_1 {:.2f}({:.6f}) RMSE_OUT_2 {:.2f}({:.6f})'.format(
                                    batch + 1,
                                    test_rmse_in_trans_1.result() * trans_max,
                                    test_rmse_in_trans_1.result(),
                                    test_rmse_in_trans_2.result() * trans_max,
                                    test_rmse_in_trans_2.result(),
                                    test_rmse_out_trans_1.result() * trans_max,
                                    test_rmse_out_trans_1.result(),
                                    test_rmse_out_trans_2.result() * trans_max,
                                    test_rmse_out_trans_2.result()
                                )
                            )

                if verbose:
                    if not testing:
                        template = 'Epoch {} RMSE_IN_1 {:.6f} RMSE_IN_2 {:.6f} RMSE_OUT_1 {:.6f} RMSE_OUT_2 {:.6f}\n\n'.format(
                            epoch + 1,
                            test_rmse_in_trans_1.result(),
                            test_rmse_in_trans_2.result(),
                            test_rmse_out_trans_1.result(),
                            test_rmse_out_trans_2.result())
                        write_result(result_output_path,
                                     "Validation Result (after Min-Max Normalization, filtering out grids with flow less than consideration threshold):\n" + template)
                        print(template)
                    else:
                        template = 'Final results: RMSE_IN_1 {:.2f}({:.6f}) RMSE_IN_2 {:.2f}({:.6f}) RMSE_OUT_1 {:.2f}({:.6f}) RMSE_OUT_2 {:.2f}({:.6f})\n'.format(
                            test_rmse_in_trans_1.result() * trans_max,
                            test_rmse_in_trans_1.result(),
                            test_rmse_in_trans_2.result() * trans_max,
                            test_rmse_in_trans_2.result(),
                            test_rmse_out_trans_1.result() * trans_max,
                            test_rmse_out_trans_1.result(),
                            test_rmse_out_trans_2.result() * trans_max,
                            test_rmse_out_trans_2.result())
                        write_result(result_output_path, template)
                        print(template)

                return test_rmse_in_trans_1.result(), test_rmse_in_trans_2.result(), test_rmse_out_trans_1.result(), test_rmse_out_trans_2.result()

            @tf.function(experimental_relax_shapes=True)
            def distributed_train_step(stream_t, inp, tar):
                per_replica_losses = strategy.experimental_run_v2(train_step, args=(stream_t, inp, tar,))

                return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

            """ Start training... """
            print('\nStart training...\n')
            write_result(result_output_path, "Start training:\n")
            earlystop_flag = False
            check_flag = False
            earlystop_helper = EarlystopHelper(self.earlystop_patiences_1, self.earlystop_thres_1)
            reshuffle_helper = ReshuffleHelper(self.earlystop_patiences_1[1], self.reshuffle_thres_stream_t)
            summary_writer = tf.summary.create_file_writer('./tensorboard/stream_t/{}'.format(self.model_index))
            step_cnt = 0
            for epoch in range(self.args.MAX_EPOCH):

                start = time.time()

                train_rmse_in_trans_1.reset_states()
                train_rmse_in_trans_2.reset_states()
                train_rmse_out_trans_1.reset_states()
                train_rmse_out_trans_2.reset_states()

                for (batch, (inp, tar)) in enumerate(train_dataset):

                    total_loss = distributed_train_step(self.stream_t, inp, tar)

                    step_cnt += 1
                    with summary_writer.as_default():
                        tf.summary.scalar("loss", total_loss, step=step_cnt)

                    if (batch + 1) % 100 == 0 and self.args.verbose_train:
                        print(
                            'Epoch {} Batch {} RMSE_IN_1 {:.6f} RMSE_IN_2 {:.6f} RMSE_OUT_1 {:.6f} RMSE_OUT_2 {:.6f}'.format(
                                epoch + 1,
                                batch + 1,
                                train_rmse_in_trans_1.result(),
                                train_rmse_in_trans_2.result(),
                                train_rmse_out_trans_1.result(),
                                train_rmse_out_trans_2.result()))

                if self.args.verbose_train:
                    template = 'Epoch {} RMSE_IN_1 {:.6f} RMSE_IN_2 {:.6f} RMSE_OUT_1 {:.6f} RMSE_OUT_2 {:.6f}'.format(
                        epoch + 1,
                        train_rmse_in_trans_1.result(),
                        train_rmse_in_trans_2.result(),
                        train_rmse_out_trans_1.result(),
                        train_rmse_out_trans_2.result())
                    print(template)
                    write_result(result_output_path, template + '\n')

                eval_rmse = (
                                    train_rmse_in_trans_1.result() + train_rmse_in_trans_2.result() + train_rmse_out_trans_1.result() + train_rmse_out_trans_2.result()) / 4

                if check_flag == False and earlystop_helper.refresh_status(eval_rmse):
                    check_flag = True

                if check_flag:
                    print(
                        "Validation Result (after Min-Max Normalization, filtering out grids with flow less than consideration threshold): ")
                    rmse_value_1, rmse_value_2, rmse_value_3, rmse_value_4 = evaluate(self.stream_t, val_dataset,
                                                                                      self.trans_max,
                                                                                      epoch)
                    earlystop_flag = earlystop_helper.check(rmse_value_1 + rmse_value_2 + rmse_value_3 + rmse_value_4,
                                                            epoch)

                ckpt_save_path = ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}\n'.format(epoch + 1, ckpt_save_path))

                if earlystop_flag:
                    print("Early stoping...")
                    ckpt.restore(ckpt_manager.checkpoints[0])
                    print('Checkpoint restored!! At epoch {}\n'.format(
                        int(epoch - self.args.earlystop_patience_stream_t + 1)))
                    break

                if reshuffle_helper.check(epoch):
                    train_dataset, val_dataset = self.dataset_generator.load_dataset('train', self.args.load_saved_data,
                                                                                     strategy)

                print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

            print(
                "Start testing (without Min-Max Normalization, filtering out grids with flow less than consideration threshold):")
            write_result(result_output_path,
                         "Start testing (without Min-Max Normalization, filtering out grids with flow less than consideration threshold):\n")
            _, _, _, _ = evaluate(self.stream_t, test_dataset, self.trans_max, epoch, testing=True)

    def train_st_san(self):
        result_output_path = "results/st_san/{}.txt".format(self.model_index)

        train_dataset, val_dataset = self.dataset_generator.load_dataset('train', self.args.load_saved_data, strategy)
        test_dataset = self.dataset_generator.load_dataset('test', self.args.load_saved_data, strategy)

        with strategy.scope():

            loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

            def loss_function(real, pred):
                loss_ = loss_object(real, pred)
                return tf.nn.compute_average_loss(loss_, global_batch_size=self.GLOBAL_BATCH_SIZE)

            train_inflow_rmse = tf.keras.metrics.RootMeanSquaredError(dtype=tf.float32)
            train_outflow_rmse = tf.keras.metrics.RootMeanSquaredError(dtype=tf.float32)
            test_inflow_rmse = tf.keras.metrics.RootMeanSquaredError(dtype=tf.float32)
            test_outflow_rmse = tf.keras.metrics.RootMeanSquaredError(dtype=tf.float32)

            train_inflow_mae = MAE()
            train_outflow_mae = MAE()
            test_inflow_mae = MAE()
            test_outflow_mae = MAE()

            learning_rate = CustomSchedule(self.args.d_model, self.args.warmup_steps)

            optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

            if not self.stream_t:
                self.stream_t = Stream_T(self.args.num_layers,
                                         self.args.d_model,
                                         self.args.num_heads,
                                         self.args.dff,
                                         self.args.cnn_layers,
                                         self.args.cnn_filters,
                                         4,
                                         self.args.num_intervals_enc,
                                         self.args.dropout_rate)

                print('Loading tranied Stream-T...')
                stream_t_checkpoint_path = "./checkpoints/stream_t/{}".format(self.model_index)

                stream_t_ckpt = tf.train.Checkpoint(Stream_T=self.stream_t)

                stream_t_ckpt_manager = tf.train.CheckpointManager(stream_t_ckpt, stream_t_checkpoint_path,
                                                                   max_to_keep=(
                                                                           self.args.earlystop_patience_stream_t + 1))

                stream_t_ckpt.restore(
                    stream_t_ckpt_manager.checkpoints[0]).expect_partial()

                print('Stream-T restored...')

            self.st_san = ST_SAN(self.stream_t, self.args.num_layers,
                                 self.args.d_model,
                                 self.args.num_heads,
                                 self.args.dff,
                                 self.args.cnn_layers,
                                 self.args.cnn_filters,
                                 self.args.num_intervals_enc,
                                 self.args.d_final,
                                 self.args.dropout_rate)

            checkpoint_path = "./checkpoints/st_san/{}".format(self.model_index)

            ckpt = tf.train.Checkpoint(ST_SAN=self.st_san, optimizer=optimizer)

            ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path,
                                                      max_to_keep=(self.args.earlystop_patience_stsan + 1))

            if ckpt_manager.latest_checkpoint:
                if len(ckpt_manager.checkpoints) <= self.args.earlystop_patience_stsan:
                    ckpt.restore(ckpt_manager.checkpoints[-1])
                else:
                    ckpt.restore(ckpt_manager.checkpoints[0])
                print('Latest checkpoint restored!!')

            def train_step(st_san, inp, tar):
                flow_hist = inp["flow_hist"]
                trans_hist = inp["trans_hist"]
                ex_hist = inp["ex_hist"]
                flow_curr = inp["flow_curr"]
                trans_curr = inp["trans_curr"]
                ex_curr = inp["ex_curr"]

                ys = tar["ys"]

                with tf.GradientTape() as tape:
                    predictions, _ = st_san(flow_hist,
                                            trans_hist,
                                            ex_hist,
                                            flow_curr,
                                            trans_curr,
                                            ex_curr,
                                            training=True)
                    loss = loss_function(ys, predictions)

                gradients = tape.gradient(loss, st_san.trainable_variables)
                optimizer.apply_gradients(zip(gradients, st_san.trainable_variables))

                train_inflow_rmse(ys[:, 0], predictions[:, 0])
                train_outflow_rmse(ys[:, 1], predictions[:, 1])
                train_inflow_mae(ys[:, 0], predictions[:, 0])
                train_outflow_mae(ys[:, 1], predictions[:, 1])

                return loss

            def test_step(st_san, inp, tar, threshold):
                flow_hist = inp["flow_hist"]
                trans_hist = inp["trans_hist"]
                ex_hist = inp["ex_hist"]
                flow_curr = inp["flow_curr"]
                trans_curr = inp["trans_curr"]
                ex_curr = inp["ex_curr"]

                ys = tar["ys"]

                predictions, _ = st_san(flow_hist, trans_hist, ex_hist, flow_curr, trans_curr, ex_curr,
                                        training=False)

                """ here we filter out all nodes where their real flows are less than 10 """
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
                test_inflow_rmse(masked_real_in, masked_pred_in)
                test_outflow_rmse(masked_real_out, masked_pred_out)
                test_inflow_mae(masked_real_in, masked_pred_in)
                test_outflow_mae(masked_real_out, masked_pred_out)

            @tf.function(experimental_relax_shapes=True)
            def distributed_test_step(st_san, inp, tar, threshold):
                return strategy.experimental_run_v2(test_step, args=(st_san, inp, tar, threshold,))

            def evaluate(st_san, eval_dataset, flow_max, epoch, verbose=1, testing=False):
                threshold = 10 / flow_max

                test_inflow_rmse.reset_states()
                test_outflow_rmse.reset_states()
                test_inflow_mae.reset_states()
                test_outflow_mae.reset_states()

                for (batch, (inp, tar)) in enumerate(eval_dataset):

                    distributed_test_step(st_san, inp, tar, threshold)

                    if verbose and (batch + 1) % 100 == 0:
                        if not testing:
                            print(
                                "Epoch {} Batch {} INFLOW_RMSE {:.6f} OUTFLOW_RMSE {:.6f} INFLOW_MAE {:.6f} OUTFLOW_MAE {:.6f}".format(
                                    epoch + 1, batch + 1,
                                    test_inflow_rmse.result(),
                                    test_outflow_rmse.result(),
                                    test_inflow_mae.result(),
                                    test_outflow_mae.result()
                                ))
                        else:
                            print(
                                "Testing: Batch {} INFLOW_RMSE {:.2f}({:.6f}) OUTFLOW_RMSE {:.2f}({:.6f}) INFLOW_MAE {:.2f}({:.6f}) OUTFLOW_MAE {:.2f}({:.6f})".format(
                                    batch + 1,
                                    test_inflow_rmse.result() * flow_max,
                                    test_inflow_rmse.result(),
                                    test_outflow_rmse.result() * flow_max,
                                    test_outflow_rmse.result(),
                                    test_inflow_mae.result() * flow_max,
                                    test_inflow_mae.result(),
                                    test_outflow_mae.result() * flow_max,
                                    test_outflow_mae.result()
                                ))

                if verbose:
                    if not testing:
                        template = 'Epoch {} INFLOW_RMSE {:.6f} OUTFLOW_RMSE {:.6f} INFLOW_MAE {:.6f} OUTFLOW_MAE {:.6f}\n\n'.format(
                            epoch + 1, test_inflow_rmse.result(), test_outflow_rmse.result(), test_inflow_mae.result(),
                            test_outflow_mae.result())
                        write_result(result_output_path,
                                     'Validation Result (after Min-Max Normalization, filtering out grids with flow less than consideration threshold):\n' + template)
                        print(template)
                    else:
                        template = 'Final results: INFLOW_RMSE {:.2f}({:.6f}) OUTFLOW_RMSE {:.2f}({:.6f}) INFLOW_MAE {:.2f}({:.6f}) OUTFLOW_MAE {:.2f}({:.6f})\n'.format(
                            test_inflow_rmse.result() * flow_max,
                            test_inflow_rmse.result(),
                            test_outflow_rmse.result() * flow_max,
                            test_outflow_rmse.result(),
                            test_inflow_mae.result() * flow_max,
                            test_inflow_mae.result(),
                            test_outflow_mae.result() * flow_max,
                            test_outflow_mae.result())
                        write_result(result_output_path, template)
                        print(template)

                return test_inflow_rmse.result(), test_outflow_rmse.result()

            @tf.function(experimental_relax_shapes=True)
            def distributed_train_step(st_san, inp, tar):
                per_replica_losses = strategy.experimental_run_v2(train_step, args=(st_san, inp, tar,))

                return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

            """ Start training... """
            print('\nStart training...\n')
            write_result(result_output_path, "Start training:\n")
            earlystop_flag = False
            check_flag = False
            earlystop_helper = EarlystopHelper(self.earlystop_patiences_2, self.earlystop_thres_2, in_weight=0.3,
                                               out_weight=0.7)
            reshuffle_helper = ReshuffleHelper(self.earlystop_patiences_2[1], self.reshuffle_thres_stsan)
            summary_writer = tf.summary.create_file_writer('./tensorboard/st_san/{}'.format(self.model_index))
            step_cnt = 0
            for epoch in range(self.args.MAX_EPOCH):

                start = time.time()

                train_inflow_rmse.reset_states()
                train_outflow_rmse.reset_states()
                train_inflow_mae.reset_states()
                train_outflow_mae.reset_states()

                for (batch, (inp, tar)) in enumerate(train_dataset):

                    total_loss = distributed_train_step(self.st_san, inp, tar)

                    step_cnt += 1
                    with summary_writer.as_default():
                        tf.summary.scalar("loss", total_loss, step=step_cnt)

                    if (batch + 1) % 100 == 0 and self.args.verbose_train:
                        print('Epoch {} Batch {} in_RMSE {:.6f} out_RMSE {:.6f} in_MAE {:.6f} out_MAE {:.6f}'.format(
                            epoch + 1,
                            batch + 1,
                            train_inflow_rmse.result(),
                            train_outflow_rmse.result(),
                            train_inflow_mae.result(),
                            train_outflow_mae.result()))

                if self.args.verbose_train:
                    template = 'Epoch {} in_RMSE {:.6f} out_RMSE {:.6f} in_MAE {:.6f} out_MAE {:.6f}'.format(
                        epoch + 1,
                        train_inflow_rmse.result(),
                        train_outflow_rmse.result(),
                        train_inflow_mae.result(),
                        train_outflow_mae.result())
                    print(template)
                    write_result(result_output_path, template + '\n')

                eval_rmse = (train_inflow_rmse.result() + train_outflow_rmse.result()) / 2

                if check_flag == False and earlystop_helper.refresh_status(eval_rmse):
                    check_flag = True

                if check_flag:
                    print(
                        "Validation Result (after Min-Max Normalization, filtering out grids with flow less than consideration threshold): ")
                    in_rmse_value, out_rmse_value = evaluate(self.st_san, val_dataset, self.flow_max, epoch)
                    earlystop_flag = earlystop_helper.check(in_rmse_value + out_rmse_value, epoch)

                ckpt_save_path = ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}\n'.format(epoch + 1, ckpt_save_path))

                if earlystop_flag:
                    print("Early stoping...")
                    ckpt.restore(ckpt_manager.checkpoints[0])
                    print('Checkpoint restored!! At epoch {}\n'.format(
                        int(epoch - self.args.earlystop_patience_stsan)))
                    break

                if reshuffle_helper.check(epoch):
                    train_dataset, val_dataset = self.dataset_generator.load_dataset('train', self.args.load_saved_data,
                                                                                     strategy)

                print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

            print(
                "Start testing (without Min-Max Normalization, filtering out grids with flow less than consideration threshold):")
            write_result(result_output_path,
                         "Start testing (without Min-Max Normalization, filtering out grids with flow less than consideration threshold):\n")
            _, _ = evaluate(self.st_san, test_dataset, self.flow_max, epoch, testing=True)

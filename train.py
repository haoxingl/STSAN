from __future__ import absolute_import, division, print_function, unicode_literals

import time, os, codecs, json

import numpy as np
from utils.tools import DatasetGenerator, ResultWriter, create_masks, generate_masks
from utils.CustomSchedule import CustomSchedule
from utils.EarlystopHelper import EarlystopHelper
from utils.Metrics import MAE, MAPE
from models import Stream_T, STSAN

import tensorflow as tf

from data_parameters import data_parameters


class TrainModel:
    def __init__(self, model_index, args):

        """ use mirrored strategy for distributed training """
        self.strategy = tf.distribute.MirroredStrategy()
        strategy = self.strategy
        print('Number of GPU devices: {}'.format(strategy.num_replicas_in_sync))

        param = data_parameters[args.dataset]
        self.param = param

        self.model_index = model_index
        if args.test_model:
            args.n_layer = 1
            args.d_model = 8
            args.dff = 32
            args.n_head = 1
            args.conv_layer = 1
            args.conv_filter = 8
            args.n_w = 0
            args.n_d = 1
            args.n_wd_times = 1
            args.n_p = 0
            args.n_before = 0
            args.l_half = 3
        self.args = args
        self.args.l_hist = (args.n_w + args.n_d) * args.n_wd_times + args.n_p
        self.GLOBAL_BATCH_SIZE = args.BATCH_SIZE * strategy.num_replicas_in_sync
        self.dataset_generator = DatasetGenerator(args.dataset,
                                                  self.GLOBAL_BATCH_SIZE,
                                                  args.n_w,
                                                  args.n_d,
                                                  args.n_wd_times,
                                                  args.n_p,
                                                  args.n_before,
                                                  args.l_half,
                                                  args.pre_shuffle,
                                                  args.test_model)

        self.es_patiences = [5, args.es_patience]
        self.es_threshold = args.es_threshold
        self.data_max = param['data_max'][:param['pred_type']]

    def pretrain(self, train_dataset, val_dataset):
        strategy = self.strategy
        args = self.args
        param = self.param
        test_model = args.test_model
        test_threshold_t = 2 / param['data_max'][2]
        result_writer = ResultWriter("results/{}.txt".format(self.model_index))

        test_dataset = None

        def tf_summary_scalar(summary_writer, name, value, step):
            with summary_writer.as_default():
                tf.summary.scalar(name, value, step=step)

        def print_verbose_t(epoch, final_test):
            if final_test:
                template_rmse = "Transition RMSE: {:.2f}({:.6f})\n".format \
                    (rmse_test_t.result() * param['data_max'][2], rmse_test_t.result())
                template = "Final:\n" + template_rmse
                result_writer.write(template)
            else:
                template = "Epoch {} Transition RMSE: {:.6f}\n".format(epoch + 1, rmse_test_t.result())
                result_writer.write('Validation Result (Min-Max Norm, filtering out trivial grids):\n' + template)

        loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        def loss_function(real, pred):
            loss_ = loss_object(real, pred)
            return tf.nn.compute_average_loss(loss_, global_batch_size=self.GLOBAL_BATCH_SIZE)

        rmse_train_t = tf.keras.metrics.RootMeanSquaredError(dtype=tf.float32)
        rmse_test_t = tf.keras.metrics.RootMeanSquaredError(dtype=tf.float32)

        learning_rate = CustomSchedule(args.d_model, args.warmup_steps)

        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        stream_t = Stream_T(
            args.n_layer,
            args.d_model,
            args.n_head,
            args.dff,
            args.conv_layer,
            args.conv_filter,
            args.l_hist,
            args.l_half,
            args.r_d)

        def train_step(enc_ft, enc_ex, dec_ft, dec_ex, y_t):

            enc_inp = enc_ft[..., 2:]
            dec_inp = dec_ft[..., 2:]

            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(enc_inp, dec_inp)

            with tf.GradientTape() as tape:
                pred_t, _ = stream_t(enc_inp, enc_ex, dec_inp, dec_ex, True,
                                     enc_padding_mask, combined_mask, dec_padding_mask)

                loss_t = loss_function(y_t, pred_t)

            gradients = tape.gradient(loss_t, stream_t.trainable_variables)
            optimizer.apply_gradients(zip(gradients, stream_t.trainable_variables))

            rmse_train_t(y_t, pred_t)

            return loss_t

        @tf.function
        def distributed_train_step(enc_ft, enc_ex, dec_ft, dec_ex, y_t):
            per_replica_losses = strategy.run(train_step, args=(enc_ft, enc_ex, dec_ft, dec_ex, y_t,))

            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        def test_step(enc_ft, enc_ex, dec_ft, dec_ex, y_t):
            enc_inp = enc_ft[..., 2:]
            dec_inp = dec_ft[..., 2:]

            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(enc_inp, dec_inp)

            pred_t, _ = stream_t(enc_inp, enc_ex, dec_inp, dec_ex, False,
                                 enc_padding_mask, combined_mask, dec_padding_mask)

            mask = tf.where(tf.math.greater(y_t, test_threshold_t))
            masked_real = tf.gather_nd(y_t, mask)
            masked_pred = tf.gather_nd(pred_t, mask)
            rmse_test_t(masked_real, masked_pred)

        @tf.function
        def distributed_test_step(enc_ft, enc_ex, dec_ft, dec_ex, y_t):
            return strategy.run(test_step, args=(enc_ft, enc_ex, dec_ft, dec_ex, y_t,))

        def evaluate(eval_dataset, epoch, verbose, final_test=False):
            rmse_test_t.reset_states()

            for (batch, (inp, tar)) in enumerate(eval_dataset):
                enc_ft = inp["enc_inp_ft"]
                enc_ex = inp["enc_inp_ex"]
                dec_ft = inp["dec_inp_ft"]
                dec_ex = inp["dec_inp_ex"]

                y_t = tar["y_t"]

                distributed_test_step(
                    enc_ft, enc_ex, dec_ft, dec_ex, y_t)

            if verbose:
                print_verbose_t(epoch, final_test)

        """ Start training... """
        built = False
        es_flag_t = False
        check_flag_t = False
        es_helper_t = EarlystopHelper('t', self.es_patiences, self.es_threshold)
        summary_writer = tf.summary.create_file_writer(
            os.environ['HOME'] + '/tensorboard/stsan/{}'.format(self.model_index))
        step_cnt = 0
        last_epoch = 0

        checkpoint_path = "./checkpoints/stream_t/{}".format(self.model_index)

        ckpt = tf.train.Checkpoint(Stream_T=stream_t, optimizer=optimizer)

        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path,
                                                  max_to_keep=(args.es_patience + 1))

        if os.path.isfile(checkpoint_path + '/ckpt_record.json'):
            with codecs.open(checkpoint_path + '/ckpt_record.json', encoding='utf-8') as json_file:
                ckpt_record = json.load(json_file)

            last_epoch = ckpt_record['epoch']
            es_flag_t = ckpt_record['es_flag_t']
            check_flag_t = ckpt_record['check_flag_t']
            es_helper_t.load_ckpt(checkpoint_path)
            step_cnt = ckpt_record['step_cnt']

            ckpt.restore(ckpt_manager.checkpoints[-1])
            result_writer.write("Check point restored at epoch {}".format(last_epoch))

        result_writer.write("Start training Stream-T...\n")

        for epoch in range(last_epoch, args.MAX_EPOCH + 1):

            if es_flag_t or epoch == args.MAX_EPOCH:
                print("Stream-T: Early stoping...")
                if es_flag_t:
                    ckpt.restore(ckpt_manager.checkpoints[0])
                else:
                    ckpt.restore(ckpt_manager.checkpoints[es_helper_t.get_bestepoch() - epoch - 1])
                print('Checkpoint restored!! At epoch {}\n'.format(es_helper_t.get_bestepoch()))
                break

            start = time.time()

            rmse_train_t.reset_states()

            for (batch, (inp, tar)) in enumerate(train_dataset):

                enc_ft = inp["enc_inp_ft"]
                enc_ex = inp["enc_inp_ex"]
                dec_ft = inp["dec_inp_ft"]
                dec_ex = inp["dec_inp_ex"]

                y_t = tar["y_t"]

                total_loss = distributed_train_step(enc_ft, enc_ex, dec_ft, dec_ex, y_t)

                if not built and args.model_summary:
                    stream_t.summary(print_fn=result_writer.write)
                    built = True

                step_cnt += 1
                tf_summary_scalar(summary_writer, "total_loss_t", total_loss, step_cnt)

                if (batch + 1) % 100 == 0 and args.verbose_train:
                    template = 'Epoch {} Batch {} Transition RMSE: {:.6f}'.format \
                        (epoch + 1, batch + 1, rmse_train_t.result())
                    print(template)

            if args.verbose_train:
                template = 'Epoch {} Transition RMSE: {:.6f}\n'.format(epoch + 1, rmse_train_t.result())
                result_writer.write(template)
                tf_summary_scalar(summary_writer, "rmse_train_transition", rmse_train_t.result(), epoch + 1)

            eval_rmse = float(rmse_train_t.result().numpy())

            if test_model or (not check_flag_t and es_helper_t.refresh_status(eval_rmse)):
                check_flag_t = True

            if check_flag_t:
                evaluate(val_dataset, epoch, 1, False)
                tf_summary_scalar(summary_writer, "rmse_test_transition", rmse_test_t.result(), epoch + 1)
                es_rmse = float(rmse_test_t.result().numpy())
                es_flag_t = es_helper_t.check(es_rmse, epoch)
                tf_summary_scalar(summary_writer, "best_epoch_t", es_helper_t.get_bestepoch(), epoch + 1)
                if args.always_test and (epoch + 1) % args.always_test == 0:
                    if not test_dataset:
                        test_dataset = self.dataset_generator.build_dataset(
                            'test', args.load_saved_data, strategy, args.no_save)
                    result_writer.write("Always Test:")
                    evaluate(test_dataset, epoch, 1, False)

            ckpt_save_path = ckpt_manager.save()
            ckpt_record = {'built': built, 'epoch': epoch + 1, 'best_epoch': es_helper_t.get_bestepoch(),
                           'check_flag_t': check_flag_t, 'es_flag_t': es_flag_t, 'step_cnt': step_cnt}
            ckpt_record = json.dumps(ckpt_record, indent=4)
            with codecs.open(checkpoint_path + '/ckpt_record.json', 'w', 'utf-8') as outfile:
                outfile.write(ckpt_record)
            es_helper_t.save_ckpt(checkpoint_path)
            print('Save checkpoint for epoch {} at {}\n'.format(epoch + 1, ckpt_save_path))

            tf_summary_scalar(summary_writer, "epoch_time_t", time.time() - start, epoch + 1)
            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

            if test_model:
                es_flag_t = True

        result_writer.write("Start testing Stream-T (filtering out trivial grids):")
        test_dataset = self.dataset_generator.build_dataset(
            'test', args.load_saved_data, strategy, args.no_save) if not test_dataset else test_dataset
        evaluate(test_dataset, epoch, 1, True)
        tf_summary_scalar(summary_writer, "rmse_final_transition", rmse_test_t.result(), 1)

        return stream_t, test_dataset

    def train_stsan(self, stream_t, train_dataset, val_dataset, test_dataset):
        strategy = self.strategy
        args = self.args
        param = self.param
        test_model = args.test_model
        test_threshold_f = [param['test_threshold'][i] / self.data_max[i]
                            for i in range(param['pred_type'])]
        pred_type = param['pred_type']
        data_name = param['data_name']
        weights = args.weights
        is_weights = type(weights) is np.ndarray
        result_writer = ResultWriter("results/{}.txt".format(self.model_index))

        def tf_summary_scalar(summary_writer, name, value, step):
            with summary_writer.as_default():
                tf.summary.scalar(name, value, step=step)

        def print_verbose_f(epoch, final_test):
            if final_test:
                template_rmse = "RMSE:\n"
                for i in range(pred_type):
                    template_rmse += '{}: {:.2f}({:.6f})\n'.format(
                        data_name[i],
                        rmse_test_f[i].result() * self.data_max[i],
                        rmse_test_f[i].result()
                    )
                template_mae = "MAE:\n"
                for i in range(pred_type):
                    template_mae += '{}: {:.2f}({:.6f})\n'.format(
                        data_name[i],
                        mae_test[i].result() * self.data_max[i],
                        mae_test[i].result()
                    )
                template_mape = "MAPE:\n"
                for i in range(pred_type):
                    template_mape += '{}: {:.2f}\n'.format(data_name[i], mape_test[i].result())
                template = "Final:\n" + template_rmse + template_mae + template_mape
                result_writer.write(template)
            else:
                template = "Epoch {} RMSE:\n".format(epoch + 1)
                for i in range(pred_type):
                    template += "{}: {:.6f}\n".format(data_name[i], rmse_test_f[i].result())
                result_writer.write('Validation Result (Min-Max Norm, filtering out trivial grids):\n' + template)

        loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        def loss_function(real, pred):
            loss_ = loss_object(real, pred)
            return tf.nn.compute_average_loss(loss_, global_batch_size=self.GLOBAL_BATCH_SIZE)

        rmse_train_f = [tf.keras.metrics.RootMeanSquaredError(dtype=tf.float32) for _ in range(pred_type)]
        rmse_test_f = [tf.keras.metrics.RootMeanSquaredError(dtype=tf.float32) for _ in range(pred_type)]

        mae_test = [MAE() for _ in range(pred_type)]
        mape_test = [MAPE() for _ in range(pred_type)]

        learning_rate = CustomSchedule(args.d_model, args.warmup_steps)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        stsan = STSAN(
            stream_t,
            args.n_layer,
            args.d_model,
            args.n_head,
            args.dff,
            args.conv_layer,
            args.conv_filter,
            args.l_hist,
            args.l_half,
            args.r_d)

        def train_step(enc_ft, enc_ex, dec_ft, dec_ex, y):

            enc_padding_mask, combined_mask, dec_padding_mask = generate_masks(enc_ft, dec_ft)

            with tf.GradientTape() as tape:
                pred_f, _ = stsan(enc_ft, enc_ex, dec_ft, dec_ex, True,
                                  enc_padding_mask, combined_mask, dec_padding_mask)
                loss_f = loss_function(y * weights, pred_f * weights) if is_weights else loss_function(y, pred_f)

            gradients = tape.gradient(loss_f, stsan.trainable_variables)
            optimizer.apply_gradients(zip(gradients, stsan.trainable_variables))

            for i in range(pred_type):
                rmse_train_f[i](y[..., i], pred_f[..., i])

            return loss_f

        @tf.function
        def distributed_train_step(enc_ft, enc_ex, dec_ft, dec_ex, y):
            per_replica_losses = strategy.run(train_step, args=(enc_ft, enc_ex, dec_ft, dec_ex, y,))

            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        def test_step(enc_ft, enc_ex, dec_ft, dec_ex, y, final_test=False):
            enc_padding_mask, combined_mask, dec_padding_mask = generate_masks(enc_ft, dec_ft)

            pred_f, _ = stsan(enc_ft, enc_ex, dec_ft, dec_ex, False,
                              enc_padding_mask, combined_mask, dec_padding_mask)

            for i in range(pred_type):
                real = y[..., i] * (weights[i] if is_weights else 1)
                mask = tf.where(tf.math.greater(real, test_threshold_f[i]))
                masked_real = tf.gather_nd(real, mask)
                masked_pred = tf.gather_nd(pred_f[..., i], mask)
                rmse_test_f[i](masked_real, masked_pred)
                if final_test:
                    mae_test[i](masked_real, masked_pred)
                    mape_test[i](masked_real, masked_pred)

        @tf.function
        def distributed_test_step(enc_ft, enc_ex, dec_ft, dec_ex, y, final_test):
            return strategy.run(test_step, args=(enc_ft, enc_ex, dec_ft, dec_ex, y, final_test,))

        def evaluate(eval_dataset, epoch, verbose, final_test):
            for i in range(pred_type):
                rmse_test_f[i].reset_states()

            for (batch, (inp, tar)) in enumerate(eval_dataset):
                enc_ft = inp["enc_inp_ft"]
                enc_ex = inp["enc_inp_ex"]
                dec_ft = inp["dec_inp_ft"]
                dec_ex = inp["dec_inp_ex"]

                y = tar["y"]

                distributed_test_step(
                    enc_ft, enc_ex, dec_ft, dec_ex, y, final_test)

            if verbose:
                print_verbose_f(epoch, final_test)

        """ Start training... """
        built = False
        es_flag_f = False
        check_flag_f = False
        es_helper_f = EarlystopHelper('f', self.es_patiences, self.es_threshold)
        summary_writer = tf.summary.create_file_writer(
            os.environ['HOME'] + '/tensorboard/stsan/{}'.format(self.model_index))
        step_cnt = 0
        last_epoch = 0

        checkpoint_path = "./checkpoints/stsan/{}".format(self.model_index)

        ckpt = tf.train.Checkpoint(STSAN=stsan, optimizer=optimizer)

        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path,
                                                  max_to_keep=(args.es_patience + 1))

        if os.path.isfile(checkpoint_path + '/ckpt_record.json'):
            with codecs.open(checkpoint_path + '/ckpt_record.json', encoding='utf-8') as json_file:
                ckpt_record = json.load(json_file)

            built = ckpt_record['built']
            last_epoch = ckpt_record['epoch']
            es_flag_f = ckpt_record['es_flag_f']
            check_flag_f = ckpt_record['check_flag_f']
            es_helper_f.load_ckpt(checkpoint_path)
            step_cnt = ckpt_record['step_cnt']

            ckpt.restore(ckpt_manager.checkpoints[-1])
            result_writer.write("Check point restored at epoch {}".format(last_epoch))

        result_writer.write("Start training STSAN...\n")

        for epoch in range(last_epoch, args.MAX_EPOCH + 1):

            if es_flag_f or epoch == args.MAX_EPOCH:
                print("STSAN: Early stoping...")
                if es_flag_f:
                    ckpt.restore(ckpt_manager.checkpoints[0])
                else:
                    ckpt.restore(ckpt_manager.checkpoints[es_helper_f.get_bestepoch() - epoch - 1])
                print('Checkpoint restored!! At epoch {}\n'.format(es_helper_f.get_bestepoch()))
                break

            start = time.time()

            for i in range(pred_type):
                rmse_train_f[i].reset_states()

            for (batch, (inp, tar)) in enumerate(train_dataset):

                enc_ft = inp["enc_inp_ft"]
                enc_ex = inp["enc_inp_ex"]
                dec_ft = inp["dec_inp_ft"]
                dec_ex = inp["dec_inp_ex"]

                y = tar["y"]

                total_loss = distributed_train_step(enc_ft, enc_ex, dec_ft, dec_ex, y)

                if not built and args.model_summary:
                    stsan.summary(print_fn=result_writer.write)
                    built = True

                step_cnt += 1
                tf_summary_scalar(summary_writer, "total_loss_f", total_loss, step_cnt)

                if (batch + 1) % 100 == 0 and args.verbose_train:
                    template = 'Epoch {} Batch {} RMSE:'.format(epoch + 1, batch + 1)
                    for i in range(pred_type):
                        template += ' {} {:.6f}'.format(data_name[i], rmse_train_f[i].result())
                    print(template)

            if args.verbose_train:
                template = ''
                for i in range(pred_type):
                    template += ' {} {:.6f}'.format(data_name[i], rmse_train_f[i].result())
                    tf_summary_scalar(
                        summary_writer, " rmse_train_{}".format(data_name[i]), rmse_train_f[i].result(), epoch + 1)
                template = 'Epoch {} RMSE: {}\n'.format(epoch + 1, template)
                result_writer.write(template)

            eval_rmse = 0.0
            for i in range(pred_type):
                eval_rmse += float(rmse_train_f[i].result().numpy() * (weights[i] if is_weights else 1))

            if test_model or (not check_flag_f and es_helper_f.refresh_status(eval_rmse)):
                check_flag_f = True

            if check_flag_f:
                evaluate(val_dataset, epoch, 1, False)
                es_rmse = [0.0 for _ in range(pred_type)]
                for i in range(pred_type):
                    if is_weights:
                        es_rmse[i] += float(rmse_test_f[i].result().numpy() * weights[i])
                    else:
                        es_rmse[i] += float(rmse_test_f[i].result().numpy())
                    tf_summary_scalar(summary_writer, "rmse_test_{}".format(data_name[i]),
                                      rmse_test_f[i].result(), epoch + 1)
                es_flag_f = es_helper_f.check(es_rmse[0] + es_rmse[1], epoch)
                tf_summary_scalar(summary_writer, "best_epoch_f", es_helper_f.get_bestepoch(), epoch + 1)
                if args.always_test and (epoch + 1) % args.always_test == 0:
                    if not test_dataset:
                        test_dataset = self.dataset_generator.build_dataset(
                            'test', args.load_saved_data, strategy, args.no_save)
                    result_writer.write("Always Test:")
                    evaluate(test_dataset, epoch, 1, False)

            ckpt_save_path = ckpt_manager.save()
            ckpt_record = {'built': built, 'epoch': epoch + 1, 'best_epoch': es_helper_f.get_bestepoch(),
                           'check_flag_f': check_flag_f, 'es_flag_f': es_flag_f, 'step_cnt': step_cnt}
            ckpt_record = json.dumps(ckpt_record, indent=4)
            with codecs.open(checkpoint_path + '/ckpt_record.json', 'w', 'utf-8') as outfile:
                outfile.write(ckpt_record)
            es_helper_f.save_ckpt(checkpoint_path)
            print('Save checkpoint for epoch {} at {}\n'.format(epoch + 1, ckpt_save_path))

            tf_summary_scalar(summary_writer, "epoch_time_f", time.time() - start, epoch + 1)
            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

            if test_model:
                es_flag_f = True

        result_writer.write("Start testing (filtering out trivial grids):")
        test_dataset = self.dataset_generator.build_dataset(
            'test', args.load_saved_data, strategy, args.no_save) if not test_dataset else test_dataset
        evaluate(test_dataset, epoch, 1, True)
        for i in range(pred_type):
            tf_summary_scalar(summary_writer, "rmse_final_{}".format(data_name[i]), rmse_test_f[i].result(), 1)

        return stsan

    def train(self):
        strategy = self.strategy
        args = self.args

        train_dataset = self.dataset_generator.build_dataset('train', args.load_saved_data, strategy, args.no_save)
        val_dataset = self.dataset_generator.build_dataset('val', args.load_saved_data, strategy, args.no_save)

        with self.strategy.scope():
            stream_t, test_dataset = self.pretrain(train_dataset, val_dataset)
            _ = self.train_stsan(stream_t, train_dataset, val_dataset, test_dataset)

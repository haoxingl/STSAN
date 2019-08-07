from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import data_loader as dl


def create_masks(tar):
    size = tf.shape(tar)[1]
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def create_masks_3d(tar):
    size = tf.shape(tar)[-2]
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def load_dataset(dataset='taxi', predicted_weather=True, batch_size=64, target_size=6, hist_day_num=7,
                 hist_day_seq_len=7, curr_day_seq_len=12, total_slot=4320, conv_embedding=False):
    data_loader = dl.data_loader(dataset)
    if not conv_embedding:
        hist_flow_inputs, hist_ex_inputs, curr_flow_inputs, curr_ex_inputs, next_exs, targets = data_loader.generate_data(
            predicted_weather=predicted_weather, conv_embedding=conv_embedding, target_size=target_size,
            hist_day_num=hist_day_num,
            hist_day_seq_len=hist_day_seq_len, curr_day_seq_len=curr_day_seq_len)
    else:
        flow_inputs, ex_inputs, next_exs, targets = data_loader.generate_data(
            predicted_weather=predicted_weather, conv_embedding=conv_embedding, target_size=target_size,
            hist_day_num=hist_day_num,
            hist_day_seq_len=hist_day_seq_len, curr_day_seq_len=curr_day_seq_len)

    if not conv_embedding:
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "hist_flow": hist_flow_inputs,
                    "hist_ex": hist_ex_inputs,
                    "curr_flow": curr_flow_inputs,
                    "curr_ex": curr_ex_inputs,
                    "next_exs": next_exs
                },
                targets
            )
        )
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "flow": flow_inputs,
                    "ex": ex_inputs,
                    "next_exs": next_exs
                },
                targets
            )
        )

    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(total_slot)
    train_set = train_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    if not conv_embedding:
        hist_flow_inputs, hist_ex_inputs, curr_flow_inputs, curr_ex_inputs, next_exs, targets = data_loader.generate_data(
            datatype='val', predicted_weather=predicted_weather, conv_embedding=conv_embedding, target_size=target_size,
            hist_day_num=hist_day_num,
            hist_day_seq_len=hist_day_seq_len, curr_day_seq_len=curr_day_seq_len)
    else:
        flow_inputs, ex_inputs, next_exs, targets = data_loader.generate_data(
            datatype='val', predicted_weather=predicted_weather, conv_embedding=conv_embedding, target_size=target_size,
            hist_day_num=hist_day_num,
            hist_day_seq_len=hist_day_seq_len, curr_day_seq_len=curr_day_seq_len)

    if not conv_embedding:
        val_set = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "hist_flow": hist_flow_inputs,
                    "hist_ex": hist_ex_inputs,
                    "curr_flow": curr_flow_inputs,
                    "curr_ex": curr_ex_inputs,
                    "next_exs": next_exs
                },
                targets
            )
        )
    else:
        val_set = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "flow": flow_inputs,
                    "ex": ex_inputs,
                    "next_exs": next_exs
                },
                targets
            )
        )

    val_set = val_set.cache()
    val_set = val_set.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    if not conv_embedding:
        hist_flow_inputs, hist_ex_inputs, curr_flow_inputs, curr_ex_inputs, next_exs, targets = data_loader.generate_data(
            datatype='test', predicted_weather=predicted_weather, conv_embedding=conv_embedding,
            target_size=target_size,
            hist_day_num=hist_day_num,
            hist_day_seq_len=hist_day_seq_len, curr_day_seq_len=curr_day_seq_len)
    else:
        flow_inputs, ex_inputs, next_exs, targets = data_loader.generate_data(datatype='test',
                                                                              predicted_weather=predicted_weather,
                                                                              conv_embedding=conv_embedding,
                                                                              target_size=target_size,
                                                                              hist_day_num=hist_day_num,
                                                                              hist_day_seq_len=hist_day_seq_len,
                                                                              curr_day_seq_len=curr_day_seq_len)

    if not conv_embedding:
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "hist_flow": hist_flow_inputs,
                    "hist_ex": hist_ex_inputs,
                    "curr_flow": curr_flow_inputs,
                    "curr_ex": curr_ex_inputs,
                    "next_exs": next_exs
                },
                targets
            )
        )
    else:
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "flow": flow_inputs,
                    "ex": ex_inputs,
                    "next_exs": next_exs
                },
                targets
            )
        )

    test_dataset = test_dataset.batch(batch_size)

    return train_set, val_set, test_dataset


def evaluate(model, test_dataset, TARGET_SIZE, flow_max, half_size, verbose=1, conv_embedding=False):
    threshold = 10 / flow_max

    in_rmse = [tf.keras.metrics.RootMeanSquaredError() for _ in range(TARGET_SIZE)]
    out_rmse = [tf.keras.metrics.RootMeanSquaredError() for _ in range(TARGET_SIZE)]
    in_mae = [tf.keras.metrics.MeanAbsoluteError() for _ in range(TARGET_SIZE)]
    out_mae = [tf.keras.metrics.MeanAbsoluteError() for _ in range(TARGET_SIZE)]

    for (batch, (inp, tar)) in enumerate(test_dataset):
        if not conv_embedding:
            hist_flow = inp['hist_flow']
            hist_ex = inp['hist_ex']
            curr_flow = inp['curr_flow']
            curr_ex = inp['curr_ex']
            next_exs = inp['next_exs']

            dec_input = tar[:, 0, :]
            output = tf.expand_dims(dec_input, 1)
            real = tar[:, 1:, :]
        else:
            hist_flow = inp['flow']
            hist_ex = inp['ex']
            curr_flow = inp['flow']
            curr_ex = inp['ex']
            next_exs = inp['next_exs']

            dec_input = tar[:, :, :, 0, :]
            output = tf.expand_dims(dec_input, 3)
            real = tar[:, :, :, 1:, :]

        for i in range(TARGET_SIZE):
            next_ex_inp = next_exs[:, :i + 1, :]

            if conv_embedding:
                tar_size = i + 1
                look_ahead_mask = create_masks_3d(output)
            else:
                tar_size = None
                look_ahead_mask = create_masks(output)

            predictions = model(hist_flow,
                                hist_ex,
                                curr_flow,
                                curr_ex,
                                next_ex_inp,
                                output,
                                training=False,
                                look_ahead_mask=look_ahead_mask,
                                tar_size=tar_size)

            if not conv_embedding:
                predictions = predictions[:, -1:, :]

                output = tf.concat([output, predictions], axis=1)

                pred = output[:, 1:, :]
            else:
                predictions = predictions[:, :, :, -1:, :]

                output = tf.concat([output, predictions], axis=-2)

                pred = output[:, :, :, 1:, :]

        mask = tf.math.greater(real, threshold)
        mask = tf.cast(mask, dtype=pred.dtype)
        pred_masked = pred * mask
        real_masked = real * mask
        if not conv_embedding:
            for i in range(TARGET_SIZE):
                in_rmse[i](real_masked[:, i, :half_size], pred_masked[:, i, :half_size])
                out_rmse[i](real_masked[:, i, half_size:], pred_masked[:, i, half_size:])
                in_mae[i](real_masked[:, i, :half_size], pred_masked[:, i, :half_size])
                out_mae[i](real_masked[:, i, half_size:], pred_masked[:, i, half_size:])
        else:
            for i in range(TARGET_SIZE):
                in_rmse[i](real_masked[:, :, :, i, 0], pred_masked[:, :, :, i, 0])
                out_rmse[i](real_masked[:, :, :, i, 1], pred_masked[:, :, :, i, 1])
                in_mae[i](real_masked[:, :, :, i, 0], pred_masked[:, :, :, i, 0])
                out_mae[i](real_masked[:, :, :, i, 1], pred_masked[:, :, :, i, 1])

    if verbose:
        for i in range(TARGET_SIZE):
            print('Slot {} INFLOW_RMSE {:.8f} OUTFLOW_RMSE {:.8f} INFLOW_MAE {:.8f} OUTFLOW_MAE {:.8f}'.format(
                i + 1,
                in_rmse[i].result(),
                out_rmse[i].result(),
                in_mae[i].result(),
                out_mae[i].result()))

    return in_rmse[0].result() + in_rmse[-1].result(), out_rmse[0].result() + out_rmse[-1].result()


class early_stop_helper():
    def __init__(self, patience, test_period, start_epoch, thres):
        assert patience % test_period == 0
        self.patience = patience / test_period
        self.start_epoch = start_epoch
        self.thres = thres
        self.count = 0
        self.best_rmse = 2000.0

    def check(self, in_rmse, out_rmse, epoch):

        if epoch < self.start_epoch:
            return False

        if (in_rmse + out_rmse) > self.best_rmse * self.thres:
            self.count += 1
        else:
            self.count = 0
            self.best_rmse = in_rmse + out_rmse

        if self.count >= self.patience:
            return True
        else:
            return False

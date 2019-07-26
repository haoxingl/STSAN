from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import data_loader as dl


def create_masks(tar):
    size = tf.shape(tar)[1]
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


def load_dataset(dataset='taxi', predicted_weather=True, batch_size=64, target_size=5, hist_day_num=7,
                 hist_day_seq_len=7, curr_day_seq_len=12, total_slot=4320, conv_embedding=False):
    data_loader = dl.data_loader(dataset)
    hist_flow_inputs, hist_ex_inputs, curr_flow_inputs, curr_ex_inputs, next_exs, targets = data_loader.generate_data(
        predicted_weather=predicted_weather, conv_embedding=conv_embedding, target_size=target_size,
        hist_day_num=hist_day_num,
        hist_day_seq_len=hist_day_seq_len, curr_day_seq_len=curr_day_seq_len)

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

    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(total_slot).batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    hist_flow_inputs, hist_ex_inputs, curr_flow_inputs, curr_ex_inputs, next_exs, targets = data_loader.generate_data(
        datatype='test', predicted_weather=predicted_weather, conv_embedding=conv_embedding, target_size=target_size,
        hist_day_num=hist_day_num,
        hist_day_seq_len=hist_day_seq_len, curr_day_seq_len=curr_day_seq_len)

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

    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, test_dataset

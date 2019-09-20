from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import data_loader as dl


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, lr_exp=1, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.lr_exp = lr_exp
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2) * self.lr_exp


def load_dataset(dataset='taxi', load_saved_data=False, batch_size=64, num_weeks_hist=0, num_days_hist=7,
                 num_intervals_hist=3, num_intervals_curr=1, num_intervals_before_predict=1, local_block_len=3):
    data_loader = dl.data_loader(dataset)
    flow_inputs_hist, transition_inputs_hist, ex_inputs_hist, flow_inputs_curr, transition_inputs_curr, \
    ex_inputs_curr, ys_transitions, ys = \
        data_loader.generate_data('train',
                                  num_weeks_hist,
                                  num_days_hist,
                                  num_intervals_hist,
                                  num_intervals_curr,
                                  num_intervals_before_predict,
                                  local_block_len,
                                  load_saved_data)

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                "flow_hist": flow_inputs_hist,
                "trans_hist": transition_inputs_hist,
                "ex_hist": ex_inputs_hist,
                "flow_curr": flow_inputs_curr,
                "trans_curr": transition_inputs_curr,
                "ex_curr": ex_inputs_curr
            },
            {
                "ys_transitions": ys_transitions,
                "ys": ys
            }
        )
    )

    train_size = int(0.8 * flow_inputs_hist.shape[0])

    dataset = dataset.cache()
    dataset = dataset.shuffle(flow_inputs_hist.shape[0], reshuffle_each_iteration=False)
    train_set = dataset.take(train_size).shuffle(train_size)
    val_set = dataset.skip(train_size)
    train_set = train_set.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    val_set = val_set.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    flow_inputs_hist, transition_inputs_hist, ex_inputs_hist, flow_inputs_curr, transition_inputs_curr, \
    ex_inputs_curr, ys_transitions, ys = \
        data_loader.generate_data('test',
                                  num_weeks_hist,
                                  num_days_hist,
                                  num_intervals_hist,
                                  num_intervals_curr,
                                  num_intervals_before_predict,
                                  local_block_len,
                                  load_saved_data)

    test_set = tf.data.Dataset.from_tensor_slices(
        (
            {
                "flow_hist": flow_inputs_hist,
                "trans_hist": transition_inputs_hist,
                "ex_hist": ex_inputs_hist,
                "flow_curr": flow_inputs_curr,
                "trans_curr": transition_inputs_curr,
                "ex_curr": ex_inputs_curr
            },
            {
                "ys_transitions": ys_transitions,
                "ys": ys
            }
        )
    )

    test_set = test_set.batch(batch_size)

    return train_set, val_set, test_set


class early_stop_helper():
    def __init__(self, patience, test_period, start_epoch, thres, in_weight=0.4, out_weight=0.6):
        assert patience % test_period == 0
        self.patience = patience / test_period
        self.start_epoch = start_epoch
        self.thres = thres
        self.count = 0
        self.best_rmse = 2000.0
        self.best_in = 2000.0
        self.best_out = 2000.0
        self.best_epoch = -1
        self.in_weight = in_weight
        self.out_weight = out_weight

    def check(self, in_rmse, out_rmse, epoch):

        if epoch < self.start_epoch:
            return False

        if (self.in_weight * in_rmse + self.out_weight * out_rmse) > self.best_rmse * self.thres:
            self.count += 1
        else:
            self.count = 0
            self.best_rmse = self.in_weight * in_rmse + self.out_weight * out_rmse
            self.best_in = in_rmse
            self.best_out = out_rmse
            self.best_epoch = epoch + 1

        if self.count >= self.patience:
            return True
        else:
            return False

    def get_bestepoch(self):
        return self.best_epoch

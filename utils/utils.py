from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from utils.DataLoader import DataLoader as dl

class DatasetGenerator:
    def __init__(self, dataset='taxi', batch_size=64, num_weeks_hist=0, num_days_hist=7,
                     num_intervals_hist=3, num_intervals_curr=1, num_intervals_before_predict=1, local_block_len=3):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_weeks_hist = num_weeks_hist
        self.num_days_hist = num_days_hist
        self.num_intervals_hist = num_intervals_hist
        self.num_intervals_curr = num_intervals_curr
        self.num_intervals_before_predict = num_intervals_before_predict
        self.local_block_len = local_block_len
        self.train_data_loaded = False
        self.test_data_loaded = False
        self.data_loader = dl(self.dataset)

    def load_dataset(self, datatype='train', load_saved_data=False, strategy=None):
        assert datatype == 'train' or datatype == 'test'
        if datatype == 'train':
            if not self.train_data_loaded:
                self.train_data_loaded = True
                flow_inputs_hist, transition_inputs_hist, ex_inputs_hist, flow_inputs_curr, transition_inputs_curr, \
                ex_inputs_curr, ys_transitions, ys = \
                    self.data_loader.generate_data(datatype,
                                                   self.num_weeks_hist,
                                                   self.num_days_hist,
                                                   self.num_intervals_hist,
                                                   self.num_intervals_curr,
                                                   self.num_intervals_before_predict,
                                                   self.local_block_len,
                                                   load_saved_data)

                self.train_dataset = tf.data.Dataset.from_tensor_slices(
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

                self.data_size = int(flow_inputs_hist.shape[0])
                self.train_size = int(self.data_size * 0.8)

            dataset_cached = self.train_dataset.cache()
            dataset_shuffled = dataset_cached.shuffle(self.data_size, reshuffle_each_iteration=False)
            train_set = dataset_shuffled.take(self.train_size).shuffle(self.train_size)
            val_set = dataset_shuffled.skip(self.train_size)
            train_set = train_set.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
            val_set = val_set.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

            if strategy:
                return strategy.experimental_distribute_dataset(train_set), strategy.experimental_distribute_dataset(val_set)
            else:
                return train_set, val_set

        else:
            if not self.test_data_loaded:
                self.test_data_loaded = True
                flow_inputs_hist, transition_inputs_hist, ex_inputs_hist, flow_inputs_curr, transition_inputs_curr, \
                ex_inputs_curr, ys_transitions, ys = \
                    self.data_loader.generate_data(datatype,
                                              self.num_weeks_hist,
                                              self.num_days_hist,
                                              self.num_intervals_hist,
                                              self.num_intervals_curr,
                                              self.num_intervals_before_predict,
                                              self.local_block_len,
                                              load_saved_data)

                self.test_set = tf.data.Dataset.from_tensor_slices(
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

                self.test_set = self.test_set.batch(self.batch_size)

            if strategy:
                return strategy.experimental_distribute_dataset(self.test_set)
            else:
                return self.test_set


def write_result(path, str):
    with open(path, 'a+') as file:
        file.write(str)

from __future__ import absolute_import, division, print_function, unicode_literals

from utils.DataLoader import DataLoader as dl
import tensorflow as tf


class DatasetGenerator:
    def __init__(self, dataset='taxi', batch_size=64, n_w=0, n_d=7, n_wd_times=3, n_p=1, n_before=1,
                 l_half=3, pre_shuffle=True, test_model=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_w = n_w
        self.n_d = n_d
        self.n_wd_times = n_wd_times
        self.n_p = n_p
        self.n_before = n_before
        self.l_half = l_half
        self.pre_shuffle = pre_shuffle
        self.test_model = test_model

        self.val_set = None

    def load_data(self, datatype, load_saved_data=False, no_save=False):
        data_loader = dl(self.dataset, self.l_half, self.pre_shuffle, self.test_model)
        enc_inp_ft, enc_inp_ex, dec_inp_ft, dec_inp_ex, y_t, y = data_loader.generate_data(
            datatype,
            self.n_w,
            self.n_d,
            self.n_wd_times,
            self.n_p,
            self.n_before,
            load_saved_data,
            no_save
        )

        if self.pre_shuffle and datatype == 'train':
            train_set = tf.data.Dataset.from_tensor_slices(
                (
                    {
                        "enc_inp_ft": enc_inp_ft[0],
                        "enc_inp_ex": enc_inp_ex[0],
                        "dec_inp_ft": dec_inp_ft[0],
                        "dec_inp_ex": dec_inp_ex[0]
                    },
                    {
                        "y_t": y_t[0],
                        "y": y[0]
                    }
                )
            )

            val_set = tf.data.Dataset.from_tensor_slices(
                (
                    {
                        "enc_inp_ft": enc_inp_ft[1],
                        "enc_inp_ex": enc_inp_ex[1],
                        "dec_inp_ft": dec_inp_ft[1],
                        "dec_inp_ex": dec_inp_ex[1]
                    },
                    {
                        "y_t": y_t[1],
                        "y": y[1]
                    }
                )
            )

            return [train_set, val_set], enc_inp_ft[0].shape[0]
        else:
            dataset = tf.data.Dataset.from_tensor_slices(
                (
                    {
                        "enc_inp_ft": enc_inp_ft,
                        "enc_inp_ex": enc_inp_ex,
                        "dec_inp_ft": dec_inp_ft,
                        "dec_inp_ex": dec_inp_ex
                    },
                    {
                        "y_t": y_t,
                        "y": y
                    }
                )
            )

            return dataset, enc_inp_ft.shape[0]

    def build_dataset(self, datatype='train', load_saved_data=False, strategy=None, no_save=None):
        assert datatype in ['train', 'val', 'test']

        if datatype == 'val' and self.pre_shuffle:
            assert self.val_set
            pass
        else:
            dataset, data_size = self.load_data(datatype, load_saved_data, no_save)

        if datatype == 'train':
            if not self.pre_shuffle:
                dataset_out = dataset.shuffle(data_size).batch(self.batch_size).cache().prefetch(
                    tf.data.experimental.AUTOTUNE)
            else:
                self.val_set = dataset[1]
                dataset_out = dataset[0].shuffle(data_size).batch(self.batch_size).cache().prefetch(
                    tf.data.experimental.AUTOTUNE)
        elif datatype == 'val':
            if not self.pre_shuffle:
                dataset_out = dataset.batch(self.batch_size) \
                    .cache().prefetch(tf.data.experimental.AUTOTUNE)
            else:
                dataset_out = self.val_set.batch(self.batch_size) \
                    .cache().prefetch(tf.data.experimental.AUTOTUNE)
        else:
            if self.batch_size == 1:
                dataset_out = dataset.shuffle(data_size).batch(self.batch_size)
            else:
                dataset_out = dataset.batch(self.batch_size)

        return strategy.experimental_distribute_dataset(dataset_out) if strategy else dataset_out


class ResultWriter:
    def __init__(self, path):
        self.path = path

    def write(self, str, print_str=True):
        if print_str:
            print(str)
        with open(self.path, 'a+') as file:
            file.write(str + '\n')


def create_padding_mask(inp):
    oup = tf.math.reduce_sum(inp, axis=-1)
    mask = tf.cast(tf.math.equal(oup, 0), tf.float32)
    return mask


def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)[:, tf.newaxis, :, :, tf.newaxis, :]
    dec_padding_mask = create_padding_mask(inp)[:, tf.newaxis, :, :, tf.newaxis, :]

    dec_target_padding_mask = create_padding_mask(tar)[:, tf.newaxis, :, :, tf.newaxis, :]

    return enc_padding_mask, dec_target_padding_mask, dec_padding_mask


def generate_masks(inp_ft, tar_ft):
    enc_padding_mask_f, combined_mask_f, dec_padding_mask_f = create_masks(inp_ft[..., :2], tar_ft[..., :2])
    enc_padding_mask_t, combined_mask_t, dec_padding_mask_t = create_masks(inp_ft[..., 2:], tar_ft[..., 2:])

    return [enc_padding_mask_f, enc_padding_mask_t], [combined_mask_f, combined_mask_t], [dec_padding_mask_f, dec_padding_mask_t]


if __name__ == "__main__":
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '4'

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    dg = DatasetGenerator()
    a = dg.build_dataset(load_saved_data=False)
    b = dg.build_dataset(datatype='val', load_saved_data=False)
    c = dg.build_dataset(datatype='test', load_saved_data=False)

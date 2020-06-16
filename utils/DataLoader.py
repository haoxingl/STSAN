import numpy as np
from data_parameters import data_parameters


class DataLoader:
    def __init__(self, dataset='taxi', l_half=3, pre_shuffle=True, test_model=None):
        assert dataset in ['taxi', 'bike']
        self.dataset = dataset
        self.pmt = data_parameters[dataset]
        self.l_half = l_half
        self.pre_shuffle = pre_shuffle
        self.test_model = test_model

    def load_data(self, datatype='train'):
        pred_type = self.pmt['pred_type']
        if datatype == 'train':
            data = np.load(self.pmt['data_train'])
        elif datatype == 'val':
            data = np.load(self.pmt['data_val'])
        else:
            data = np.load(self.pmt['data_test'])

        self.data_mtx = np.array(data['flow'], dtype=np.float32) / np.array(self.pmt['data_max'][:pred_type],
                                                                            dtype=np.float32)
        self.t_mtx = np.array(data['trans'], dtype=np.float32) / np.array(self.pmt['data_max'][2], dtype=np.float32)
        self.ex_mtx = data['ex_knlg']

    def generate_data(self, datatype='train', n_w=0, n_d=7, n_wd_times=3, n_p=1, n_before=1,
                      load_saved_data=False, no_save=False):

        assert datatype in ['train', 'val', 'test']

        """ loading saved data """
        if load_saved_data and not self.test_model:
            enc_inp_ft = np.load("data/enc_inp_ft_{}_{}.npz".format(self.dataset, datatype))['data']
            enc_inp_ex = np.load("data/enc_inp_ex_{}_{}.npz".format(self.dataset, datatype))['data']
            dec_inp_ft = np.load("data/dec_inp_ft_{}_{}.npz".format(self.dataset, datatype))['data']
            dec_inp_ex = np.load("data/dec_inp_ex_{}_{}.npz".format(self.dataset, datatype))['data']
            y = np.load("data/y_{}_{}.npz".format(self.dataset, datatype))['data']
            y_t = np.load("data/y_t_{}_{}.npz".format(self.dataset, datatype))['data']
        else:
            print("Loading {} data...".format(datatype))
            """ loading data """
            self.load_data(datatype)

            data_mtx = self.data_mtx
            ex_mtx = self.ex_mtx
            t_mtx = self.t_mtx
            data_shape = data_mtx.shape

            l_half = self.l_half
            if l_half:
                l_full = 2 * l_half + 1

            """ initialize the array to hold the final inputs """

            enc_inp_ft = []
            enc_inp_ex = []

            dec_inp_ft = []
            dec_inp_ex = []

            y = []
            y_t = []

            assert n_w >= 0 and n_d >= 0 and n_d <= 7
            """ set the start time interval to sample the data"""
            s1 = n_d * self.pmt['n_int'] + n_before
            s2 = n_w * 7 * self.pmt['n_int'] + n_before
            time_start = max(s1, s2)
            time_end = data_shape[0]

            for t in range(time_start, time_end):
                if (t - time_start + 1) % 100 == 0:
                    print("Loading {}/{}".format(t - time_start + 1, time_end - time_start))

                for r in range(data_shape[1]):
                    for c in range(data_shape[2]):

                        """ initialize the array to hold the samples of each node at each time interval """

                        enc_inp_ft_sample = []
                        enc_inp_ex_sample = []

                        if l_half:
                            """ initialize the boundaries of the area of interest """
                            r_start = r - l_half  # the start location of each AoI
                            c_start = c - l_half

                            """ adjust the start location if it is on the boundaries of the grid map """
                            if r_start < 0:
                                r_start_l = 0 - r_start
                                r_start = 0
                            else:
                                r_start_l = 0
                            if c_start < 0:
                                c_start_l = 0 - c_start
                                c_start = 0
                            else:
                                c_start_l = 0

                            r_end = r + l_half + 1  # the end location of each AoI
                            c_end = c + l_half + 1
                            if r_end >= data_shape[1]:
                                r_end_l = l_full - (r_end - data_shape[1])
                                r_end = data_shape[1]
                            else:
                                r_end_l = l_full
                            if c_end >= data_shape[2]:
                                c_end_l = l_full - (c_end - data_shape[2])
                                c_end = data_shape[2]
                            else:
                                c_end_l = l_full

                        """ start the samplings of previous weeks """
                        t_hist = []

                        for week_cnt in range(n_w):
                            s_time_w = int(t - (n_w - week_cnt) * 7 * self.pmt['n_int'] - n_before)

                            for int_cnt in range(n_wd_times):
                                t_hist.append(s_time_w + int_cnt)

                        """ start the samplings of previous days"""
                        for hist_day_cnt in range(n_d):
                            """ define the start time in previous days """
                            s_time_d = int(t - (n_d - hist_day_cnt) * self.pmt['n_int'] - n_before)

                            """ generate samples from the previous days """
                            for int_cnt in range(n_wd_times):
                                t_hist.append(s_time_d + int_cnt)

                        """ sampling of inputs of current day, the details are similar to those mentioned above """
                        for int_cnt in range(n_p):
                            t_hist.append(t - n_p + int_cnt)

                        for t_now in t_hist:
                            if not l_half:
                                one_inp = data_mtx[t_now, ...]

                                one_inp_t = np.zeros((data_shape[1], data_shape[2], 4), dtype=np.float32)
                                one_inp_t[..., 0] = t_mtx[t_now, ..., r, c, 0]
                                one_inp_t[..., 1] = t_mtx[t_now, ..., r, c, 1]
                                one_inp_t[..., 2] = t_mtx[t_now, r, c, ..., 0]
                                one_inp_t[..., 3] = t_mtx[t_now, r, c, ..., 1]
                            else:
                                one_inp = np.zeros((l_full, l_full, 2), dtype=np.float32)
                                one_inp[r_start_l:r_end_l, c_start_l:c_end_l, :] = \
                                    data_mtx[t_now, r_start:r_end, c_start:c_end, :]

                                one_inp_t = np.zeros((l_full, l_full, 4), dtype=np.float32)
                                one_inp_t[r_start_l:r_end_l, c_start_l:c_end_l, 0] = \
                                    t_mtx[t_now, r_start:r_end, c_start:c_end, r, c, 0]
                                one_inp_t[r_start_l:r_end_l, c_start_l:c_end_l, 1] = \
                                    t_mtx[t_now, r_start:r_end, c_start:c_end, r, c, 1]
                                one_inp_t[r_start_l:r_end_l, c_start_l:c_end_l, 2] = \
                                    t_mtx[t_now, r, c, r_start:r_end, c_start:c_end, 0]
                                one_inp_t[r_start_l:r_end_l, c_start_l:c_end_l, 3] = \
                                    t_mtx[t_now, r, c, r_start:r_end, c_start:c_end, 1]

                            enc_inp_ft_sample.append(np.concatenate([one_inp, one_inp_t], axis=-1))
                            enc_inp_ex_sample.append(ex_mtx[t_now, :])

                        enc_inp_ft.append(enc_inp_ft_sample)
                        enc_inp_ex.append(enc_inp_ex_sample)

                        if not l_half:
                            dec_inp_f_sample = data_mtx[t - 1:t, ..., :]

                            dec_inp_t_sample = np.zeros((1, data_shape[1], data_shape[2], 4), dtype=np.float32)
                            dec_inp_t_sample[..., 0] = t_mtx[t - 1:t, ..., r, c, 0]
                            dec_inp_t_sample[..., 1] = t_mtx[t - 1:t, ..., r, c, 1]
                            dec_inp_t_sample[..., 2] = t_mtx[t - 1:t, r, c, ..., 0]
                            dec_inp_t_sample[..., 3] = t_mtx[t - 1:t, r, c, ..., 1]

                            tar_t = np.zeros((data_shape[1], data_shape[2], 4), dtype=np.float32)
                            tar_t[..., 0] = t_mtx[t, ..., r, c, 0]
                            tar_t[..., 1] = t_mtx[t, ..., r, c, 1]
                            tar_t[..., 2] = t_mtx[t, r, c, ..., 0]
                            tar_t[..., 3] = t_mtx[t, r, c, ..., 1]
                        else:
                            dec_inp_f_sample = np.zeros((1, l_full, l_full, 2), dtype=np.float32)
                            dec_inp_f_sample[:, r_start_l:r_end_l, c_start_l:c_end_l, :] = \
                                data_mtx[t - 1:t, r_start:r_end, c_start:c_end, :]

                            dec_inp_t_sample = np.zeros((1, l_full, l_full, 4), dtype=np.float32)
                            dec_inp_t_sample[:, r_start_l:r_end_l, c_start_l:c_end_l, 0] = \
                                t_mtx[t - 1:t, r_start:r_end, c_start:c_end, r, c, 0]
                            dec_inp_t_sample[:, r_start_l:r_end_l, c_start_l:c_end_l, 1] = \
                                t_mtx[t - 1:t, r_start:r_end, c_start:c_end, r, c, 1]
                            dec_inp_t_sample[:, r_start_l:r_end_l, c_start_l:c_end_l, 2] = \
                                t_mtx[t - 1:t, r, c, r_start:r_end, c_start:c_end, 0]
                            dec_inp_t_sample[:, r_start_l:r_end_l, c_start_l:c_end_l, 3] = \
                                t_mtx[t - 1:t, r, c, r_start:r_end, c_start:c_end, 1]

                            tar_t = np.zeros((l_full, l_full, 4), dtype=np.float32)
                            tar_t[r_start_l:r_end_l, c_start_l:c_end_l, 0] = \
                                t_mtx[t, r_start:r_end, c_start:c_end, r, c, 0]
                            tar_t[r_start_l:r_end_l, c_start_l:c_end_l, 1] = \
                                t_mtx[t, r_start:r_end, c_start:c_end, r, c, 1]
                            tar_t[r_start_l:r_end_l, c_start_l:c_end_l, 2] = \
                                t_mtx[t, r, c, r_start:r_end, c_start:c_end, 0]
                            tar_t[r_start_l:r_end_l, c_start_l:c_end_l, 3] = \
                                t_mtx[t, r, c, r_start:r_end, c_start:c_end, 1]

                        dec_inp_ft.append(np.concatenate([dec_inp_f_sample, dec_inp_t_sample], axis=-1))
                        dec_inp_ex.append(ex_mtx[t - 1:t, :])
                        y_t.append(tar_t)
                        y.append(data_mtx[t, r, c, :])

                if self.test_model and t + 1 - time_start >= self.test_model:
                    break

            """ convert the inputs arrays to matrices """
            enc_inp_ft = np.array(enc_inp_ft, dtype=np.float32).transpose((0, 2, 3, 1, 4))
            enc_inp_ex = np.array(enc_inp_ex, dtype=np.float32)

            dec_inp_ft = np.array(dec_inp_ft, dtype=np.float32).transpose((0, 2, 3, 1, 4))
            dec_inp_ex = np.array(dec_inp_ex, dtype=np.float32)

            y = np.array(y, dtype=np.float32)
            y_t = np.array(y_t, dtype=np.float32)

            """ save the matrices """
            if not (self.test_model or no_save):
                print('Saving .npz files...')
                np.savez_compressed("data/enc_inp_ft_{}_{}.npz".format(self.dataset, datatype), data=enc_inp_ft)
                np.savez_compressed("data/enc_inp_ex_{}_{}.npz".format(self.dataset, datatype), data=enc_inp_ex)
                np.savez_compressed("data/dec_inp_ft_{}_{}.npz".format(self.dataset, datatype), data=dec_inp_ft)
                np.savez_compressed("data/dec_inp_ex_{}_{}.npz".format(self.dataset, datatype), data=dec_inp_ex)
                np.savez_compressed("data/y_t_{}_{}.npz".format(self.dataset, datatype), data=y_t)
                np.savez_compressed("data/y_{}_{}.npz".format(self.dataset, datatype), data=y)

        if self.pre_shuffle and datatype == 'train':
            inp_shape = enc_inp_ft.shape[0]
            train_size = int(inp_shape * 0.8)
            random_index = np.random.permutation(inp_shape)

            enc_inp_ft = np.split(enc_inp_ft[random_index, ...], (train_size,))
            enc_inp_ex = np.split(enc_inp_ex[random_index, ...], (train_size,))
            dec_inp_ft = np.split(dec_inp_ft[random_index, ...], (train_size,))
            dec_inp_ex = np.split(dec_inp_ex[random_index, ...], (train_size,))

            y = np.split(y[random_index, ...], (train_size,))
            y_t = np.split(y_t[random_index, ...], (train_size,))

        return enc_inp_ft, enc_inp_ex, dec_inp_ft, dec_inp_ex, y_t, y


if __name__ == "__main__":
    dl = DataLoader(64)
    enc_inp_ft, enc_inp_ex, dec_inp_ft, dec_inp_ex, y_t, y = dl.generate_data()

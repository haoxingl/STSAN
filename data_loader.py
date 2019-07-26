import numpy as np
import parameters_nyctaxi as param_taxi
import parameters_nycbike as param_bike


class data_loader:
    def __init__(self, dataset='taxi'):
        self.dataset = dataset
        if self.dataset == 'taxi':
            self.parameters = param_taxi
        elif self.dataset == 'bike':
            self.parameters = param_bike
        else:
            print('Dataset should be \'taxi\' or \'bike\'')
            raise Exception

    def load_flow(self):
        self.flow_train = np.load(self.parameters.flow_train)['flow'] / self.parameters.flow_train_max
        self.flow_test = np.load(self.parameters.flow_test)['flow'] / self.parameters.flow_train_max

    def load_external_knowledge(self):
        self.ex_knlg_train = np.load(self.parameters.external_knowledge_train)['external_knowledge']
        self.ex_knlg_test = np.load(self.parameters.external_knowledge_test)['external_knowledge']

    def generate_data(self, datatype='train', predicted_weather=True, conv_embedding=False, target_size=2, hist_day_num=7,
                      hist_day_seq_len=7, curr_day_seq_len=12):

        self.load_flow()
        self.load_external_knowledge()

        if hist_day_seq_len % 2 != 1:
            print("Previous days attention seq len must be odd")
            raise Exception

        if datatype == "train":
            flow_data = self.flow_train
            ex_knlg = self.ex_knlg_train
        elif datatype == "test":
            flow_data = self.flow_test
            ex_knlg = self.ex_knlg_test
        else:
            print("Please select **train** or **test**")
            raise Exception

        next_exs = []
        targets = []

        hist_flow_inputs = []
        hist_ex_inputs = []

        curr_flow_inputs = []
        curr_ex_inputs = []

        time_start = hist_day_num * self.parameters.timeslot_daily + hist_day_seq_len
        time_end = flow_data.shape[0] - target_size + 1

        for t in range(time_start, time_end):
            if t % 100 == 0:
                print("Currently at {} slot...".format(t))

            hist_flow_inputs_sample = []
            hist_ex_inputs_sample = []

            curr_flow_inputs_sample = []
            curr_ex_inputs_sample = []

            for hist_day_cnt in range(hist_day_num):
                hist_t = int(t - (hist_day_num - hist_day_cnt) * self.parameters.timeslot_daily - (
                        hist_day_seq_len - 1) / 2)

                for hist_seq in range(hist_day_seq_len):
                    if not conv_embedding:
                        hist_flow_inputs_sample.append(flow_data[hist_t + hist_seq, :, :, :].transpose((2, 0, 1)).flatten())
                    else:
                        hist_flow_inputs_sample.append(flow_data[hist_t + hist_seq, :, :, :])
                    hist_ex_inputs_sample.append(ex_knlg[hist_t + hist_seq, :])

            for curr_seq in range(curr_day_seq_len):
                if not conv_embedding:
                    curr_flow_inputs_sample.append(
                        flow_data[int(t - (curr_day_seq_len + curr_seq)), :, :, :].transpose((2, 0, 1)).flatten())
                else:
                    curr_flow_inputs_sample.append(flow_data[int(t - (curr_day_seq_len + curr_seq)), :, :, :])
                curr_ex_inputs_sample.append(ex_knlg[int(t - (curr_day_seq_len + curr_seq)), :])
            curr_flow_inputs.append(np.array(curr_flow_inputs_sample))
            curr_ex_inputs.append(np.array(curr_ex_inputs_sample))
            hist_flow_inputs.append(np.array(hist_flow_inputs_sample))
            hist_ex_inputs.append(np.array(hist_ex_inputs_sample))
            targets.append(flow_data[t:t + target_size, :, :, :].transpose((0, 3, 1, 2)).reshape(target_size, -1))
            next_exs_sample = ex_knlg[t:t + target_size, :]
            if not predicted_weather:
                next_exs_sample[:, 55:] = next_exs_sample[0, 55:]
            next_exs.append(next_exs_sample)

        curr_flow_inputs = np.array(curr_flow_inputs, dtype=np.float32)
        curr_ex_inputs = np.array(curr_ex_inputs, dtype=np.float32)
        hist_flow_inputs = np.array(hist_flow_inputs, dtype=np.float32)
        hist_ex_inputs = np.array(hist_ex_inputs, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        next_exs = np.array(next_exs, dtype=np.float32)

        return hist_flow_inputs, hist_ex_inputs, curr_flow_inputs, curr_ex_inputs, next_exs, targets

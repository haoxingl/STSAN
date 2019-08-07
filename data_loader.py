import numpy as np
import parameters_nyctaxi as param_taxi
import parameters_nycbike as param_bike

# 这个是根据他们的改的

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
        self.flow_train = np.load(self.parameters.flow_train)['flow'] / self.parameters.flow_train_max # 和他们的一样，读了数据后除以一个最大值来正则化
        self.flow_val = np.load(self.parameters.flow_val)['flow'] / self.parameters.flow_train_max
        self.flow_test = np.load(self.parameters.flow_test)['flow'] / self.parameters.flow_train_max

    def load_external_knowledge(self):
        self.ex_knlg_train = np.load(self.parameters.external_knowledge_train)['external_knowledge']    # 外部信息，我加的，他们没有用任何外部信息
        self.ex_knlg_val = np.load(self.parameters.external_knowledge_val)['external_knowledge']
        self.ex_knlg_test = np.load(self.parameters.external_knowledge_test)['external_knowledge']

    # 这一段大部分都是根据他们的逻辑改的，因为模型没他们复杂所以短了很多
    def generate_data(self, datatype='train', predicted_weather=True, # 这个可以不用管
                      conv_embedding=False, # 如果True就会输出原始的矩阵， 否则会将每个矩阵flat之后输出，flat之后每个向量前半部分是流入的值，后半部分是流出的值，True只在我的模型里用到，baselines皆为false
                      target_size=13, # 需要预测未来多少个时间段
                      hist_day_num=7,   # 参考过去多少天的数据
                      hist_day_seq_len=25, # 每天参考多少个时间点的数据
                      curr_day_seq_len=12): # 当天参考多少个时间点的数据

        self.load_flow()
        self.load_external_knowledge()

        if hist_day_seq_len % 2 != 1:
            print("Previous days attention seq len must be odd")
            raise Exception

        # 选择数据集类型
        if datatype == "train":
            flow_data = self.flow_train
            ex_knlg = self.ex_knlg_train
        elif datatype == "val":
            flow_data = self.flow_val
            ex_knlg = self.ex_knlg_val
        elif datatype == "test":
            flow_data = self.flow_test
            ex_knlg = self.ex_knlg_test
        else:
            print("Please select **train** or **val** or **test**")
            raise Exception

        next_exs = []
        targets = []

        hist_flow_inputs = []
        hist_ex_inputs = []

        curr_flow_inputs = []
        curr_ex_inputs = []

        time_start = hist_day_num * self.parameters.timeslot_daily + hist_day_seq_len # 确定开始时间，预留足够空间给前几天的数据提取
        time_end = flow_data.shape[0] - target_size + 1 # 确定结束时间，预留足够空间给未来多个时间点的数据提取

        for t in range(time_start, time_end):
            if t % 100 == 0:
                print("Currently at {} slot...".format(t))

            hist_flow_inputs_sample = []
            hist_ex_inputs_sample = []

            curr_flow_inputs_sample = []
            curr_ex_inputs_sample = []

            for hist_day_cnt in range(hist_day_num):
                hist_t = int(t - (hist_day_num - hist_day_cnt) * self.parameters.timeslot_daily - (
                        hist_day_seq_len - 1) / 2) # 计算当天的开始时间

                for hist_seq in range(hist_day_seq_len):
                    if not conv_embedding:
                        # 将流量矩阵转置之后flat，flat之后前半为流入量后半为流出量
                        hist_flow_inputs_sample.append(flow_data[hist_t + hist_seq, :, :, :].transpose((2, 0, 1)).flatten())
                    else:
                        hist_flow_inputs_sample.append(flow_data[hist_t + hist_seq, :, :, :])
                    hist_ex_inputs_sample.append(ex_knlg[hist_t + hist_seq, :]) # 外部信息

            # 逻辑同上
            for curr_seq in range(curr_day_seq_len):
                if not conv_embedding:
                    curr_flow_inputs_sample.append(
                        flow_data[int(t - (curr_day_seq_len - curr_seq)), :, :, :].transpose((2, 0, 1)).flatten())
                else:
                    curr_flow_inputs_sample.append(flow_data[int(t - (curr_day_seq_len - curr_seq)), :, :, :])
                curr_ex_inputs_sample.append(ex_knlg[int(t - (curr_day_seq_len - curr_seq)), :])
            curr_flow_inputs.append(np.array(curr_flow_inputs_sample))
            curr_ex_inputs.append(np.array(curr_ex_inputs_sample))
            hist_flow_inputs.append(np.array(hist_flow_inputs_sample))
            hist_ex_inputs.append(np.array(hist_ex_inputs_sample))
            if not conv_embedding:
                targets.append(flow_data[t:t + target_size, :, :, :].transpose((0, 3, 1, 2)).reshape(target_size, -1))
            else:
                targets.append(flow_data[t:t + target_size, :, :, :])
            next_exs_sample = ex_knlg[t:t + target_size, :]
            if not predicted_weather:
                next_exs_sample[:, 56:] = next_exs_sample[0, 56:]
            next_exs.append(next_exs_sample)

        curr_flow_inputs = np.array(curr_flow_inputs, dtype=np.float32) # 将列表矩阵化，和他们一样
        curr_ex_inputs = np.array(curr_ex_inputs, dtype=np.float32)
        hist_flow_inputs = np.array(hist_flow_inputs, dtype=np.float32)
        hist_ex_inputs = np.array(hist_ex_inputs, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        next_exs = np.array(next_exs, dtype=np.float32)

        if not conv_embedding:
            return hist_flow_inputs, hist_ex_inputs, curr_flow_inputs, curr_ex_inputs, next_exs, targets
        else:
            flow_inputs = np.concatenate([hist_flow_inputs, curr_flow_inputs], axis=1).transpose((0, 2, 3, 1, 4))
            ex_inputs = np.concatenate([hist_ex_inputs, curr_ex_inputs], axis=1)
            targets = targets.transpose((0, 2, 3, 1, 4))

            return flow_inputs, ex_inputs, next_exs, targets


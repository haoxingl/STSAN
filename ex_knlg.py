import pandas as pd
import numpy as np

class ex_knlg_processor:
    def __init__(self, dataset='taxi'):
        self.dataset = dataset
        if self.dataset == 'taxi':
            self.csv_file = 'data/NYTaxi/external_knowledge.csv'
            self.out_path = 'data/NYTaxi/'
        elif self.dataset == 'bike':
            self.csv_file = 'data/NYBike/external_knowledge.csv'
            self.out_path = 'data/NYBike/'
        else:
            print('arg error')
            raise Exception

    def process(self):
        self.num_days = 60
        self.time_interval_sec = 1800
        self.total_sec = self.num_days * 24 * 60 * 60
        self.total_interval = int(self.total_sec / self.time_interval_sec)
        self.train_size = int(int(self.num_days * 2/3) * 24 * 60 * 60 // self.time_interval_sec)
        self.interval_daily = int(24 * 60 * 60 / self.time_interval_sec)

        ex_matrix = np.zeros([self.total_interval, 65], dtype=np.float32)

        df = pd.read_csv(self.csv_file)

        df.insert(9, "precipitation trace", [0] * df.shape[0], True)
        df.insert(10, "snow fall trace", [0] * df.shape[0], True)
        df.insert(11, "snow depth trace", [0] * df.shape[0], True)

        df = df.astype({'maximum temperature': float, 'minimum temperature': float, 'holiday': float})

        max_max = df['maximum temperature'].max()
        max_min = df['maximum temperature'].min()
        max_delta = max_max - max_min

        min_max = df['minimum temperature'].max()
        min_min = df['minimum temperature'].min()
        min_delta = min_max - min_min

        ave_max = df['average temperature'].max()
        ave_min = df['average temperature'].min()
        ave_delta = ave_max - ave_min

        for idx, row in df.iterrows():

            if row['precipitation'] == 'T':
                df.at[idx, 'precipitation'] = 0.0
                df.at[idx, 'precipitation trace'] = 1
            if row['snow fall'] == 'T':
                df.at[idx, 'snow fall'] = 0.0
                df.at[idx, 'snow fall trace'] = 1
            if row['snow depth'] == 'T':
                df.at[idx, 'snow depth'] = 0.0
                df.at[idx, 'snow depth trace'] = 1

            df.at[idx, 'maximum temperature'] = (df.at[idx, 'maximum temperature'] - max_min) / max_delta
            df.at[idx, 'minimum temperature'] = (df.at[idx, 'minimum temperature'] - min_min) / min_delta
            df.at[idx, 'average temperature'] = (df.at[idx, 'average temperature'] - ave_min) / ave_delta

        df[['precipitation', 'snow fall', 'snow depth']] = df[['precipitation', 'snow fall', 'snow depth']].apply(
            pd.to_numeric)

        p_max = df['precipitation'].max()

        sf_max = df['snow fall'].max()

        sd_max = df['snow depth'].max()

        for idx, row in df.iterrows():

            df.at[idx, 'precipitation'] = df.at[idx, 'precipitation'] / p_max
            if self.dataset == 'taxi':
                df.at[idx, 'snow fall'] = df.at[idx, 'snow fall'] / sf_max
                df.at[idx, 'snow depth'] = df.at[idx, 'snow depth'] / sd_max

        i = 0
        for idx, row in df.iterrows():
            for j in range(self.interval_daily):
                if i >= ex_matrix.shape[0]:
                    break
                if i % 100 == 0:
                    print(self.dataset, i)
                ex_matrix[i, int(row['day'])] = 1
                ex_matrix[i, 7 + j] = 1
                ex_matrix[i, 55] = row['holiday']
                ex_matrix[i, 56] = row['maximum temperature']
                ex_matrix[i, 57] = row['minimum temperature']
                ex_matrix[i, 58] = row['average temperature']
                ex_matrix[i, 59] = row['precipitation']
                ex_matrix[i, 60] = row['snow fall']
                ex_matrix[i, 61] = row['snow depth']
                ex_matrix[i, 62] = row['precipitation trace']
                ex_matrix[i, 63] = row['snow fall trace']
                ex_matrix[i, 64] = row['snow depth trace']
                i += 1

        ex_matrix_train = ex_matrix[:self.train_size, :]
        ex_matrix_test = ex_matrix[self.train_size:, :]

        np.savez_compressed(self.out_path + 'ex_knlg_train.npz', external_knowledge=ex_matrix_train)
        np.savez_compressed(self.out_path + 'ex_knlg_test.npz', external_knowledge=ex_matrix_test)
        np.savez_compressed(self.out_path + 'ex_knlg.npz', external_knowledge=ex_matrix)

if __name__ == "__main__":
    p = ex_knlg_processor(dataset='bike')
    mtx = p.process()
    p = ex_knlg_processor(dataset='taxi')
    mtx = p.process()

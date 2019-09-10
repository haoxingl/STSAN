flow_train = "./data/NYTaxi/flow_train.npz"
flow_test = "./data/NYTaxi/flow_test.npz"
trans_train = "./data/NYTaxi/trans_train.npz"
trans_test = "./data/NYTaxi/trans_test.npz"
external_knowledge_train = "./data/NYTaxi/ex_knlg_train.npz"
external_knowledge_test = "./data/NYTaxi/ex_knlg_test.npz"
flow_train_max = 1518.0
flow_test_max = 1427.0
trans_train_max = 186.0
trans_test_max = 186.0
time_interval_sec = 1800
time_interval_daily = 48
total_day = 60
time_interval_total = total_day * 24 * 60 * 60 / time_interval_sec
loss_threshold = 10

if __name__ == "__main__":
    import numpy as np
    flow_train = np.load(flow_train)['flow']
    flow_test = np.load(flow_test)['flow']
    trans_train = np.load(trans_train)['trans']
    trans_test = np.load(trans_test)['trans']
    print(flow_train.shape[0], "flow_train_max = {}".format(flow_train.max()))
    print(flow_test.shape[0], "flow_test_max = {}".format(flow_test.max()))
    print(trans_train.shape[1], "trans_train_max = {}".format(trans_train.max()))
    print(trans_test.shape[1], "trans_test_max = {}".format(trans_test.max()))
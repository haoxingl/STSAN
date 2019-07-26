flow_train = "./data/NYTaxi/flow_train.npz"
flow_test = "./data/NYTaxi/flow_test.npz"
external_knowledge_train = "./data/NYTaxi/ex_knlg_train.npz"
external_knowledge_test = "./data/NYTaxi/ex_knlg_test.npz"
flow_train_max = 1518.0
flow_test_max = 1483.0
trans_train_max = 143.0
trans_test_max = 186.0
timeslot_sec = 1800
timeslot_daily = 48
total_day = 90
timeslot_total = total_day * 24 * 60 * 60 / timeslot_sec
loss_threshold = 10
output_size = 384

# import numpy as np
# flow_train = np.load("./data/NYTaxi/flow_train.npz")['flow']
# flow_test = np.load("./data/NYTaxi/flow_test.npz")['flow']
# trans_train = np.load("./data/NYTaxi/trans_train.npz")['trans']
# trans_test = np.load("./data/NYTaxi/trans_test.npz")['trans']
# flow_train_max = flow_train.max()
# flow_test_max = flow_test.max()
# trans_train_max = trans_train.max()
# trans_test_max = trans_test.max()
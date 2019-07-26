flow_train = "./data/NYBike/flow_train.npz"
flow_test = "./data/NYBike/flow_test.npz"
external_knowledge_train = "./data/NYBike/ex_knlg_train.npz"
external_knowledge_test = "./data/NYBike/ex_knlg_test.npz"
flow_train_max = 295.0
flow_test_max = 323.0
trans_train_max = 39.0
trans_test_max = 35.0
timeslot_sec = 1800
timeslot_daily = 48
total_day = 90
timeslot_total = total_day * 24 * 60 * 60 / timeslot_sec
loss_threshold = 10
output_size = 224

# import numpy as np
# flow_train = np.load("./data/NYBike/flow_train.npz")['flow']
# flow_test = np.load("./data/NYBike/flow_test.npz")['flow']
# trans_train = np.load("./data/NYBike/trans_train.npz")['trans']
# trans_test = np.load("./data/NYBike/trans_test.npz")['trans']
# flow_train_max = flow_train.max()
# flow_test_max = flow_test.max()
# trans_train_max = trans_train.max()
# trans_test_max = trans_test.max()
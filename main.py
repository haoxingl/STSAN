from __future__ import absolute_import, division, print_function, unicode_literals

""" 'taxi' or 'bike' dataset """
dataset = 'bike'
print("Dataset chosen: {}".format(dataset))
assert dataset == 'taxi' or dataset == 'bike'

from train_models import train_stream_t, train_st_san

if __name__ == "__main__":
    for index in range(1, 2):
        model_index = dataset + '_{}'.format(index)
        print("\nStrat training Stream-T...\n")
        train_stream_t(model_index, dataset)
        print("\nStrat training ST-SAN...\n")
        train_st_san(model_index, dataset)

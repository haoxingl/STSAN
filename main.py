from __future__ import absolute_import, division, print_function, unicode_literals

""" 'taxi' or 'bike' dataset """
dataset = 'taxi'
print("Dataset chosen: {}".format(dataset))

from train_stream_t import main as main_t
from train_ST_SAN import main as main_stsan

assert dataset == 'taxi' or dataset == 'bike'

if __name__ == "__main__":
    for index in range(0, 1):
        model_index = dataset + '_{}'.format(index)
        print("\nStrat training Stream-T...\n")
        main_t(model_index, dataset)
        print("\nStrat training ST-SAN...\n")
        main_stsan(model_index, dataset)

from __future__ import absolute_import, division, print_function, unicode_literals

""" 'taxi' or 'bike' dataset """
dataset = 'taxi'
print("Dataset chosen: {}".format(dataset))
assert dataset == 'taxi' or dataset == 'bike'

from ModelTrainer import ModelTrainer

if __name__ == "__main__":
    for index in range(1, 2):
        print('Model index: {}'.format(index))
        model_index = dataset + '_{}'.format(index)
        model_trainer = ModelTrainer(model_index, dataset)
        print("\nStrat training Stream-T...\n")
        model_trainer.train_stream_t()
        print("\nStrat training ST-SAN...\n")
        model_trainer.train_st_san()

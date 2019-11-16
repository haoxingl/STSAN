from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import shutil

parser = argparse.ArgumentParser(description='Hyperparameters')
parser.add_argument('--dataset', default='taxi', help='taxi or bike')
parser.add_argument('--gpu_ids', default='0, 1, 2, 3, 4, 5, 6, 7', help='indexes of gpus to use')
parser.add_argument('--model_indexes', default=[2, 3], help='indexes of model to be trained')

""" Model hyperparameters """
parser.add_argument('--num_layers', default=4, help='num of self-attention layers')
parser.add_argument('--d_model', default=64, help='model dimension')
parser.add_argument('--dff', default=128, help='dimension of feed-forward networks')
parser.add_argument('--d_final', default=256, help='dimension of final output dense layer')
parser.add_argument('--num_heads', default=8, help='number of attention heads')
parser.add_argument('--dropout_rate', default=0.1)
parser.add_argument('--cnn_layers', default=3)
parser.add_argument('--cnn_filters', default=64)

""" Training settings """
parser.add_argument('--remove_old_files', default=True)
parser.add_argument('--MAX_EPOCH', default=500)
parser.add_argument('--BATCH_SIZE', default=128)
parser.add_argument('--earlystop_patience_stream_t', default=10)
parser.add_argument('--earlystop_patience_stsan', default=15)
parser.add_argument('--warmup_steps', default=4000)
parser.add_argument('--verbose_train', default=1)
parser.add_argument('--in_weight', default=0.4)
parser.add_argument('--out_weight', default=0.6)

""" Data hyperparameters """
num_weeks_hist = 0
num_days_hist = 7
num_intervals_hist = 3
num_intervals_curr = 1
num_intervals_before_predict = 1
num_intervals_enc = (num_weeks_hist + num_days_hist) * num_intervals_hist + num_intervals_curr
parser.add_argument('--load_saved_data', default=True)
parser.add_argument('--num_weeks_hist', default=num_weeks_hist, help='num of previous weeks to consider')
parser.add_argument('--num_days_hist', default=num_days_hist, help='num of previous days to consider')
parser.add_argument('--num_intervals_hist', default=num_intervals_hist, help='num of time in previous days to consider')
parser.add_argument('--num_intervals_curr', default=num_intervals_curr, help='num of time in today to consider')
parser.add_argument('--num_intervals_before_predict', default=1, help='num of time before predicted time to consider')
parser.add_argument('--num_intervals_enc', default=num_intervals_enc, help='total length of historical data')
parser.add_argument('--local_block_len', default=3, help='halved size of local cnn filter')

args = parser.parse_args()

print("num_layers: {}, d_model: {}, dff: {}, num_heads: {}, cnn_layers: {}, cnn_filters: {}" \
      .format(args.num_layers, args.d_model, args.dff, args.num_heads, args.cnn_layers, args.cnn_filters))
print(
    "BATCH_SIZE: {}, earlystop_patience_stream_t: {}, earlystop_patience_stsan: {}".format(
        args.BATCH_SIZE, args.earlystop_patience_stream_t, args.earlystop_patience_stsan))
print(
    "num_weeks_hist: {}, num_days_hist: {}, num_intervals_hist: {}, num_intervals_curr: {}, num_intervals_before_predict: {}, local_block_len: {}" \
        .format(args.num_weeks_hist,
                args.num_days_hist,
                args.num_intervals_hist,
                args.num_intervals_curr,
                args.num_intervals_before_predict,
                args.local_block_len))

# os.environ['F_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

print("Dataset chosen: {}".format(args.dataset))
assert args.dataset == 'taxi' or args.dataset == 'bike'

from ModelTrainer import ModelTrainer

if __name__ == "__main__":
    for index in range(args.model_indexes[0], args.model_indexes[1]):
        model_index = args.dataset + '_{}'.format(index)
        print('Model index: {}'.format(model_index))
        if args.remove_old_files:
            try:
                shutil.rmtree('./checkpoints/stream_t/{}'.format(model_index), ignore_errors=True)

            except:
                pass
            try:
                shutil.rmtree('./checkpoints/st_san/{}'.format(model_index), ignore_errors=True)
            except:
                pass
            try:
                os.remove('./results/stream_t/{}.txt'.format(model_index))
            except:
                pass
            try:
                os.remove('./results/st_san/{}.txt'.format(model_index))
            except:
                pass
            try:
                shutil.rmtree('./tensorboard/stream_t/{}'.format(model_index), ignore_errors=True)
            except:
                pass
            try:
                shutil.rmtree('./tensorboard/st_san/{}'.format(model_index), ignore_errors=True)
            except:
                pass
        model_trainer = ModelTrainer(model_index, args)
        print("\nStrat training Stream-T...\n")
        model_trainer.train_stream_t()
        print("\nStrat training ST-SAN...\n")
        model_trainer.train_st_san()

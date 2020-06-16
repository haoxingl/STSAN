from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import shutil
import numpy as np
from utils.tools import ResultWriter

parser = argparse.ArgumentParser(description='Hyperparameters')
parser.add_argument('--dataset', default='taxi', help='taxi or bike or ctm')
parser.add_argument('--gpu_ids', default='0, 1, 2, 3', help='indexes of gpus to use')
parser.add_argument('--memory_growth', default=True)
parser.add_argument('--index', default='SOTA', help='indexes of model to be trained')
parser.add_argument('--test_name', default=None)
parser.add_argument('--hyp', default=[None])
parser.add_argument('--run_time', default=10)
parser.add_argument('--remove_old_files', default=True)
parser.add_argument('--load_saved_data', default=False)
parser.add_argument('--no_save', default=False)
parser.add_argument('--test_model', default=None)
parser.add_argument('--mixed_precision', default=False)
parser.add_argument('--always_test', default=None)

""" Model hyperparameters """
d_model = 256
parser.add_argument('--n_layer', default=3, help='num of self-attention layers')
parser.add_argument('--d_model', default=d_model, help='model dimension')
parser.add_argument('--dff', default=d_model * 4, help='dimension of feed-forward networks')
parser.add_argument('--n_head', default=8, help='number of attention heads')
parser.add_argument('--r_d', default=0.1)
parser.add_argument('--conv_layer', default=3)
parser.add_argument('--conv_filter', default=d_model)

""" Training settings """
# weights = np.array([1, 1], dtype=np.float32)
weights = None
parser.add_argument('--MAX_EPOCH', default=250)
parser.add_argument('--BATCH_SIZE', default=256)
parser.add_argument('--warmup_steps', default=4000)
parser.add_argument('--verbose_train', default=1)
parser.add_argument('--weights', default=weights)
parser.add_argument('--es_patience', default=10)
parser.add_argument('--es_threshold', default=0.01)
parser.add_argument('--model_summary', default=True)

""" Data hyperparameters """
parser.add_argument('--n_w', default=1, help='num of previous weeks to consider')
parser.add_argument('--n_d', default=3, help='num of previous days to consider')
parser.add_argument('--n_wd_times', default=1, help='num of time in previous days to consider')
parser.add_argument('--n_p', default=3, help='num of time in today to consider')
parser.add_argument('--n_before', default=0, help='num of time before predicted time to consider')
parser.add_argument('--l_half', default=3)
parser.add_argument('--pre_shuffle', default=True)
parser.add_argument('--st_revert', default=False)

args = parser.parse_args()

def remove_oldfiles(model_index):
    try:
        shutil.rmtree('./checkpoints/stream_t/{}'.format(model_index), ignore_errors=True)
    except:
        pass
    try:
        shutil.rmtree('./checkpoints/stsan/{}'.format(model_index), ignore_errors=True)
    except:
        pass
    try:
        os.remove('./results/{}.txt'.format(model_index))
    except:
        pass
    try:
        shutil.rmtree(os.environ['HOME'] + '/tensorboard/stsan/{}'.format(model_index), ignore_errors=True)
    except:
        pass

if args.mixed_precision:
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

assert args.dataset in ['taxi', 'bike']
print("Dataset chosen: {}".format(args.dataset))

from train import TrainModel
import tensorflow as tf
import tensorflow.keras.backend as K

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus and args.memory_growth:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

if not os.path.exists('./results'):
    os.makedirs('./results')

if __name__ == "__main__":
    if args.test_name:
        for this_arg in args.hyp:
            for cnt in range(1 if args.test_model else args.run_time):
                model_index = args.dataset + '_{}_{}_{}_{}'.format(
                    'test' if args.test_model else args.index, args.test_name, this_arg, cnt + 1)
                print('Model index: {}'.format(model_index))
                result_writer = ResultWriter("results/{}.txt".format(model_index))

                exec("%s = %d" % ('args.{}'.format(args.test_name), this_arg))

                if args.remove_old_files:
                    remove_oldfiles(model_index)

                result_writer.write(str(args))

                model_trainer = TrainModel(model_index, args)
                model_trainer.train()

                args.load_saved_data = True
                K.clear_session()

    else:
        for cnt in range(1 if args.test_model else args.run_time):
            model_index = args.dataset + '_{}_{}'.format('test' if args.test_model else args.index, cnt + 1)
            print('Model index: {}'.format(model_index))
            result_writer = ResultWriter("results/{}.txt".format(model_index))

            if args.remove_old_files:
                remove_oldfiles(model_index)

            result_writer.write(str(args))

            model_trainer = TrainModel(model_index, args)
            model_trainer.train()

            args.load_saved_data = True
            K.clear_session()

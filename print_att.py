""" Print out the spatial and temporal importance. Need to run training first """

from __future__ import absolute_import, division, print_function, unicode_literals

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
gpu_id = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

import tensorflow as tf
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

import parameters_nyctaxi as params

from models import Stream_T, ST_SAN

from utils.DataLoader import DataLoader

from random import randint
import matplotlib.pyplot as plt

def plt_att_t(inp):
    fig = plt.figure(figsize=(8, 4))

    trans_in = tf.reduce_mean(inp[:, :, :2], axis=-1)
    trans_out = tf.reduce_mean(inp[:, :, 2:], axis=-1)
    trans_mtx = [trans_in, trans_out]

    for slice in range(2):
        ax = fig.add_subplot(1, 2, slice + 1)

        ax.matshow(trans_mtx[slice], cmap='viridis')

        if slice == 0:
            ax.set_xlabel('spatial attention: inflow', fontdict = {'fontsize': 16})
        else:
            ax.set_xlabel('spatial attention: outflow', fontdict = {'fontsize': 16})

    plt.tight_layout()
    plt.savefig('figures/spatial_attn.png')

def plt_att_f(inp):
    fig = plt.figure(figsize=(4, 3))

    att = tf.reduce_mean(inp, axis=0)
    att = tf.reduce_mean(att, axis=0)
    att = tf.reduce_mean(att, axis=0)
    att = np.expand_dims(att, axis=0)

    ax = fig.add_subplot(1, 1, 1)

    ax.matshow(att, cmap='viridis')

    fontdict = {'fontsize': 10}

    ax.set_xticks(range(22))
    ax.set_yticks(range(1))

    ax.set_ylim(0.5, -0.5)

    ls = [['day [{}] - time [{}]'.format(j - 7, i) for i in range(3)] for j in range(7)]
    labels = []

    for l in ls:
        labels += l

    labels += ['today - time [0]']

    ax.set_xticklabels(
        labels,
        fontdict=fontdict, rotation=90)

    ax.set_yticklabels(['att'], fontdict={'fontsize': 10})

    ax.set_xlabel('temporal attention')

    plt.savefig('figures/temporal_attn.png')

trans_max = params.trans_train_max

""" Model hyperparameters """
num_layers = 4
d_model = 64
dff = 128
d_final = 256
num_heads = 8
dropout_rate = 0.1
cnn_layers = 3
cnn_filters = 64

""" Training settings"""
BATCH_SIZE = 128
MAX_EPOCHS = 500
earlystop_patience_stream_t = 10
earlystop_patience_stsan = 15
warmup_steps = 4000
verbose_train = 1

""" Data hyperparameters """
load_saved_data = True
num_weeks_hist = 0
num_days_hist = 7
num_intervals_hist = 3
num_intervals_curr = 1
num_intervals_before_predict = 1
num_intervals_enc = (num_weeks_hist + num_days_hist) * num_intervals_hist + num_intervals_curr
local_block_len = 3

stream_t = Stream_T(num_layers,
                    d_model,
                    num_heads,
                    dff,
                    cnn_layers,
                    cnn_filters,
                    4,
                    num_intervals_enc,
                    dropout_rate)

print('Loading tranied Stream-T...')
stream_t_checkpoint_path = "./checkpoints/stream_t/taxi_1"

stream_t_ckpt = tf.train.Checkpoint(Stream_T=stream_t)

stream_t_ckpt_manager = tf.train.CheckpointManager(stream_t_ckpt, stream_t_checkpoint_path,
                                                   max_to_keep=(
                                                           earlystop_patience_stream_t + 1))

stream_t_ckpt.restore(
    stream_t_ckpt_manager.checkpoints[0]).expect_partial()

print('Stream-T restored...')

st_san = ST_SAN(stream_t, num_layers, d_model, num_heads, dff, cnn_layers, cnn_filters,
                num_intervals_enc,
                d_final, dropout_rate)

checkpoint_path = "./checkpoints/st_san/taxi_1"

ckpt = tf.train.Checkpoint(ST_SAN=st_san)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path,
                                          max_to_keep=(earlystop_patience_stsan + 1))

ckpt.restore(ckpt_manager.checkpoints[0]).expect_partial()

print('ST-SAN restored...')

data_loader = DataLoader('taxi')

flow_inputs_hist, transition_inputs_hist, ex_inputs_hist, flow_inputs_curr, transition_inputs_curr, \
ex_inputs_curr, ys_transitions, ys = \
    data_loader.generate_data('test',
                              num_weeks_hist,
                              num_days_hist,
                              num_intervals_hist,
                              num_intervals_curr,
                              num_intervals_before_predict,
                              local_block_len,
                              load_saved_data)

sample_index = randint(0, flow_inputs_curr.shape[0])

flow_hist = flow_inputs_hist[sample_index:(sample_index + 1), :, :, :, :]
trans_hist = transition_inputs_hist[sample_index:(sample_index + 1), :, :, :, :]
ex_hist = ex_inputs_hist[sample_index:(sample_index + 1), :, :]
flow_curr = flow_inputs_curr[sample_index:(sample_index + 1), :, :, :, :]
trans_curr = transition_inputs_curr[sample_index:(sample_index + 1), :, :, :, :]
ex_curr = ex_inputs_curr[sample_index:(sample_index + 1), :, :]

predictions_t, att_t = stream_t(trans_hist, ex_hist, trans_curr, ex_curr, training=False)
predictions_f, att_f = st_san(flow_hist, trans_hist, ex_hist, flow_curr, trans_curr, ex_curr,
                              training=False)

predictions_t = np.array(predictions_t, dtype=np.float32)
predictions_f = np.array(predictions_f, dtype=np.float32)
att_t = np.squeeze(np.array(att_t['decoder_layer4_block2'], dtype=np.float32))
att_f = np.squeeze(np.array(att_f['decoder_layer4_block2'], dtype=np.float32))

plt_att_t(predictions_t)
plt_att_f(att_f)

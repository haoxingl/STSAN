from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import layers

import time
import numpy as np

import parameters_nyctaxi
import parameters_nycbike
import argparse

from utils import create_masks, CustomSchedule, load_dataset, early_stop_helper

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--dataset', type=str, default='taxi', help='default is taxi, bike for bike dataset')

args = arg_parser.parse_args()

print("GPU Available: ", tf.test.is_gpu_available())


class FeedForwardEmbedding(layers.Layer):
    def __init__(self, d_model, dff, rate=0.1):
        super(FeedForwardEmbedding, self).__init__()

        self.dense1 = layers.Dense(dff, activation='relu')
        self.dense2 = layers.Dense(d_model)
        self.dropout = layers.Dropout(rate)

    def call(self, input, training):
        output = self.dense1(input)
        output = self.dense2(output)
        output = self.dropout(output, training=training)

        return output


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    sines = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    cosines = np.cos(angle_rads[:, 1::2])

    pos_encoding = np.concatenate([sines, cosines], axis=-1)

    pos_encoding = pos_encoding[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        layers.Dense(dff, activation='relu'),
        layers.Dense(d_model)
    ])


class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class DecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, total_slot=4320, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = FeedForwardEmbedding(d_model, dff, rate)

        self.pos_encoding = positional_encoding(total_slot, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = layers.Dropout(rate)

    def call(self, flow, training, mask):
        seq_len = tf.shape(flow)[1]

        flow = self.embedding(flow, training=training)
        flow *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        flow += self.pos_encoding[:, :seq_len, :]

        for i in range(self.num_layers):
            flow = self.enc_layers[i](flow, training, mask)

        return flow


class Decoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, total_slot=4320, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = FeedForwardEmbedding(d_model, dff, rate)

        self.pos_encoding = positional_encoding(total_slot, self.d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate)

    def call(self, flow, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(flow)[1]

        flow = self.embedding(flow, training=training)
        flow *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        flow += self.pos_encoding[:, :seq_len, :]

        for i in range(self.num_layers):
            flow, _, _ = self.dec_layers[i](flow, enc_output, training,
                                            look_ahead_mask, padding_mask)

        return flow


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, output_size, total_slot=4320, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, total_slot, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, total_slot, rate)

        self.dense = layers.Dense(2 * dff, activation='relu')

        self.final_layer = layers.Dense(output_size)

    def call(self, hist_flow, curr_flow, tar, training, enc_padding_mask=None, look_ahead_mask=None,
             dec_padding_mask=None):
        flow = tf.concat([hist_flow, curr_flow], axis=1)

        enc_output_flow = self.encoder(flow, training,
                                       enc_padding_mask)

        dec_output = self.decoder(
            tar, enc_output_flow, training, look_ahead_mask, dec_padding_mask)

        output_1 = self.dense(dec_output)

        final_output = self.final_layer(output_1)

        return final_output


def evaluate(model, test_dataset, TARGET_SIZE, flow_max, half_size, verbose=1):
    threshold = 10 / flow_max

    in_rmse = [tf.keras.metrics.RootMeanSquaredError() for _ in range(TARGET_SIZE)]
    out_rmse = [tf.keras.metrics.RootMeanSquaredError() for _ in range(TARGET_SIZE)]
    in_mae = [tf.keras.metrics.MeanAbsoluteError() for _ in range(TARGET_SIZE)]
    out_mae = [tf.keras.metrics.MeanAbsoluteError() for _ in range(TARGET_SIZE)]

    for (batch, (inp, tar)) in enumerate(test_dataset):
        hist_flow = inp['hist_flow']
        curr_flow = inp['curr_flow']

        dec_input = tar[:, 0, :]
        output = tf.expand_dims(dec_input, 1)
        real = tar[:, 1:, :]

        for i in range(TARGET_SIZE):
            look_ahead_mask = create_masks(output)

            predictions = model(hist_flow,
                                curr_flow,
                                output,
                                training=False,
                                look_ahead_mask=look_ahead_mask)

            predictions = predictions[:, -1:, :]

            output = tf.concat([output, predictions], axis=1)

        pred = output[:, 1:, :]
        mask = tf.math.greater(real, threshold)
        mask = tf.cast(mask, dtype=pred.dtype)
        pred_masked = pred * mask
        real_masked = real * mask
        for i in range(TARGET_SIZE):
            in_rmse[i](real_masked[:, i, :half_size], pred_masked[:, i, :half_size])
            out_rmse[i](real_masked[:, i, half_size:], pred_masked[:, i, half_size:])
            in_mae[i](real_masked[:, i, :half_size], pred_masked[:, i, :half_size])
            out_mae[i](real_masked[:, i, half_size:], pred_masked[:, i, half_size:])

    if verbose:
        for i in range(TARGET_SIZE):
            print('Slot {} INFLOW_RMSE {:.8f} OUTFLOW_RMSE {:.8f} INFLOW_MAE {:.8f} OUTFLOW_MAE {:.8f}'.format(
                i + 1,
                in_rmse[i].result(),
                out_rmse[i].result(),
                in_mae[i].result(),
                out_mae[i].result()))

    return in_rmse[0].result() + in_rmse[-1].result(), out_rmse[0].result() + out_rmse[-1].result()


loss_object = tf.keras.losses.MeanSquaredError()


def loss_function(real, pred, threshold=None):
    loss_ = loss_object(real, pred)
    if threshold:
        mask = tf.math.greater(real, threshold)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

    return tf.reduce_mean(loss_)


if args.dataset == 'taxi':
    output_size = parameters_nyctaxi.output_size
    flow_max = parameters_nyctaxi.flow_train_max
    total_slot = int(parameters_nyctaxi.timeslot_total)
elif args.dataset == 'bike':
    output_size = parameters_nycbike.output_size
    flow_max = parameters_nycbike.flow_train_max
    total_slot = int(parameters_nycbike.timeslot_total)
else:
    raise Exception("Dataset should be taxi or bike")

""" Model hyperparameters """
num_layers = 1  # 4
d_model = 32  # 128
dff = 128  # 512
num_heads = 4  # 4
dropout_rate = 0.1
half_size = int(output_size / 2)

""" Training settings"""
save_ckpt = True
BATCH_SIZE = 64
MAX_EPOCHS = 500
verbose_train = 1
test_period = 1
earlystop_epoch = 30
earlystop_patience = 10
earlystop_threshold = 1.0

""" Data hyperparameters """
TARGET_SIZE = 12  # number of future slots to predict
predicted_weather = True  # use weather prediction
hist_day_num = 7
hist_day_seq_len = 25
curr_day_seq_len = 24

train_dataset, val_dataset, test_dataset = load_dataset(args.dataset, predicted_weather, BATCH_SIZE, TARGET_SIZE + 1,
                                           hist_day_num,
                                           hist_day_seq_len, curr_day_seq_len, total_slot)

learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

train_loss = tf.keras.metrics.Mean()
train_mse = tf.keras.metrics.MeanSquaredError()
train_rmse = tf.keras.metrics.RootMeanSquaredError()
train_mae = tf.keras.metrics.MeanAbsoluteError()

transformer = Transformer(num_layers, d_model, num_heads, dff, output_size, total_slot=total_slot,
                          rate=dropout_rate)

if save_ckpt:
    checkpoint_path = "./checkpoints/transformer"

    ckpt = tf.train.Checkpoint(Transformer=transformer,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=25)

    # if ckpt_manager.latest_checkpoint:
    #     ckpt.restore(ckpt_manager.latest_checkpoint)
    #     print('Latest checkpoint restored!!')


@tf.function
def train_step(hist_flow, curr_flow, tar):
    tar_inp = tar[:, :-1, :]
    tar_real = tar[:, 1:, :]

    look_ahead_mask = create_masks(tar_inp)

    with tf.GradientTape() as tape:
        predictions = transformer(hist_flow,
                                  curr_flow,
                                  tar_inp,
                                  training=True,
                                  look_ahead_mask=look_ahead_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_mse(tar_real, predictions)
    train_rmse(tar_real, predictions)
    train_mae(tar_real, predictions)


""" Start training... """
earlystop_flag = False
earlystop_helper = early_stop_helper(earlystop_patience, test_period, earlystop_epoch, earlystop_threshold)
for epoch in range(MAX_EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_mse.reset_states()
    train_rmse.reset_states()
    train_mae.reset_states()

    for (batch, (inp, tar)) in enumerate(train_dataset):
        hist_flow = inp['hist_flow']
        curr_flow = inp['curr_flow']

        train_step(hist_flow, curr_flow, tar)

        if (batch + 1) % 100 == 0 and verbose_train:
            print('Epoch {} Batch {} Loss {:.8f} MSE {:.8f} sumed_RMSE {:.8f} sumed_MAE {:.8f}'.format(epoch + 1,
                                                                                                       batch,
                                                                                                       train_loss.result(),
                                                                                                       train_mse.result(),
                                                                                                       train_rmse.result(),
                                                                                                       train_mae.result()))
    if verbose_train:
        print(
            'Epoch {} Loss {:.8f} MSE {:.8f} sumed_RMSE {:.8f} sumed_MAE {:.8f}'.format(epoch + 1,
                                                                                        train_loss.result(),
                                                                                        train_mse.result(),
                                                                                        train_rmse.result(),
                                                                                        train_mae.result()))
        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    if epoch >= earlystop_epoch and (epoch + 1) % test_period == 0:
        print("Validation Result: ")
        in_rmse, out_rmse = evaluate(transformer, val_dataset, TARGET_SIZE, flow_max, half_size, verbose=1)
        earlystop_flag = earlystop_helper.check(in_rmse, out_rmse, epoch)

    if earlystop_flag:
        print("Early stoping...")
        if save_ckpt:
            ckpt.restore(ckpt_manager.checkpoints[int(- earlystop_patience / test_period)])
            print('Checkpoint restored!!')
        break

    if epoch >= earlystop_epoch and save_ckpt and (epoch + 1) % test_period == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}\n'.format(epoch + 1, ckpt_save_path))

print("Final Test Result: ")
_, _ = evaluate(transformer, test_dataset, TARGET_SIZE, flow_max, half_size)

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import layers

import numpy as np

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"


class FeedForwardEmbedding(layers.Layer):
    def __init__(self, d_model, dff, rate=0.1):
        super(FeedForwardEmbedding, self).__init__()

        self.dense1 = layers.Dense(dff)
        self.dense2 = layers.Dense(d_model)
        self.dropout = layers.Dropout(rate)

    def call(self, input, training):
        output = gelu(self.dense1(input))
        output = self.dense2(output)
        output = self.dropout(output, training=training)

        return output


""" This layer is not available yet!!!!! """
class ConvEmbedding(layers.Layer):
    def __init__(self, d_model, filters=64, rate=0.1):
        super(ConvEmbedding, self).__init__()
        pass

    def call(self, input, training):
        return False


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


def gelu(tensor):
    return tensor * 0.5 * (1.0 + tf.math.erf(tensor / tf.math.sqrt(2.0)))


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

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

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)

        self.dense = layers.Dense(d_model)

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


class FuseLayer(layers.Layer):
    def __init__(self, d_model):
        super(FuseLayer, self).__init__()

        self.dense_flow_1 = layers.Dense(d_model)
        self.dense_ex_1 = layers.Dense(d_model)
        self.dense_flow_2 = layers.Dense(d_model)
        self.dense_ex_2 = layers.Dense(d_model)

    def call(self, flow, ex):
        f = gelu(self.dense_flow_1(flow) + self.dense_ex_1(ex))

        flow_output = gelu(self.dense_flow_2(f))
        ex_output = gelu(self.dense_ex_2(f))

        return flow_output, ex_output


class FuseLayer_FF(layers.Layer):
    def __init__(self, d_model, dff=2048):
        super(FuseLayer_FF, self).__init__()

        self.dense_flow_1 = layers.Dense(d_model)
        self.dense_ex_1 = layers.Dense(d_model)

        self.dense_flow_ff = layers.Dense(dff)
        self.dense_ex_ff = layers.Dense(dff)

        self.dense_flow_2 = layers.Dense(d_model)
        self.dense_ex_2 = layers.Dense(d_model)

    def call(self, flow, ex):
        f = gelu(self.dense_flow_1(flow) + self.dense_ex_1(ex))

        flow_output = gelu(self.dense_flow_ff(f))
        ex_output = gelu(self.dense_ex_ff(f))

        flow_output = self.dense_flow_2(flow_output)
        ex_output = self.dense_ex_2(ex_output)

        return flow_output, ex_output


class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

        self.mha_ex = MultiHeadAttention(d_model, num_heads)

        self.layernorm1_ex = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2_ex = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1_ex = layers.Dropout(rate)
        self.dropout2_ex = layers.Dropout(rate)

        self.fuselayer = FuseLayer_FF(d_model, dff=dff)

    def call(self, x, ex, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        attn_output_ex, _ = self.mha_ex(ex, ex, ex, mask)
        attn_output_ex = self.dropout1_ex(attn_output_ex, training=training)
        out1_ex = self.layernorm1_ex(ex + attn_output_ex)

        out1_fused, out1_ex_fused = self.fuselayer(out1, out1_ex)

        ffn_output = self.dropout2(out1_fused, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        ffn_output_ex = self.dropout2_ex(out1_ex_fused, training=training)
        out2_ex = self.layernorm2_ex(out1_ex + ffn_output_ex)

        return out2, out2_ex


class DecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)

        self.mha1_ex = MultiHeadAttention(d_model, num_heads)
        self.mha2_ex = MultiHeadAttention(d_model, num_heads)

        self.layernorm1_ex = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2_ex = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3_ex = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1_ex = layers.Dropout(rate)
        self.dropout2_ex = layers.Dropout(rate)
        self.dropout3_ex = tf.keras.layers.Dropout(rate)

        self.fuselayer = FuseLayer(d_model)
        self.fuselayer_ff = FuseLayer_FF(d_model, dff=dff)

    def call(self, flow, ex, enc_output_flow, enc_output_ex, training,
             look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(flow, flow, flow,
                                               look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + flow)

        attn1_ex, attn_weights_block1_ex = self.mha1_ex(ex, ex, ex,
                                                        look_ahead_mask)
        attn1_ex = self.dropout1_ex(attn1_ex, training=training)
        out1_ex = self.layernorm1_ex(attn1_ex + ex)

        out1_fused, out1_ex_fused = self.fuselayer(out1, out1_ex)

        attn2, attn_weights_block2 = self.mha2(
            enc_output_flow, enc_output_flow, out1_fused, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        attn2_ex, attn_weights_block2_ex = self.mha2_ex(
            enc_output_ex, enc_output_ex, out1_ex_fused, padding_mask)
        attn2_ex = self.dropout2(attn2_ex, training=training)
        out2_ex = self.layernorm2_ex(attn2_ex + out1_ex)

        out2_fused, out2_ex_fused = self.fuselayer_ff(out2, out2_ex)

        ffn_output = self.dropout3(out2_fused, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        ffn_output_ex = self.dropout3_ex(out2_ex_fused, training=training)
        out3_ex = self.layernorm3_ex(ffn_output_ex + out2)

        return out3, out3_ex, attn_weights_block1, attn_weights_block2, attn_weights_block1_ex, attn_weights_block2_ex


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, dff_ex, total_slot=4320, rate=0.1, conv_embedding=False):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        if not conv_embedding:
            self.embedding_flow = FeedForwardEmbedding(d_model, dff, rate)
        else:
            self.embedding_flow = ConvEmbedding(d_model, rate)
        self.embedding_ex = FeedForwardEmbedding(d_model, dff_ex, rate)
        self.pos_encoding = positional_encoding(total_slot, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        # self.dropout_flow = layers.Dropout(rate)
        # self.dropout_ex = layers.Dropout(rate)

    def call(self, hist_flow, hist_ex, curr_flow, curr_ex, training, mask):
        flow = tf.concat([hist_flow, curr_flow], axis=1)
        ex = tf.concat([hist_ex, curr_ex], axis=1)
        flow = self.embedding_flow(flow, training)
        ex = self.embedding_ex(ex, training)
        flow *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        ex *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        seq_len = tf.shape(flow)[1]

        ex += self.pos_encoding[:, :seq_len, :]
        flow += self.pos_encoding[:, :seq_len, :]

        # flow = self.dropout_flow(flow, training=training)
        # ex = self.dropout_ex(ex, training=training)

        for i in range(self.num_layers):
            flow, ex = self.enc_layers[i](flow, ex, training, mask)

        return flow, ex


class Decoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, dff_ex, total_slot=4320, rate=0.1, conv_embedding=False):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        if not conv_embedding:
            self.embedding_flow = FeedForwardEmbedding(d_model, dff, rate)
        else:
            self.embedding_flow = ConvEmbedding(d_model, rate)
        self.embedding_ex = FeedForwardEmbedding(d_model, dff_ex, rate)
        self.pos_encoding = positional_encoding(total_slot, self.d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        # self.dropout_flow = layers.Dropout(rate)
        # self.dropout_ex = layers.Dropout(rate)

    def call(self, flow, ex, enc_output_flow, enc_output_ex, training,
             look_ahead_mask, padding_mask):
        seq_len = tf.shape(flow)[1]
        attention_weights = {}

        flow = self.embedding_flow(flow, training)
        ex = self.embedding_ex(ex, training)
        flow *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        ex *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        flow += self.pos_encoding[:, :seq_len, :]
        ex += self.pos_encoding[:, :seq_len, :]

        # flow = self.dropout_flow(flow, training=training)
        # ex = self.dropout_ex(ex, training=training)

        for i in range(self.num_layers):
            flow, ex, block1, block2, block1_ex, block2_ex = self.dec_layers[i](flow, ex, enc_output_flow,
                                                                                enc_output_ex, training,
                                                                                look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2
            attention_weights['decoder_layer{}_block1_ex'.format(i + 1)] = block1_ex
            attention_weights['decoder_layer{}_block2_ex'.format(i + 1)] = block2_ex

        return flow, ex, attention_weights


class AMEX(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, output_size, dff_ex=16, total_slot=4320, rate=0.1,
                 conv_embedding=False):
        super(AMEX, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, dff_ex, total_slot, rate, conv_embedding)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, dff_ex, total_slot, rate, conv_embedding)

        self.final_layer = layers.Dense(output_size)

    def call(self, hist_flow, hist_ex, curr_flow, curr_ex, next_exs, tar, training, enc_padding_mask=None, look_ahead_mask=None,
             dec_padding_mask=None):
        enc_output_flow, enc_output_ex = self.encoder(hist_flow, hist_ex, curr_flow, curr_ex, training,
                                                      enc_padding_mask)

        dec_output_flow, dec_output_ex, attention_weights = self.decoder(
            tar, next_exs, enc_output_flow, enc_output_ex, training, look_ahead_mask, dec_padding_mask)

        dec_output = tf.concat([dec_output_flow, dec_output_ex], axis=2)

        final_output = self.final_layer(dec_output)

        return final_output, attention_weights

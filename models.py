from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.utils import get_custom_objects


def gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))


get_custom_objects().update({'gelu': layers.Activation(gelu)})

act = 'gelu'


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def spatial_posenc(position_r, position_c, d_model):
    angle_rads_r = get_angles(position_r, np.arange(d_model)[np.newaxis, :], d_model)

    angle_rads_c = get_angles(position_c, np.arange(d_model)[np.newaxis, :], d_model)

    pos_encoding = np.zeros(angle_rads_r.shape, dtype=np.float32)

    pos_encoding[:, 0::2] = np.sin(angle_rads_r[:, 0::2])

    pos_encoding[:, 1::2] = np.cos(angle_rads_c[:, 1::2])

    return tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)


def get_spe(l_half, d_model):
    l_full = 2 * l_half + 1

    mtx_r = np.repeat(np.arange(l_full)[:, np.newaxis], [l_full], axis=1)
    mtx_c = np.repeat(np.arange(l_full)[np.newaxis, :], [l_full], axis=0)

    mtx_r_flat = np.reshape(mtx_r - l_half, (-1, 1))
    mtx_c_flat = np.reshape(mtx_c - l_half, (-1, 1))

    spe = spatial_posenc(mtx_r_flat, mtx_c_flat, d_model)
    spe = np.reshape(spe, (l_full, l_full, d_model))

    return np.array(spe, dtype=np.float32)


class Convs(layers.Layer):
    def __init__(self, n_layer, n_filter, l_hist, r_d=0.1):
        super(Convs, self).__init__()

        self.n_layer = n_layer
        self.l_hist = l_hist

        self.convs = [[layers.Conv2D(n_filter, (3, 3), activation=act, padding='same')
                       for _ in range(l_hist)] for _ in range(n_layer)]
        self.dropouts = [[layers.Dropout(r_d) for _ in range(l_hist)] for _ in range(n_layer)]

    def call(self, inps, training):
        outputs = tf.split(inps, self.l_hist, axis=-2)
        for i in range(self.n_layer):
            for j in range(self.l_hist):
                if i == 0:
                    outputs[j] = tf.squeeze(outputs[j], axis=-2)
                outputs[j] = self.convs[i][j](outputs[j])
                outputs[j] = self.dropouts[i][j](outputs[j], training=training)
                if i == self.n_layer - 1:
                    outputs[j] = tf.expand_dims(outputs[j], axis=-2)

        output = tf.concat(outputs, axis=-2)

        return output


class Gated_Fusion(layers.Layer):
    def __init__(self, conv_filter, conv_layer, r_d=0.1):
        super(Gated_Fusion, self).__init__(name='final_f')

        self.conv_layer = conv_layer

        self.sigs = [layers.Activation('sigmoid') for _ in range(2)]

        self.convs_f = [layers.Conv2D(conv_filter, (3, 3), activation=act) for _ in range(conv_layer)]
        self.convs_t = [layers.Conv2D(conv_filter, (3, 3), activation=act) for _ in range(conv_layer - 1)]

        self.dense = layers.Dense(2, activation='tanh')

        self.dropouts_f = [layers.Dropout(r_d) for _ in range(conv_layer - 1)]
        self.dropouts_t = [layers.Dropout(r_d) for _ in range(conv_layer - 1)]
        self.dropout = layers.Dropout(r_d)

    def call(self, inp_f, inp_t, training):
        output_f = tf.squeeze(inp_f, axis=-2)
        output_t = tf.squeeze(inp_t, axis=-2)

        for i in range(self.conv_layer - 1):
            output_t = self.convs_t[i](output_t)
            output_f = self.convs_f[i](output_f) * self.sigs[i](output_t)
            output_t = self.dropouts_t[i](output_t, training=training)
            output_f = self.dropouts_f[i](output_f, training=training)

        output_f = tf.squeeze(self.convs_f[-1](output_f))
        output = self.dense(self.dropout(output_f))

        return output


def scaled_dot_product_attention(q, k, v, mask=None):
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, n_head, self_att=True):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.self_att = self_att

        assert d_model % n_head == 0

        self.depth = d_model // n_head

        if self_att:
            self.wx = layers.Dense(d_model * 3)
        else:
            self.wq = layers.Dense(d_model)
            self.wkv = layers.Dense(d_model * 2)

        self.wo = layers.Dense(d_model)

    def split_heads(self, x):
        shape = tf.shape(x)
        x = tf.reshape(x, (shape[0], shape[1], shape[2], shape[3], self.n_head, self.depth))
        return tf.transpose(x, perm=[0, 4, 1, 2, 3, 5])

    def call(self, v, k, q, mask):
        if self.self_att:
            q, k, v = tf.split(self.wx(q), 3, axis=-1)
        else:
            q = self.wq(q)
            k, v = tf.split(self.wkv(k), 2, axis=-1)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 3, 4, 1, 5])

        d_shape = tf.shape(scaled_attention)

        concat_attention = tf.reshape(scaled_attention, (d_shape[0], d_shape[1], d_shape[2], d_shape[3], self.d_model))

        output = self.wo(concat_attention)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return Sequential([
        layers.Dense(dff, activation=act),
        layers.Dense(d_model)
    ])


def ex_encoding(d_model, dff):
    return Sequential([
        layers.Dense(dff),
        layers.Dense(d_model, activation='tanh')
    ])


class EncoderLayer(layers.Layer):
    def __init__(self, d_model, n_head, dff, r_d=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, n_head)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(r_d)
        self.dropout2 = layers.Dropout(r_d)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class DecoderLayer(layers.Layer):
    def __init__(self, d_model, n_head, dff, r_d=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, n_head)
        self.mha2 = MultiHeadAttention(d_model, n_head, self_att=False)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(r_d)
        self.dropout2 = layers.Dropout(r_d)
        self.dropout3 = layers.Dropout(r_d)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(layers.Layer):
    def __init__(self, n_layer, d_model, n_head, dff, conv_layer, conv_filter, l_hist, l_half, r_d=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.n_layer = n_layer
        self.spe = get_spe(l_half, d_model)[np.newaxis, :, :, np.newaxis, :]

        self.ex_encoder = ex_encoding(d_model, dff)
        self.dropout = layers.Dropout(r_d)

        self.convs = Convs(conv_layer, conv_filter, l_hist, r_d)

        self.encs = [EncoderLayer(d_model, n_head, dff, r_d) for _ in range(n_layer)]

    def call(self, x, ex, training, mask):
        ex_enc = self.ex_encoder(ex)[:, tf.newaxis, tf.newaxis, :, :]

        x = self.convs(x, training)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += ex_enc + self.spe

        x = self.dropout(x, training=training)

        for i in range(self.n_layer):
            x = self.encs[i](x, training, mask)

        return x


class Decoder(layers.Layer):
    def __init__(self, n_layer, d_model, n_head, dff, conv_layer, conv_filter, l_half, r_d=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.n_layer = n_layer
        self.spe = get_spe(l_half, d_model)[np.newaxis, :, :, np.newaxis, :]

        self.ex_encoder = ex_encoding(d_model, dff)
        self.dropout = layers.Dropout(r_d)

        self.convs = Convs(conv_layer, conv_filter, 1, r_d)

        self.decs = [DecoderLayer(d_model, n_head, dff, r_d) for _ in range(n_layer)]

    def call(self, x, ex, enc_output, training, look_ahead_mask, padding_mask):
        attention_weights = {}

        ex_enc = self.ex_encoder(ex)[:, tf.newaxis, tf.newaxis, :, :]

        x = self.convs(x, training)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += ex_enc + self.spe

        x = self.dropout(x, training=training)

        for i in range(self.n_layer):
            x, block1, block2 = self.decs[i](x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        return x, attention_weights


class Stream(layers.Layer):
    def __init__(self, name, n_layer, d_model, n_head, dff, conv_layer, conv_filter, l_hist, l_half, r_d=0.1):
        super(Stream, self).__init__(name=name)

        self.encoder = Encoder(n_layer, d_model, n_head, dff, conv_layer, conv_filter, l_hist, l_half, r_d)

        self.decoder = Decoder(n_layer, d_model, n_head, dff, conv_layer, conv_filter, l_half, r_d)

    def call(self, x, enc_ex, dec_x, dec_ex, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(x, enc_ex, training, enc_padding_mask)

        dec_output, attention_weights = \
            self.decoder(dec_x, dec_ex, enc_output, training, look_ahead_mask, dec_padding_mask)

        return dec_output, attention_weights


class Stream_T(Model):
    def __init__(self, n_layer, d_model, n_head, dff, conv_layer, conv_filter, l_hist, l_half, r_d=0.1):
        super(Stream_T, self).__init__()

        self.stream_t = Stream('stream-t', n_layer, d_model, n_head, dff, conv_layer, conv_filter, l_hist, l_half, r_d)
        self.dense = layers.Dense(4, activation='tanh')
        self.dropout = layers.Dropout(r_d)

    def call(self, enc_inp, enc_ex, dec_inp, dec_ex, training,
             enc_padding_mask, look_ahead_mask, dec_padding_mask):
        dec_output_t, attention_weights_t = \
            self.stream_t(enc_inp, enc_ex, dec_inp, dec_ex, training,
                          enc_padding_mask, look_ahead_mask, dec_padding_mask)

        output = self.dense(self.dropout(tf.squeeze(dec_output_t), training=training))

        return output, attention_weights_t


class STSAN(Model):
    def __init__(self, stream_t, n_layer, d_model, n_head, dff, conv_layer, conv_filter, l_hist, l_half, r_d=0.1):
        super(STSAN, self).__init__()

        self.stream_f = Stream('stream-f', n_layer, d_model, n_head, dff, conv_layer, conv_filter, l_hist, l_half, r_d)
        self.final_convs = Gated_Fusion(d_model, l_half, r_d)

        self.stream_t = stream_t.get_layer('stream-t')
        self.stream_t.trainable = False

    def call(self, enc_ft, enc_ex, dec_ft, dec_ex, training,
             enc_padding_mask, look_ahead_mask, dec_padding_mask):
        dec_output_f, attention_weights_f = \
            self.stream_f(enc_ft[..., :2], enc_ex, dec_ft[..., :2], dec_ex, training,
                          enc_padding_mask[0], look_ahead_mask[0], dec_padding_mask[0])

        dec_output_t, attention_weights_t = \
            self.stream_t(enc_ft[..., 2:], enc_ex, dec_ft[..., 2:], dec_ex, False,
                          enc_padding_mask[1], look_ahead_mask[1], dec_padding_mask[1])

        output = self.final_convs(dec_output_f, dec_output_t, training)

        return output, [attention_weights_f, attention_weights_t]

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

import numpy as np


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


class ConvolutionalEmbedding(layers.Layer):
    def __init__(self, d_model, seq_len, num_layers=3, rate=0.1):
        super(ConvolutionalEmbedding, self).__init__()

        self.seq_len = seq_len
        self.num_layers = num_layers

        self.conv_layers = [layers.Conv2D(d_model, (3, 3), padding='same') for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate)

    def call(self, input, training, tar_size=None):
        outputs = []
        if tar_size is None:
            seq_len = self.seq_len
        else:
            seq_len = tar_size
        for i in range(seq_len):
            tensor = gelu(self.conv_layers[0](input[:, :, :, i, :]))
            for j in range(self.num_layers - 1):
                tensor = gelu(self.conv_layers[j + 1](tensor))
            tensor = tf.expand_dims(tensor, axis=3)
            outputs.append(tensor)

        output = tf.concat(outputs, axis=3)
        output = self.dropout(output, training)

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


def gelu(tensor):
    return tensor * 0.5 * (1.0 + tf.math.erf(tensor / tf.math.sqrt(2.0)))


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, v)

    return output, attention_weights


def scaled_dot_product_attention_3d(q, k, v, mask, ex_flag=False):
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    if ex_flag:
        attention_weights_2 = tf.broadcast_to(attention_weights,
                                            [tf.shape(v)[1], tf.shape(v)[2], tf.shape(v)[0], tf.shape(v)[3],
                                             tf.shape(attention_weights)[-2], tf.shape(attention_weights)[-1]])
        attention_weights_2 = tf.transpose(attention_weights_2, perm=[2, 0, 1, 3, 4, 5])

        output = tf.matmul(attention_weights_2, v)

    else:
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


class MultiHeadAttention_3D(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention_3D, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)

        self.dense = layers.Dense(d_model)

    def split_heads_1d(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def split_heads_3d(self, x, batch_size):
        x = tf.reshape(x, (batch_size, tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3], self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 1, 2, 4, 3, 5])

    def call(self, v, k, q, mask, ex_flag=False):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        if ex_flag:
            q = self.split_heads_1d(q, batch_size)
            k = self.split_heads_1d(k, batch_size)
        else:
            q = self.split_heads_3d(q, batch_size)
            k = self.split_heads_3d(k, batch_size)
        v = self.split_heads_3d(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention_3d(
            q, k, v, mask, ex_flag)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 1, 2, 4, 3, 5])

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, tf.shape(scaled_attention)[1], tf.shape(scaled_attention)[2],
                                       tf.shape(scaled_attention)[3], self.d_model))

        output = self.dense(concat_attention)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        layers.Dense(dff, activation='relu'),
        layers.Dense(d_model)
    ])


class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, conv_embedding=False):
        super(EncoderLayer, self).__init__()

        self.conv_embedding = conv_embedding

        if not conv_embedding:
            self.mha = MultiHeadAttention(d_model, num_heads)
        else:
            self.mha = MultiHeadAttention_3D(d_model, num_heads)
        self.ffn_x = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, ex, training, mask, ex_flag=False):
        if not self.conv_embedding:
            attn_output, _ = self.mha(x, ex, ex, mask)
        else:
            attn_output, _ = self.mha(x, ex, ex, mask, ex_flag)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn_x(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class DecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, conv_embedding=False):
        super(DecoderLayer, self).__init__()

        self.conv_embedding = conv_embedding

        if not conv_embedding:
            self.mha1 = MultiHeadAttention(d_model, num_heads)
            self.mha2 = MultiHeadAttention(d_model, num_heads)
        else:
            self.mha1 = MultiHeadAttention_3D(d_model, num_heads)
            self.mha2 = MultiHeadAttention_3D(d_model, num_heads)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)

        self.ffn_x = point_wise_feed_forward_network(d_model, dff)

    def call(self, x, ex, enc_output_x, training, look_ahead_mask, padding_mask, ex_flag=False):
        if not self.conv_embedding:
            attn1, _ = self.mha1(x, ex, ex, look_ahead_mask)
        else:
            attn1, _ = self.mha1(x, ex, ex, look_ahead_mask, ex_flag)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        if not self.conv_embedding:
            attn2, _ = self.mha2(enc_output_x, enc_output_x, out1, padding_mask)
        else:
            attn2, _ = self.mha2(enc_output_x, enc_output_x, out1, padding_mask, False)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn_x(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3

class DecoderFuseLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, conv_embedding=False):
        super(DecoderFuseLayer, self).__init__()

        self.conv_embedding = conv_embedding

        self.mha0 = MultiHeadAttention(65, 1)
        if not conv_embedding:
            self.mha1 = MultiHeadAttention(d_model, num_heads)
        else:
            self.mha1 = MultiHeadAttention_3D(d_model, num_heads)

        self.layernorm0 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout0 = layers.Dropout(rate)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

        self.ffn_x = point_wise_feed_forward_network(d_model, dff)

    def call(self, ex, enc_output_x, enc_output_ex, training, look_ahead_mask, padding_mask, ex_flag=False):
        attn0, _ = self.mha0(ex, ex, ex, look_ahead_mask)
        attn0 = self.dropout0(attn0, training=training)
        out0 = self.layernorm0(attn0 + ex)

        if not self.conv_embedding:
            attn1, _ = self.mha1(enc_output_x, enc_output_ex, out0, padding_mask)
        else:
            attn1, _ = self.mha1(enc_output_x, enc_output_ex, out0, padding_mask, ex_flag)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1)

        ffn_output = self.ffn_x(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(ffn_output + out1)

        return out2


class Encoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, dff_ex, seq_len=199, total_slot=4320, rate=0.1,
                 conv_embedding=False, conv_layers=3):
        super(Encoder, self).__init__()

        self.conv_embedding = conv_embedding

        self.d_model = d_model
        self.num_layers = num_layers

        if not conv_embedding:
            self.embedding_flow = FeedForwardEmbedding(d_model, dff, rate)
        else:
            self.embedding_flow = ConvolutionalEmbedding(d_model, seq_len, conv_layers, rate)

        self.pos_encoding = positional_encoding(total_slot, self.d_model)
        self.pos_encoding_ex = positional_encoding(total_slot, 65)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate, conv_embedding=conv_embedding)
                           for _ in range(num_layers)]

        self.enc_layers_ex = [EncoderLayer(d_model, num_heads, dff, rate, conv_embedding=conv_embedding)
                              for _ in range(num_layers)]

    def call(self, flow, ex, training, mask):
        seq_len = tf.shape(flow)[-2]

        flow = self.embedding_flow(flow, training=training)
        flow *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        flow += self.pos_encoding[:, :seq_len, :]

        ex *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        ex += self.pos_encoding_ex[:, :seq_len, :]

        for i in range(self.num_layers):
            if i == 0:
                if self.conv_embedding:
                    flow_ex = self.enc_layers_ex[i](flow, ex, training, mask, ex_flag=True)
                else:
                    flow_ex = self.enc_layers_ex[i](flow, ex, training, mask)
            else:
                if self.conv_embedding:
                    flow_ex = self.enc_layers_ex[i](flow_ex, ex, training, mask, ex_flag=True)
                else:
                    flow_ex = self.enc_layers_ex[i](flow_ex, ex, training, mask)
            flow = self.enc_layers[i](flow, flow, training, mask)

        return flow, flow_ex, ex


class Decoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, dff_ex, seq_len=12, total_slot=4320, rate=0.1,
                 conv_embedding=False, conv_layers=3):
        super(Decoder, self).__init__()

        self.conv_embedding = conv_embedding

        self.d_model = d_model
        self.num_layers = num_layers

        if not conv_embedding:
            self.embedding_flow = FeedForwardEmbedding(d_model, dff, rate)
        else:
            self.embedding_flow = ConvolutionalEmbedding(d_model, seq_len, conv_layers, rate)

        self.pos_encoding = positional_encoding(total_slot, self.d_model)
        self.pos_encoding_ex = positional_encoding(total_slot, 65)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate, conv_embedding=conv_embedding)
                           for _ in range(num_layers)]
        self.dec_layers_ex = [DecoderLayer(d_model, num_heads, dff, rate, conv_embedding=conv_embedding)
                              for _ in range(num_layers - 1)]
        self.dec_fuse_layer = DecoderFuseLayer(d_model, num_heads, dff, rate, conv_embedding=conv_embedding)

    def call(self, flow, ex, enc_output_flow, enc_output_flow_ex, enc_output_ex, training,
             look_ahead_mask, padding_mask, tar_size=None):
        seq_len = tf.shape(flow)[-2]

        if self.conv_embedding:
            flow = self.embedding_flow(flow, training=training, tar_size=tar_size)
        else:
            flow = self.embedding_flow(flow, training=training)
        flow *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        flow += self.pos_encoding[:, :seq_len, :]

        ex *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        ex *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        ex += self.pos_encoding_ex[:, :seq_len, :]

        for i in range(self.num_layers):
            if i == 0:
                if self.conv_embedding:
                    flow_ex = self.dec_fuse_layer(ex, enc_output_flow_ex, enc_output_ex, training, look_ahead_mask, padding_mask,
                                                  ex_flag=True)
                else:
                    flow_ex = self.dec_fuse_layer(ex, enc_output_flow_ex, enc_output_ex, training, look_ahead_mask, padding_mask)
            else:
                if self.conv_embedding:
                    flow_ex = self.dec_layers_ex[i - 1](flow_ex, flow_ex, enc_output_flow_ex, training, look_ahead_mask, padding_mask)
                else:
                    flow_ex = self.dec_layers_ex[i - 1](flow_ex, flow_ex, enc_output_flow_ex, training, look_ahead_mask, padding_mask)
            flow = self.dec_layers[i](flow, flow, enc_output_flow, training, look_ahead_mask, padding_mask)

        return flow, flow_ex

CONCAT_or_SUM = 0

class Transformer_Ex(Model):
    def __init__(self, num_layers, d_model, num_heads, dff, output_size, seq_len=187, seq_len_tar=12, dff_ex=128, total_slot=4320,
                 rate=0.1,
                 conv_embedding=False, conv_layers=3):
        super(Transformer_Ex, self).__init__()

        self.conv_embedding = conv_embedding

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, dff_ex, seq_len, total_slot, rate, conv_embedding, conv_layers)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, dff_ex, seq_len_tar, total_slot, rate, conv_embedding, conv_layers)

        self.dense = layers.Dense(dff)

        if not conv_embedding:
            self.final_layer = layers.Dense(output_size)
        else:
            self.final_layer = layers.Dense(2)

    def call(self, hist_flow, hist_ex, curr_flow, curr_ex, next_exs, tar, training, enc_padding_mask=None,
             look_ahead_mask=None,
             dec_padding_mask=None,
             tar_size=None):
        if not self.conv_embedding:
            flow = tf.concat([hist_flow, curr_flow], axis=1)
            ex = tf.concat([hist_ex, curr_ex], axis=1)
        else:
            flow = hist_flow
            ex = hist_ex

        enc_output_flow, enc_output_flow_ex, enc_output_ex = self.encoder(flow, ex, training,
                                                      enc_padding_mask)

        dec_output_flow, dec_output_ex = self.decoder(
            tar, next_exs, enc_output_flow, enc_output_flow_ex, enc_output_ex, training, look_ahead_mask, dec_padding_mask, tar_size)


        if not CONCAT_or_SUM:
            dec_output = tf.concat([dec_output_flow, dec_output_ex], axis=-1)
        else:
            dec_output = dec_output_flow + dec_output_ex

        output_1 = gelu(self.dense(dec_output))

        final_output = self.final_layer(output_1)

        return final_output

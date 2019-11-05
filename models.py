from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

final_cnn_filters = 128

""" the local convolutional layer before the encoder and deconder stacks """


class Local_Conv(layers.Layer):
    def __init__(self, num_layers, num_filters, num_intervals, dropout_rate=0.1):
        super(Local_Conv, self).__init__()

        self.num_intervals = num_intervals  # indicate how many time intervals are included in the historical inputs
        self.num_layers = num_layers

        """ data from each time interval will be handled by one set of convolutional layers, therefore totally
            totally num_intervals sets of convolutional layers are employed """
        self.conv_layers = [[layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same')
                             for _ in range(num_layers)] for _ in range(num_intervals)]

        self.dropout_layers = [layers.Dropout(dropout_rate) for _ in range(num_intervals)]

    def call(self, inputs, training):
        outputs = []
        for i in range(self.num_intervals):
            input = inputs[:, :, :, i, :]
            output = self.conv_layers[i][0](input)
            for j in range(self.num_layers - 1):
                output = self.conv_layers[i][j + 1](output)
            output = self.dropout_layers[i](output, training=training)
            output = tf.expand_dims(output, axis=3)
            outputs.append(output)

        output_final = tf.concat(outputs, axis=3)

        return output_final


""" the implementation of the Masked Fusion mechanism, check the details in my paper """


class Masked_Fusion(layers.Layer):
    def __init__(self, name, d_final, dropout_rate=0.1):
        super(Masked_Fusion, self).__init__(name=name)

        self.sigact_layers = [layers.Activation('sigmoid') for _ in range(3)]

        self.conv_layers_flow = [layers.Conv2D(final_cnn_filters, (3, 3), activation='relu') for i in range(2)]
        self.conv_layers_trans = [layers.Conv2D(final_cnn_filters, (3, 3), activation='relu') for i in range(2)]

        self.dense1 = layers.Dense(d_final, activation='relu')
        self.dense2 = layers.Dense(2, activation='tanh')

        self.flatten = layers.Flatten()
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, input_flow, input_trans, training):
        input_flow = tf.squeeze(input_flow, axis=-2)
        input_trans = tf.squeeze(input_trans, axis=-2)

        input_flow *= self.sigact_layers[0](input_trans)

        flow_output1 = self.conv_layers_flow[0](input_flow)
        trans_output1 = self.conv_layers_trans[0](input_trans)
        flow_output1 *= self.sigact_layers[1](trans_output1)

        flow_output2 = self.conv_layers_flow[1](flow_output1)
        trans_output2 = self.conv_layers_trans[1](trans_output1)
        flow_output3 = flow_output2 * self.sigact_layers[2](trans_output2)

        output = self.flatten(flow_output3)
        output = self.dense1(output)
        output = self.dropout(output, training=training)
        output = self.dense2(output)

        return output


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, v)

    return output, attention_weights


""" the implementation of Spatial-Temporal Multi-Head Attention, detailed in my paper"""


class SpatialTemporal_MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(SpatialTemporal_MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)

        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3], self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 1, 2, 4, 3, 5])

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
                                        perm=[0, 1, 2, 4, 3, 5])

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, tf.shape(scaled_attention)[1], tf.shape(scaled_attention)[2],
                                       tf.shape(scaled_attention)[3], self.d_model))

        output = self.dense(concat_attention)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return Sequential([
        layers.Dense(dff, activation='relu'),
        layers.Dense(d_model)
    ])


class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff=128, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = SpatialTemporal_MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class DecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff=128, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = SpatialTemporal_MultiHeadAttention(d_model, num_heads)
        self.mha2 = SpatialTemporal_MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dropout3 = layers.Dropout(dropout_rate)

    def call(self, x, enc_output_x, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output_x, enc_output_x, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(layers.Layer):
    def __init__(self, name, num_layers, d_model, num_heads, dff, cnn_layers, cnn_filters, num_intervals=40,
                 dropout_rate=0.1):
        super(Encoder, self).__init__(name=name)

        self.d_model = d_model
        self.num_layers = num_layers

        self.ex_encoding = Sequential(
            [
                layers.Dense(d_model * 2, activation='relu'),
                layers.Dense(d_model, activation='sigmoid')
            ]
        )

        self.local_conv = Local_Conv(cnn_layers, cnn_filters, num_intervals, dropout_rate)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate)
                           for _ in range(num_layers)]

    def call(self, x, ex, training, mask):
        ex_enc = self.ex_encoding(ex[:, :, :55])
        pos_enc = tf.expand_dims(tf.expand_dims(ex_enc, axis=1), axis=1)

        x = self.local_conv(x, training)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += pos_enc

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x


class Decoder(layers.Layer):
    def __init__(self, name, num_layers, d_model, num_heads, dff, cnn_layers, cnn_filters, dropout_rate=0.1):
        super(Decoder, self).__init__(name=name)

        self.d_model = d_model
        self.num_layers = num_layers

        self.local_conv = Local_Conv(cnn_layers, cnn_filters, 1, dropout_rate)

        self.ex_encoding = Sequential(
            [
                layers.Dense(d_model * 2, activation='relu'),
                layers.Dense(d_model, activation='sigmoid')
            ]
        )

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, dropout_rate)
                           for _ in range(num_layers)]

    def call(self, x, ex, enc_output, training,
             look_ahead_mask, padding_mask):
        attention_weights = {}

        ex_enc = self.ex_encoding(ex[:, :, :55])
        pos_enc = tf.expand_dims(tf.expand_dims(ex_enc, axis=1), axis=1)

        x = self.local_conv(x, training=training)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += pos_enc

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        return x, attention_weights


""" define the model structure for Stream-T """


class Stream_T(Model):
    def __init__(self, num_layers, d_model, num_heads, dff, cnn_layers, cnn_filters, output_size,
                 num_intervals,
                 dropout_rate=0.1):
        super(Stream_T, self).__init__()

        self.encoder = Encoder('Encoder', num_layers, d_model, num_heads, dff, cnn_layers, cnn_filters, num_intervals,
                               dropout_rate)

        self.decoder = Decoder('Decoder', num_layers, d_model, num_heads, dff, cnn_layers, cnn_filters, dropout_rate)

        self.final_layer = layers.Dense(output_size, activation='tanh')
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x_hist, ex_hist, x_curr, ex_curr, training, enc_padding_mask=None,
             look_ahead_mask=None,
             dec_padding_mask=None):
        # concat all historical data to form the encoder input
        x_enc_inputs = tf.concat([x_hist, x_curr[:, :, :, 1:, :]], axis=-2)
        ex_enc_inputs = tf.concat([ex_hist, ex_curr[:, 1:, :]], axis=-2)
        # x_enc_inputs = x[:, :, :, :-1, :]
        # ex_enc_inputs = ex[:, :-1, :]

        # use the last interval as the decoder output
        x_dec_input = x_curr[:, :, :, -1:, :]
        ex_dec_input = ex_curr[:, -1:, :]

        enc_output = self.encoder(x_enc_inputs, ex_enc_inputs, training, enc_padding_mask)

        dec_output, attention_weights = self.decoder(
            x_dec_input, ex_dec_input, enc_output, training, look_ahead_mask, dec_padding_mask)

        dec_output = tf.squeeze(dec_output)

        dec_output = self.dropout(dec_output)

        final_output = self.final_layer(dec_output)

        return final_output, attention_weights


class ST_SAN(Model):
    def __init__(self, stream_t, num_layers, d_model, num_heads, dff, cnn_layers, cnn_filters, num_intervals, d_final,
                 dropout_rate):
        super(ST_SAN, self).__init__()

        self.flow_encoder = Encoder('Flow_Encoder', num_layers, d_model, num_heads, dff, cnn_layers, cnn_filters,
                                    num_intervals,
                                    dropout_rate)
        self.flow_decoder = Decoder('Flow_Decoder', num_layers, d_model, num_heads, dff, cnn_layers, cnn_filters,
                                    dropout_rate)

        self.trans_encoder = stream_t.get_layer('Encoder')
        self.trans_decoder = stream_t.get_layer('Decoder')

        self.trans_encoder.trainable = False
        self.trans_decoder.trainable = False

        self.final_layer = Masked_Fusion('Masked_Fusion', d_final, dropout_rate)

    def call(self, flow_hist, trans_hist, ex_hist, flow_curr, trans_curr, ex_curr, training):
        flow_enc_inputs = tf.concat([flow_hist, flow_curr[:, :, :, 1:, :]], axis=-2)
        trans_enc_inputs = tf.concat([trans_hist, trans_curr[:, :, :, 1:, :]], axis=-2)
        ex_enc_inputs = tf.concat([ex_hist, ex_curr[:, 1:, :]], axis=-2)

        flow_dec_input = flow_curr[:, :, :, -1:, :]
        trans_dec_input = trans_curr[:, :, :, -1:, :]
        ex_dec_input = ex_curr[:, -1:, :]

        enc_output_flow = self.flow_encoder(flow_enc_inputs, ex_enc_inputs, training, None)
        enc_output_trans = self.trans_encoder(trans_enc_inputs, ex_enc_inputs, False, None)

        dec_output_flow, attention_weights_t = self.flow_decoder(flow_dec_input, ex_dec_input, enc_output_flow,
                                                                 training, None, None)
        dec_output_trans, _ = self.trans_decoder(trans_dec_input, ex_dec_input, enc_output_trans, False, None, None)

        final_output = self.final_layer(dec_output_flow, dec_output_trans, training)

        return final_output, attention_weights_t

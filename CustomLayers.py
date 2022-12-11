from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
"""
This file contains custom layers for ML_composer
"""

"""
Positional encoding layer
"""
def dot_product(x, kernel):
   if K.backend() == 'tensorflow':
       return K.squeeze(K.dot(x, K.expand_dims(kernel)),axis=-1)
   else:
       return K.dot(x, kernel)

class PositionalEncoding(layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=np.arange(position)[:, np.newaxis],
            i=np.arange(d_model)[np.newaxis, :],
            d_model=d_model)
        # apply sin to even indices in the array; 2i
        sines = np.sin(angle_rads[:, 0::2])
        # apply cos to odd indices in the array; 2i+1
        cosines = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        #return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
        return self.pos_encoding[:, :tf.shape(inputs)[1], :]

##Create a self attention layer with weights
class PosAttention(layers.Layer):
    def __init__(self, **kwargs):
        super(PosAttention, self).__init__(**kwargs)


    def build(self, input_shape):
        assert len(input_shape) >= 2
        attention_dim = input_shape[-1]
        amount_size = input_shape[-1]
        #input_shape = tf.TensorShape(input_shape)
        #input_channel = self._get_input_channel(input_shape)
        self.W1 = self.add_weight(name='Attention_weight', shape=(amount_size,attention_dim),
                                  initializer='normal', trainable=True)

        self.W2 = self.add_weight(name='Attention_weight', shape=(amount_size,attention_dim),
                                  initializer='normal', trainable=True)
        self.V = layers.Dense(1, use_bias=False)
        self.built = True
        super(PosAttention, self).build(input_shape)

    def call(self, x, pos):
        # require constant attention score from attention layer
        # x shape == (batch_size, seq_len,seq_len, d_model)
        pos_attention = dot_product(pos, self.W1)

        tanh_attention = tf.nn.tanh(pos_attention)
        alpha = tf.nn.softmax(dot_product(tanh_attention, self.W2), axis=1)

        epi = dot_product(tf.nn.softmax(tanh_attention, axis=1), x)

        # score shape == (batch_size, seq_len, seq_len, 1)


        return K.sum(epi, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        return super(PosAttention, self).get_config()


def residual_fl_block(input, width, activation=layers.ReLU(),downsample=False):
    #residual fully connected layers block
    X = layers.Dense(width)(input)
    if activation == layers.ReLU():
        X = layers.BatchNormalization()(X)

    if downsample:
        out = layers.Add()([X, input])
        out = activation(out)
        return out
    else:
        X = activation(X)
        return X



def residual_conv1D_block(input,filters,kernel_size,activation=layers.ReLU(),downsample=False):
    X = layers.Conv1D(filters, kernel_size, padding="same")(input)
    X = layers.BatchNormalization()(X)
    X = activation(X)

    X1 = layers.Conv1D(filters, kernel_size, padding="same")(X)
    X1 = layers.BatchNormalization()(X1)

    if downsample:
        input = layers.Conv1D(filters, 1, padding="same")(input)
        input = layers.BatchNormalization()(input)
        out = layers.Add()([X1, input])
    else:
        out = layers.Add()([X1])

    out = activation(out)
    return out
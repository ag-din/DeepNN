# -*- coding: utf-8 -*-

import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.initializers import he_normal, glorot_normal
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU, ThresholdedReLU

def create_dense_network(input_shape, target_dim):

    model = Sequential()

    model.add(Dense(2000, input_shape=input_shape, kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5))

    model.add(Dense(2000, kernel_initializer='he_normal'))
    model.add(PReLU(alpha_initializer='zeros'))
    model.add(Dropout(0.5))

    model.add(Dense(2000, kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5))

    model.add(Dense(2000, kernel_initializer='he_normal'))
    model.add(PReLU(alpha_initializer='zeros'))
    model.add(Dropout(0.5))

    model.add(Dense(target_dim, activation='softmax', kernel_initializer='glorot_normal'))

    return model




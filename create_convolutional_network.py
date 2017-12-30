# -*- coding: utf-8 -*-

import tensorflow as tf
tf.python.control_flow_ops = tf

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, LocallyConnected2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU, ThresholdedReLU


def create_convolutional_network(input_shape, target_dim):

    model = Sequential()

    model.add(Conv2D(96, (5, 5), padding='same', input_shape=input_shape, activation='relu', kernel_initializer='he_normal'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization(scale=False))
    model.add(Conv2D(192, (5, 5), padding='same', input_shape=input_shape, activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(2000, input_shape=input_shape, kernel_initializer='he_normal'))
    model.add(PReLU(alpha_initializer='zeros'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, kernel_initializer='he_normal'))
    model.add(PReLU(alpha_initializer='zeros'))
    model.add(Dropout(0.5))

    model.add(Dense(target_dim, activation='softmax', kernel_initializer='glorot_normal'))


    return model

# -*- coding: utf-8 -*-

import tensorflow as tf
tf.python.control_flow_ops = tf

from architecture import Architecture
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, LocallyConnected2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.initializers import he_normal, glorot_normal
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU, ThresholdedReLU


TYPE = "convolutional"

def load_convolutional_network_architectures():
    architectures = []

    architectures.append(Architecture("example1", TYPE, sample_network_constructor1))
    architectures.append(Architecture("example2", TYPE, sample_network_constructor2))
    architectures.append(Architecture("example3", TYPE, sample_network_constructor3))
    architectures.append(Architecture("example4", TYPE, sample_network_constructor4))
    architectures.append(Architecture("example5", TYPE, sample_network_constructor5))
    architectures.append(Architecture("example6", TYPE, sample_network_constructor6))
    architectures.append(Architecture("example7", TYPE, sample_network_constructor7))
    architectures.append(Architecture("example8", TYPE, sample_network_constructor8))
    architectures.append(Architecture("example9", TYPE, sample_network_constructor9))
    architectures.append(Architecture("example10", TYPE, sample_network_constructor10))
    architectures.append(Architecture("example11", TYPE, sample_network_constructor11))

    return architectures


def sample_network_constructor1(input_shape, target_dim):

    model = Sequential()

    model.add(Conv2D(64, (5, 5), padding='same', input_shape=input_shape, activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization(scale=False))

    model.add(Conv2D(64, (5, 5), padding='same', activation='relu', kernel_initializer='he_normal'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization(scale=False))

    model.add(Conv2D(64, (5, 5), padding='same', activation='relu', kernel_initializer='he_normal'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(LocallyConnected2D(16, (3, 3)))
    model.add(Dropout(0.50))

    model.add(Flatten())
    model.add(Dense(target_dim, activation='softmax', kernel_initializer='glorot_normal'))

    return model


def sample_network_constructor2(input_shape, target_dim):

    model = Sequential()

    model.add(Conv2D(64, (5, 5), padding='same', input_shape=input_shape, activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization(scale=False))

    model.add(Conv2D(64, (5, 5), padding='same', activation='relu', kernel_initializer='he_normal'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization(scale=False))

    model.add(Conv2D(128, (5, 5), padding='same', activation='relu', kernel_initializer='he_normal'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(target_dim, activation='softmax', kernel_initializer='glorot_normal'))

    return model


def sample_network_constructor3(input_shape, target_dim):

    model = Sequential()

    model.add(Conv2D(96, (3, 3), padding='same', input_shape=input_shape, activation='relu', kernel_initializer='he_normal'))
    model.add(Conv2D(96, (3, 3), padding='same', input_shape=input_shape, activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(192, (3, 3), padding='same', input_shape=input_shape, activation='relu', kernel_initializer='he_normal'))
    model.add(Conv2D(192, (3, 3), padding='same', input_shape=input_shape, activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(192, (3, 3), padding='same', input_shape=input_shape, activation='relu', kernel_initializer='he_normal'))
    model.add(Conv2D(192, (1, 1), padding='same', input_shape=input_shape, activation='relu', kernel_initializer='he_normal'))
    model.add(Conv2D(10, (1, 1), padding='same', input_shape=input_shape, activation='relu', kernel_initializer='he_normal'))
    model.add(GlobalAveragePooling2D(data_format=None))

    model.add(Dense(target_dim, activation='softmax', kernel_initializer='glorot_normal'))

    return model


def sample_network_constructor4(input_shape, target_dim):

    model = Sequential()

    model.add(Conv2D(96, (5, 5), input_shape=input_shape, padding='same', activation='relu', kernel_initializer='he_normal'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization(scale=False))

    model.add(Conv2D(64, (5, 5), padding='same', activation='relu', kernel_initializer='he_normal'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(target_dim, activation='softmax', kernel_initializer='glorot_normal'))

    return model


def sample_network_constructor5(input_shape, target_dim):

    model = Sequential()

    model.add(Conv2D(256, (3, 3), input_shape=input_shape, padding='same', activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(LocallyConnected2D(64, (3, 3)))
    model.add(Dropout(0.50))

    model.add(Flatten())

    model.add(Dense(target_dim, activation='softmax', kernel_initializer='glorot_normal'))

    return model


def sample_network_constructor6(input_shape, target_dim):

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


def sample_network_constructor7(input_shape, target_dim):

    model = Sequential()

    model.add(Conv2D(64, (5, 5), padding='same', input_shape=input_shape, activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization(scale=False))
    model.add(Conv2D(128, (5, 5), padding='same', input_shape=input_shape, activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(1000, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.5))
        
    model.add(Dense(target_dim, activation='softmax', kernel_initializer='glorot_normal'))

    return model


def sample_network_constructor8(input_shape, target_dim):

    model = Sequential()

    model.add(Conv2D(96, (5, 5), padding='same', input_shape=input_shape, activation='relu', kernel_initializer='he_normal'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(BatchNormalization(scale=False))
    model.add(Conv2D(96, (5, 5), padding='same', input_shape=input_shape, activation='relu', kernel_initializer='he_normal'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(BatchNormalization(scale=False))
    model.add(Conv2D(96, (5, 5), padding='same', input_shape=input_shape, activation='relu', kernel_initializer='he_normal'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(BatchNormalization(scale=False))
    model.add(Conv2D(96, (5, 5), padding='same', input_shape=input_shape, activation='relu', kernel_initializer='he_normal'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(BatchNormalization(scale=False))
    model.add(Conv2D(96, (5, 5), padding='same', input_shape=input_shape, activation='relu', kernel_initializer='he_normal'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(target_dim, activation='softmax', kernel_initializer='glorot_normal'))

    return model


def sample_network_constructor9(input_shape, target_dim):

    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization(scale=False))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(BatchNormalization(scale=False))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization(scale=False))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(BatchNormalization(scale=False))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization(scale=False))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(target_dim, activation='softmax', kernel_initializer='glorot_normal'))

    return model


def sample_network_constructor10(input_shape, target_dim):

    model = Sequential()

    model.add(Conv2D(32, (5, 5), padding='same', input_shape=input_shape, activation='relu', kernel_initializer='he_normal'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization(scale=False))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(BatchNormalization(scale=False))

    model.add(Flatten())

    model.add(Dense(1000, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.5))

    model.add(Dense(target_dim, activation='softmax', kernel_initializer='glorot_normal'))

    return model


def sample_network_constructor11(input_shape, target_dim):

    model = Sequential()

    model.add(Conv2D(64, (5, 5), padding='same', input_shape=input_shape, activation='relu', kernel_initializer='he_normal'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization(scale=False))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(BatchNormalization(scale=False))

    model.add(Flatten())

    model.add(Dense(target_dim, activation='softmax', kernel_initializer='glorot_normal'))

    return model

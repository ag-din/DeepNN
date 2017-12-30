# -*- coding: utf-8 -*-

import tensorflow as tf
tf.python.control_flow_ops = tf

from architecture import Architecture
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.initializers import he_normal, glorot_normal
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU, ThresholdedReLU

TYPE = "dense"

def load_dense_network_architectures():
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


# sample constructors
def sample_network_constructor1(input_shape, target_dim):
    model = Sequential()
    
    model.add(Dense(2000, input_shape=input_shape, activation='sigmoid', kernel_initializer='glorot_normal'))
    model.add(Dropout(0.50))
    model.add(Dense(2000, activation='sigmoid', kernel_initializer='glorot_normal'))
    model.add(Dropout(0.50))
    model.add(Dense(target_dim, activation='softmax', kernel_initializer='glorot_normal'))

    return model

def sample_network_constructor2(input_shape, target_dim):
    model = Sequential()
    
    model.add(Dense(4000, input_shape=input_shape, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.50))
    model.add(Dense(1000, activation='linear', kernel_initializer='glorot_normal'))
    model.add(Dense(4000, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.50))

    model.add(Dense(target_dim, activation='softmax', kernel_initializer='glorot_normal'))

    return model

def sample_network_constructor3(input_shape, target_dim):
    model = Sequential()
    
    model.add(Dense(4000, input_shape=input_shape, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(4000, activation='relu', kernel_initializer='he_normal'))

    model.add(Dense(target_dim, activation='softmax', kernel_initializer='glorot_normal'))

    return model


def sample_network_constructor4(input_shape, target_dim):
    model = Sequential()
    
    model.add(Dense(2000, input_shape=input_shape, activation='tanh', kernel_initializer='glorot_normal'))
    model.add(Dense(500, activation='linear', kernel_initializer='glorot_normal'))
    model.add(Dense(1000, activation='tanh', kernel_initializer='glorot_normal'))
    model.add(Dense(250, activation='linear', kernel_initializer='glorot_normal'))
    model.add(Dense(500, activation='tanh', kernel_initializer='glorot_normal'))

    model.add(Dense(target_dim, activation='softmax', kernel_initializer='glorot_normal'))

    return model


def sample_network_constructor5(input_shape, target_dim):
    model = Sequential()

    model.add(Dense(3000, input_shape=input_shape, activation='sigmoid', kernel_initializer='glorot_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='linear', kernel_initializer='glorot_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(2000, activation='tanh', kernel_initializer='glorot_normal'))
    model.add(Dropout(0.5))

    model.add(Dense(target_dim, activation='softmax', kernel_initializer='glorot_normal'))

    return model


def sample_network_constructor6(input_shape, target_dim):
    model = Sequential()

    model.add(Dense(1000, input_shape=input_shape, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(200, activation='linear', kernel_initializer='glorot_normal'))
    model.add(Dense(1000, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(200, activation='linear', kernel_initializer='glorot_normal'))
    model.add(Dense(1000, activation='relu', kernel_initializer='he_normal'))

    model.add(Dense(target_dim, activation='softmax', kernel_initializer='glorot_normal'))

    return model


def sample_network_constructor7(input_shape, target_dim):
    model = Sequential()

    model.add(Dense(1000, input_shape=input_shape, activation='tanh', kernel_initializer='glorot_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(2000, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(3000, activation='tanh', kernel_initializer='glorot_normal'))
    model.add(Dropout(0.5))

    model.add(Dense(target_dim, activation='softmax', kernel_initializer='glorot_normal'))

    return model



def sample_network_constructor8(input_shape, target_dim):
    model = Sequential()

    model.add(Dense(5000, input_shape=input_shape, kernel_initializer='he_normal'))
    model.add(PReLU(alpha_initializer='zeros'))
    model.add(Dropout(0.5))
    model.add(Dense(2000, kernel_initializer='he_normal'))
    model.add(PReLU(alpha_initializer='zeros'))
    model.add(Dropout(0.5))
    model.add(Dense(target_dim, activation='softmax', kernel_initializer='glorot_normal'))

    return model
    


def sample_network_constructor9(input_shape, target_dim):
    model = Sequential()

    model.add(Dense(5000, input_shape=input_shape, kernel_initializer='he_normal'))
    model.add(PReLU(alpha_initializer='zeros'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='linear', kernel_initializer='glorot_normal'))
    model.add(Dense(2000, kernel_initializer='he_normal'))
    model.add(PReLU(alpha_initializer='zeros'))
    model.add(Dropout(0.5))
    model.add(Dense(target_dim, activation='softmax', kernel_initializer='glorot_normal'))

    return model



def sample_network_constructor10(input_shape, target_dim):

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



def sample_network_constructor11(input_shape, target_dim):

    model = Sequential()

    model.add(Dense(3000, kernel_initializer='he_normal', input_shape=input_shape))
    model.add(PReLU(alpha_initializer='zeros'))
    model.add(Dropout(0.5))

    model.add(Dense(2000, kernel_initializer='he_normal'))
    model.add(ThresholdedReLU(theta=1.0))
    model.add(Dropout(0.5))

    model.add(Dense(target_dim, activation='softmax', kernel_initializer='glorot_normal'))

    return model




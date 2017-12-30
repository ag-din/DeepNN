# -*- coding: utf-8 -*-
import numpy as np
from sklearn.preprocessing import OneHotEncoder



#
# Esta función sirve para preparar los datos para que sea construida una red neuronal convolucional.
# Lo que debe hacer dicha función es devolver los arreglos de numpy X_train, X_test
# con dimensiones 50000x32x32x3 y 10000x32x32x3 respectivamente.
#
# En el caso de X_train e X_test, hay que reacomodar los arreglos para formar arreglos tridimensionales
# de 32x32 pixeles x 3 canales de color (RGB).
#

def preprocess_data(X_train, X_test):
    X_train = np.reshape(X_train, (50000, 3, 32, 32)).transpose(0, 2, 3, 1)
    X_test = np.reshape(X_test, (10000, 3, 32, 32)).transpose(0, 2, 3, 1)

    return X_train, X_test


# Lo que debe hacer dicha función es devolver los arreglos de numpy y_train, y_test
# con dimensiones 50000x10 y 10000x10 respectivamente.
#
# En el caso de y_train e y_test, los valores (entre 0 y 9) hay que convertirlos a una representación
# one-hot. En donde hay un bit correspondiente a cada clase y un valor de 1 para aquella clase a la
# que pertenece la instancia. e.g. 0001000000 para la clase identificada con el valor 3.
#

def preprocess_labels(y_train, y_test):
    from keras.utils.np_utils import to_categorical
    return to_categorical(y_train, 10), to_categorical(y_test, 10)


# Funcion para preprocesar los datos para una red densa.
#

def preprocess_data_dense(X_train, X_test, y_train, y_test):
    Y_train, Y_test = preprocess_labels(y_train, y_test)

    return X_train, X_test, Y_train, Y_test


# Función para preprocesar los datos para una red convolucional.
#

def preprocess_data_convolutional(X_train, X_test, y_train, y_test):
    X_train, X_test = preprocess_data(X_train, X_test)
    Y_train, Y_test = preprocess_labels(y_train, y_test)

    return X_train, X_test, Y_train, Y_test

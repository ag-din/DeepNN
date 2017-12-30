# -*- coding: utf-8 -*-
import sys
import numpy as np

#-------------------------------------------
# Descripción de los datos y como accederlos
#-------------------------------------------
# Fuente: #http://www.cs.toronto.edu/~kriz/cifar.html
#
# The archive contains the files data_batch_1, data_batch_2, ..., data_batch_5, as well as test_batch. Each of these files is a Python "pickled" object produced with cPickle.
# Here is a Python routine which will open such a file and return a dictionary:

def unpickle(file):
    fo = open(file, 'rb')
    if sys.version_info[0] > 2:  # python 3
        import _pickle as cPickle
        dict = cPickle.load(fo, encoding='latin1')
    else:  # python 2
        import cPickle
        dict = cPickle.load(fo)
    fo.close()
    return dict

    
# Loaded in this way, each of the batch files contains a dictionary with the following elements:
#
#    - data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
#    - labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.
#
# The dataset contains another file, called batches.meta. It too contains a Python dictionary object. It has the following entries:
#
#    - label_names -- a 10-element list which gives meaningful names to the numeric labels in the labels array described above. For example, label_names[0] == "airplane", label_names[1] == "automobile", etc.

#-----------------------------------------
# Función para cargar los datos en memoria
#-----------------------------------------
#
# modo de uso:
#	unpickle("data/cifar-10-batches-py/data_batch_1")
#

#--------------------------
# Carga de datos en memoria
#--------------------------
#Función para cargar los datos en memoria. Esta función debería retornar X_train, X_test, y_train, y_test
#los cuales deberían ser arreglos de numpy con dimensiones 50000x3072, 10000x3072, 50000x1 y 10000x1 respectivamente.
#

def load_data():
    b_1_train=unpickle("data/cifar-10-batches-py/data_batch_1")
    b_2_train=unpickle("data/cifar-10-batches-py/data_batch_2")
    b_3_train=unpickle("data/cifar-10-batches-py/data_batch_3")
    b_4_train=unpickle("data/cifar-10-batches-py/data_batch_4")
    b_5_train=unpickle("data/cifar-10-batches-py/data_batch_5")
    b_test=unpickle("data/cifar-10-batches-py/test_batch")
    
    b_1_train_x = b_1_train['data']
    b_1_train_y = b_1_train['labels']

    b_2_train_x = b_2_train['data']
    b_2_train_y = b_2_train['labels']

    b_3_train_x = b_3_train['data']
    b_3_train_y = b_3_train['labels']

    b_4_train_x = b_4_train['data']
    b_4_train_y = b_4_train['labels']

    b_5_train_x = b_5_train['data']
    b_5_train_y = b_5_train['labels']

    X_test = b_test['data']
    y_test = np.array(b_test['labels'])

    X_train = np.concatenate((b_1_train_x,b_2_train_x,b_3_train_x,b_4_train_x,b_5_train_x))
    y_train = np.concatenate((b_1_train_y,b_2_train_y,b_3_train_y,b_4_train_y,b_5_train_y))
    
    return X_train, X_test, y_train, y_test


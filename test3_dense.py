# -*- coding: utf-8 -*-

import numpy as np

from keras.optimizers import Adam
from load_data import load_data
from preprocess_data import preprocess_data_dense
from create_dense_network import *
from keras.callbacks import Callback
import time

import numpy as np

from auxx import show_confusion_matrix
from auxx import save_confusion_matrix
from auxx import to_binary

from keras.wrappers.scikit_learn import KerasClassifier
from re import search


class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


seed = 7
np.random.seed(seed)

# load data
print(">>> carga de datos...")
X_train, X_test, y_train, y_test = load_data()
print("OK")

########para hacer un preproceso sobre los datos mismos#########
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
### mean subtraction
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_test -= mean_image
### normalization
X_train /= np.std(X_train, axis = 0)
X_test /= np.std(X_test, axis = 0)

##################################################################


# preprocess data
print("\nPreprocessing dataset...")
X_train, X_test, Y_train, Y_test = preprocess_data_dense(X_train, X_test, y_train, y_test)

   
# types
print()
print(">>> tipos...")
assert isinstance(X_train, np.ndarray), "el tipo de X_train es %s y debería ser <class 'numpy.ndarray'>" % type(X_train)
assert isinstance(X_test, np.ndarray), "el tipo de X_test es %s y debería ser <class 'numpy.ndarray'>" % type(X_test)

assert isinstance(Y_train, np.ndarray), "el tipo de Y_train es %s y debería ser <class 'numpy.ndarray'>" % type(y_train)
assert isinstance(Y_test, np.ndarray), "el tipo de Y_test es %s y debería ser <class 'numpy.ndarray'>" % type(y_test)
print("OK")

# dimensions
print()
print(">>> dimensiones...")
assert X_train.shape == (50000, 3072), "las dimensiones de X_train son %s y deberían ser (50000, 3072)" % str(
    X_train.shape)
assert X_test.shape == (10000, 3072), "las dimensiones de X_test son %s y deberían ser (10000, 3072)" % str(
    X_test.shape)

assert Y_train.shape == (50000, 10), "las dimensiones de Y_train son %s y deberían ser (50000, 10)" % str(Y_train.shape)
assert Y_test.shape == (10000, 10), "las dimensiones de Y_test son %s y deberían ser (10000, 10)" % str(Y_test.shape)
print("OK")

print()
print(">>> entrenando modelo...")
# build network

# -----------------------
# getting data dimensions
input_shape = X_train.shape[1:]
if Y_train.shape[1:] == ():
    target_dim = 1
else:
    target_dim = Y_train.shape[1:][0]




def create_and_compile_model(input_shape, target_dim):

    # create model
    model = create_dense_network(input_shape, target_dim)
    optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # compile
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

############# para ver modelo
model = create_dense_network(input_shape, target_dim)
print(model.summary())

model_a = KerasClassifier(build_fn=create_and_compile_model, input_shape=input_shape, target_dim=target_dim, epochs=500, batch_size=5000)


# fit model

time_callback = TimeHistory()

model_a.fit(X_train, Y_train, callbacks=[time_callback]) 

times = time_callback.times

print(np.sum(times))





##########################
#from sklearn.metrics import confusion_matrix

Y_pred = model_a.predict(X_test)

#conf_matrix = confusion_matrix(to_binary(Y_test), Y_pred)
#print()
#classes_names = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#save_confusion_matrix("MD_confusion.pdf", conf_matrix, classes=classes_names)

print("\nDone")

from keras import backend

backend.clear_session()



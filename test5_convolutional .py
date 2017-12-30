# -*- coding: utf-8 -*-

import numpy as np

from keras.optimizers import Adam
from load_data import load_data
from preprocess_data import preprocess_data_convolutional

from keras.models import model_from_json

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, LocallyConnected2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU, ThresholdedReLU


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
X_train, X_test, Y_train, Y_test = preprocess_data_convolutional(X_train, X_test, y_train, y_test)


print(">>> entrenando modelo...")
# build network

# -----------------------
# getting data dimensions
input_shape = X_train.shape[1:]
if Y_train.shape[1:] == ():
    target_dim = 1
else:
    target_dim = Y_train.shape[1:][0]


model = Sequential()

model.add(Conv2D(64, (5, 5), padding='same', input_shape=input_shape, activation='relu', kernel_initializer='he_normal'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(BatchNormalization(scale=False))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(BatchNormalization(scale=False))

model.add(Flatten())

model.add(Dense(target_dim, activation='softmax', kernel_initializer='glorot_normal'))






model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=300, batch_size=30)


# serialize model to JSON
model_json = model.to_json()

with open("model_CNN11.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model_CNN11.h5")

print("Saved model to disk")

print("\nDone")








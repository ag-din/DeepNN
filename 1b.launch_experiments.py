# -*- coding: utf-8 -*-
# keras experimentation template

# -------------------
# imports
import numpy as np
import sys
import os
from load_data import load_data
#from preprocess_data import preprocess_data_dense
from preprocess_data import preprocess_data_convolutional
#from architectures_dense import load_dense_network_architectures
from architectures_convolutional import load_convolutional_network_architectures
from evaluate_single_model import evaluate_single_model
from checkpointing import *

# -------------------
# parameters
#try:
#    model_type = sys.argv[1]  # dense/convolutional
#except:
#    model_type = "dense"

#model_type = "dense"
model_type = "convolutional"

seed = 7
np.random.seed(seed)
test_size = 0.33

dataset = ""
# grid search
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
# optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
optimizers = ['Adam']

# -------------------
# functions
#    wrappers scikit-learn
#    modelos

print("Experiment with " + model_type + " networks...")

preprocess_data = preprocess_data_convolutional
architectures = load_convolutional_network_architectures()

#preprocess_data = preprocess_data_dense
#architectures = load_dense_network_architectures()

#if model_type == "dense":
#    preprocess_data = preprocess_data_dense
#    architectures = load_dense_network_architectures()
#else:
#    preprocess_data = preprocess_data_convolutional
#    architectures = load_convolutional_network_architectures()


# -------------------
# load data
print("\nLoading dataset...")
X_train, X_test, y_train, y_test = load_data()

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


# -------------------
# preprocess data
print("\nPreprocessing dataset...")
X_train, X_test, Y_train, Y_test = preprocess_data(X_train, X_test, y_train, y_test)


# -------------------
# architecture selection

### yo
#os.remove(".checkpointdense")
###
#os.remove(".checkpointconvolutional")


run = 0
checkpoint_run = read_checkpoint_run(model_type)
# print("checkpoint: " + str(checkpoint_run))

for architecture in architectures:
    if run <= checkpoint_run:
        print(">>> skipping experiment " + str(run) + ": " + str(architecture))
    else:
        print(">>> executing experiment " + str(run) + ": " + str(architecture))
        evaluate_single_model(run, X_train, X_test, Y_train, Y_test, architecture)
        new_checkpoint(run, model_type)

    run += 1

print("DONE")

# -*- coding: utf-8 -*-
# keras experimentation template

# -------------------
# imports
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy
import sys
from load_data import load_data
from preprocess_data import preprocess_data_dense
from preprocess_data import preprocess_data_convolutional
from create_dense_network import *
from create_convolutional_network import *

# -------------------
# parameters
try:
    model_type = sys.argv[1]  # dense/convolutional
except:
    model_type = "dense"

seed = 7
numpy.random.seed(seed)
test_size = 0.33

dataset = ""
# grid search
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
# optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']


# -------------------
# functions
#    wrappers scikit-learn
#    modelos

print("Experiment with " + model_type + " networks...")

if model_type == "dense":
    preprocess_data = preprocess_data_dense
    create_model = create_dense_network
else:
    preprocess_data = preprocess_data_convolutional
    create_model = create_convolutional_network


def create_and_compile_model(input_shape, target_dim):
    # create
    model = create_model(input_shape, target_dim)

    # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


# -------------------
# load data
print("\nLoading dataset...")
X_train, X_test, y_train, y_test = load_data()

# -------------------
# preprocess data
print("\nPreprocessing dataset...")
X_train, X_test, Y_train, Y_test = preprocess_data(X_train, X_test, y_train, y_test)

# -----------------------
# getting data dimensions
input_shape = X_train.shape[1:]
if Y_train.shape[1:] == ():
    target_dim = 1
else:
    target_dim = Y_train.shape[1:][0]

# -------------------
# model selection
print("\nSelecting best model...")
# model design
model = KerasClassifier(build_fn=create_and_compile_model, nb_epoch=10, input_shape=input_shape, target_dim=target_dim)

# hyperparameter search
param_grid = dict(batch_size=batch_size, nb_epoch=epochs)
print(" > hyperparameter space: " + str(param_grid))
# search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=5, refit=True, verbose=0)
search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_jobs=1, cv=5, n_iter=15, refit=True, verbose=0)

print(" > searching...")
search_result = search.fit(X_train, Y_train)

# best model
best_score = search_result.best_score_
best_params = search_result.best_params_
best_model = search_result.best_estimator_.model
print("Best model: %f using %s" % (search_result.best_score_, search_result.best_params_))

# -------------------
# evaluate model
# http://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
# calculate predictions
print("\nEvaluating model...")
# evaluate the model
scores = best_model.evaluate(X_test, y_test)
print("%s: %.2f%%" % (best_model.metrics_names[1], scores[1] * 100))
# predictions = search.predict(X_test)
# predictions = best_model.predict(X_test)####best_model era search!

# -------------------
# save results

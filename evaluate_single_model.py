# -*- coding: utf-8 -*-
# keras experimentation template

# -------------------
# imports
from keras.wrappers.scikit_learn import KerasClassifier
from re import search

from sklearn.model_selection import RandomizedSearchCV

from auxx import show_confusion_matrix
from auxx import save_confusion_matrix
from auxx import to_binary

from keras.optimizers import SGD
from keras.optimizers import Adam

import numpy as np

import os


# -------------------
# parameters
seed = 7
np.random.seed(seed)
test_size = 0.33

# hyper-parameter search
N_ITER_RANDOMIZED_SEARCH = 8


epochs = [30, 100, 300]
batch_size = [30, 100, 300]
#batch_size = [800, 1000]
# optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
optimizers = ['Adam']


def create_and_compile_model(architecture, input_shape, target_dim, optimizer):
    print(" > " + str(architecture) + " input_dim:" + str(input_shape) + " output_dim:" + str(target_dim) + " " + optimizer)

    # create model
    model = architecture.create_model(input_shape, target_dim)
    #optimizerr = SGD(lr=0.000005, momentum=0.9, nesterov=True)
    optimizerr = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)# decay=0.0)
    # compile
    model.compile(loss='categorical_crossentropy', optimizer=optimizerr, metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer=optimizers, metrics=['accuracy'])

    return model


def evaluate_single_model(evaluation_id, X_train, X_test, Y_train, Y_test, architecture):
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
    optimizer = optimizers[0]

    model = KerasClassifier(build_fn=create_and_compile_model, architecture=architecture, input_shape=input_shape, target_dim=target_dim, optimizer=optimizer)

    # hyper-parameter search
    param_grid = dict(batch_size=batch_size, epochs=epochs)  # , optimizer=optimizers)
    search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_jobs=1, cv=5, n_iter=N_ITER_RANDOMIZED_SEARCH, refit=True,
                                verbose=0)

    search_result = search.fit(X_train, Y_train)

    # best model
    # best_score = search_result.best_score_
    # best_params = search_result.best_params_
    best_model = search_result.best_estimator_.model
    # print("Best model: %f using %s" % (search_result.best_score_, search_result.best_params_))

    # -------------------
    # evaluate model
    # http://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
    # calculate predictions
    print("\nEvaluating model...")
    # evaluate the model
    scores = best_model.evaluate(X_test, Y_test)
    print()
    print("Summary")
    print("-------")
    print("Architecture: " + str(architecture))
    print("Best model: %f using %s" % (search_result.best_score_, search_result.best_params_))
    for metric_index in range(len(scores)):
        print("%s: %.2f" % (best_model.metrics_names[metric_index], scores[metric_index]))
    # predictions = search.predict(X_test)
    # predictions = best_model.predict(X_test)


    from sklearn.metrics import confusion_matrix
    Y_pred = search.predict(X_test)
    conf_matrix = confusion_matrix(to_binary(Y_test), Y_pred)
    print()
    #show_confusion_matrix(conf_matrix, classes=range(10))
    classes_names = ('avion', 'auto', 'ave', 'gato', 'ciervo', 'perro', 'rana', 'caballo', 'barco', 'camion')
    save_confusion_matrix(str(architecture) + "-confusion.pdf", conf_matrix, classes=classes_names)

    """
    ######
    mode_fit = model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    model_score = model.score(X_test, Y_test)
    print()

    from sklearn.metrics import confusion_matrix
    conf_matrix = confusion_matrix(to_binary(Y_test), Y_pred)
    show_confusion_matrix(conf_matrix, classes=range(10))
    """

    # -------------------
    # save results
    print("\nSaving results...")
    results_filename = "results.txt"
    with open(results_filename, "a") as results_file:
        best_model_string = "%f using %s" % (search_result.best_score_, search_result.best_params_)
        header = "evaluation-id;architecture-id;architecture-type;best-hyper-parameters"
        row = str(evaluation_id) + ";" + str(architecture.id) + ";" + str(architecture.type) + ";" + best_model_string
        for metric_index in range(len(scores)):
            header += ";" + best_model.metrics_names[metric_index]
            row += ";" + str(scores[metric_index])
            # best_scores_string = "%s: %.2f%%" % (best_model.metrics_names[1], scores[1] * 100)
            # best_scores_string = "%.2f%%" % (best_model.metrics_names[1], scores[1] * 100)

        if os.stat(results_filename).st_size == 0:
            results_file.write(header + "\n")
        results_file.write(row + "\n")
        results_file.close()

    print("\nDone")
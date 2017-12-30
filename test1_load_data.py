# -*- coding: utf-8 -*-

import numpy as np
from load_data import load_data

#load data
print(">>> carga de datos...")
X_train, X_test, y_train, y_test = load_data()
print("OK")

#perform tests
print()
print(">>> tipos...")

assert isinstance(X_train, np.ndarray), "el tipo de X_train es %s y debería ser <class 'numpy.ndarray'>" % type(X_train)
assert isinstance(X_test, np.ndarray), "el tipo de X_test es %s y debería ser <class 'numpy.ndarray'>" % type(X_test)

assert isinstance(y_train, np.ndarray), "el tipo de y_train es %s y debería ser <class 'numpy.ndarray'>" % type(y_train)
assert isinstance(y_test, np.ndarray), "el tipo de y_test es %s y debería ser <class 'numpy.ndarray'>" % type(y_test)
print("OK")

print()
print(">>> dimensiones...")
assert X_train.shape == (50000, 3072), "las dimensiones de X_train son %s y deberían ser (50000, 3072)" % X_train.shape
assert X_test.shape == (10000, 3072), "las dimensiones de X_test son %s y deberían ser (10000, 3072)" % X_test.shape

assert y_train.shape == (50000, ), "las dimensiones de y_train son %s y deberían ser (50000, )" % y_train.shape
assert y_test.shape ==(10000, ), "las dimensiones de y_test son %s y deberían ser (10000, )" % y_test.shape
print("OK")


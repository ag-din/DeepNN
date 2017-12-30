# -*- coding: utf-8 -*-

import numpy as np
from load_data import *
from preprocess_data import *
#from auxx import *

import matplotlib
matplotlib.use('Agg')


import matplotlib.pyplot as plt

#load data
print(">>> carga de datos...")
X_train, X_test, y_train, y_test = load_data()
print("OK")

#A=X_train #5000,3072
A=X_train[2,:]#3072
A1 = A[0:1024]
A2 = A[1024:2048]
A3 =A[2048:3072]
A1 =np.reshape(A1,[32,32])
A2 =np.reshape(A2,[32,32])
A3 =np.reshape(A3,[32,32])
YY=np.array([A1,A2,A3])

#k=np.reshape(A,[3,])
print(YY.shape)#3, 32, 32


fig = plt.figure (figsize=(1, 1), facecolor ='w')
axes1 = np.zeros ((1,1), dtype=np.object)
axes1 [0, 0] = fig.add_subplot (1, 1, 1)
plt.imshow (YY.T)
plt.axis ('off')
plt.show()
plt.savefig("img", format="pdf")




"""

########para hacer un preproceso sobre los datos mismos#########
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')



Y=np.reshape(X_train[5][:],[3,32,32])
R=Y[0,:,:].T
G=Y[1,:,:].T
B=Y[2,:,:].T
YY=np.array([R,G,B])



### mean subtraction
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_test -= mean_image
### normalization
X_train /= np.std(X_train, axis = 0)
X_test /= np.std(X_test, axis = 0)


W=np.reshape(X_train[5][:],[3,32,32])
R=W[0,:,:].T
G=W[1,:,:].T
B=W[2,:,:].T
WW=np.array([R,G,B])
##################################################################



#preprocess data
print()
print(">>> preproceso de datos...")
X_train, X_test = preprocess_data(X_train, X_test)
Y_train, Y_test = preprocess_labels(y_train, y_test)
print("OK")

#types
print()
print(">>> tipos...")

assert isinstance(X_train, np.ndarray), "el tipo de X_train es %s y debería ser <class 'numpy.ndarray'>" % type(X_train)
assert isinstance(X_test, np.ndarray), "el tipo de X_test es %s y debería ser <class 'numpy.ndarray'>" % type(X_test)

assert isinstance(Y_train, np.ndarray), "el tipo de Y_train es %s y debería ser <class 'numpy.ndarray'>" % type(Y_train)
assert isinstance(Y_test, np.ndarray), "el tipo de Y_test es %s y debería ser <class 'numpy.ndarray'>" % type(Y_test)
print("OK")

#dimensions
print()
print(">>> dimensiones...")
assert X_train.shape == (50000, 32, 32, 3), "las dimensiones de X_train son %s y deberían ser (50000, 32, 32, 3)" % str(X_train.shape)
assert X_test.shape == (10000, 32, 32, 3), "las dimensiones de X_test son %s y deberían ser (10000, 32, 32, 3)" % str(X_test.shape)

assert Y_train.shape == (50000, 10), "las dimensiones de Y_train son %s y deberían ser (50000, 10)" % str(Y_train.shape)
assert Y_test.shape == (10000, 10), "las dimensiones de Y_test son %s y deberían ser (10000, 10)" % str(Y_test.shape)
print("OK")

#data
print()
print(">>> datos...")
assert (Y_train.sum(axis=0) == np.array([5000.]*10)).all(), "las cantidad de instancias de cada clase en Y_train son %s y deberían ser 5000 de cada una" % str(Y_train.sum(axis=0))
assert (Y_test.sum(axis=0) == np.array([1000.]*10)).all(), "las cantidad de instancias de cada clase en Y_test son %s y deberían ser 5000 de cada una" % str(Y_test.sum(axis=0))
print("OK")



#K = X_train[0,:,:,:]



#print(K)


#visualize
#print()
print(">>> visualization...")

fig = plt.figure (figsize=(1, 1), facecolor ='w')
axes1 = np.zeros ((1,2), dtype=np.object)
axes1 [0, 0] = fig.add_subplot (1, 2, 1)
plt.imshow (YY.T)
plt.axis ('off')
axes1 [0, 1] = fig.add_subplot (1, 2, 2)
plt.imshow (WW.T.astype(float))
plt.axis ('off')
plt.show()
plt.savefig("img", format="pdf")

#visualize_random_images(X_train, 1)
#print("OK")

"""
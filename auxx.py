
# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size


def visualize_random_images(X, img_num):
    fig, axes1 = plt.subplots(img_num, img_num, figsize=(3, 3))
    for j in range(img_num):
        for k in range(img_num):
            i = np.random.choice(range(len(X)))
            axes1[j][k].set_axis_off()
            axes1[j][k].imshow(X[i:i + 1][0])
    plt.show()
    plt.savefig('img', format="pdf")

"""

# Visualizo img_num imégenes aleatorias de un conjunto de datos x (X_train, X_test)
def visualize_random_images (X, img_num):
    fig = plt.figure (figsize=(3,3), facecolor ='w')
    lst_img = []
    for i in range(img_num):
        r = np.random.choice(range(len(X)))
        #Y=np.reshape(X[r][:],[3,32,32])
        #R=Y[0,:,:].T
        #G=Y[1,:,:].T
        #B=Y[2,:,:].T
        #YY=np.array([R,G,B])
        #lst_img.append(YY) 
        lst_img.append(X[r]) 
    if img_num < 4:
        axes1 = np.zeros ((1,img_num), dtype=np.object)
        for i in range (img_num):
            axes1 [0, i] = fig.add_subplot (1, img_num, i+1)
            plt.imshow (lst_img[i].T)
            plt.axis ('off')
            plt.show()
    if img_num > 3:
        count=0
        k = int(img_num%2)
        if k == 0:
            w = int(img_num/2)
            axes1 = np.zeros ((2,w), dtype=np.object)
            for i in range (2):
                for j in range (w):
                    axes1 [i, j] = fig.add_subplot (2, w, count+1)         
                    plt.imshow (lst_img[count].T)
                    count+=1
                    plt.axis ('off')
            plt.show()
        if k != 0:
            count=0
            q = int((img_num-k)/2)
            axes1 = np.zeros ((2,q+1), dtype=np.object)
            for i in range (2):
                if i==1:
                    for j in range (k):
                        axes1 [i, j] = fig.add_subplot (2, q+1, count+1)
                        plt.imshow (lst_img[count].T)
                        count+=1
                        plt.axis ('off')
                for j in range (q):
                    axes1 [i, j] = fig.add_subplot (2, q+1, count+1)
                    plt.imshow (lst_img[count].T)
                    count+=1
                    plt.axis ('off')
            plt.show()

    plt.savefig('img', format="pdf")

"""

def plot_confusion_matrix(confusion_matrix, classes, normalize=False, title='CONFUSION MATRIX', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    :param confusion_matrix:
    :param classes:
    :param normalize:
    :param title:
    :param cmap:
    :return:
    """
    font1 = {'size': 16}

    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    
    #plt.title(title, fontdict=font1)
    plt.colorbar(fraction=0.0451, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion Matrix, without normalization')
    print(confusion_matrix)
    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        if normalize:
            value = str(confusion_matrix[i, j])
        else:
            value = "%.0f" % confusion_matrix[i, j]
        print(value)
        plt.text(j, i, value, fontsize=11, horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")
    plt.ylabel('Clase verdadera', fontsize = 14)
    plt.xlabel('Clase predicha', fontsize = 14)
    plt.tight_layout()


def show_confusion_matrix(confusion_matrix, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Creates a new figure and plots the confusion matrix.
    :param confusion_matrix:
    :param classes:
    :param normalize:
    :param title:
    :param cmap:
    :return:
    """
    plt.figure()
    plot_confusion_matrix(confusion_matrix, classes, normalize, title)
    plt.show()


def save_confusion_matrix(filename, confusion_matrix, classes, normalize=False, title='CONFUSION MATRIX', cmap=plt.cm.Blues):
    """
    Creates a new figure and plots the confusion matrix.
    :param confusion_matrix:
    :param classes:
    :param normalize:
    :param title:
    :param cmap:
    :return:
    """
    plt.figure(figsize=(8, 8), facecolor='w', edgecolor='w')
    plot_confusion_matrix(confusion_matrix, classes, normalize, title)
    
    plt.savefig(filename, format="pdf")


def to_binary(Y):
    """
    Converts from a one-hot representation to a binary one.
    :param Y: the one-hot representation matrix
    :return:
    """
    return np.where(Y == 1)[1]
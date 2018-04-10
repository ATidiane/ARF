# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from perceptron import *


################################################################################
#-------------------------------- Données USPS --------------------------------#
################################################################################

def load_usps(fn):
    """ Lire un fichier USPS """

    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)

    # Retourne la seconde colonne jusqu'à la dernière tmp[:,1:], ainsi que
    # la première colonne tmp[:,0]
    return tmp[:,1:],tmp[:,0].astype(int)


def show_usps(data):
    """ Fonction nous permettant de visualiser les doonées USPS """

    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="YlGnBu")
    plt.colorbar()
    plt.show()


def extract_usps(fichier, firstclass, secondclass):
    """ Permet de construire un ensemble de données datax, datay avec les
        données USPS, juste en lui spécifiant en paramètre:
        :param fichier: le fichier,
        :param firstclass: le premier chiffre dont on veut extraire les données
        :param secondclass: le second chiffre.
        :return: datax, datay
    """

    pixels, numbers = load_usps(fichier)
    indexes69 = np.hstack(np.where(numbers == i)[0] for i in (firstclass, secondclass))
    pixels69 = pixels[indexes69]
    numbers69 = numbers[indexes69]
    pixels69y = np.ones(pixels69.shape[0])

    for i in range(numbers69.shape[0]):
        if numbers69[i] == firstclass:
            pixels69y[i] = -1
        else:
            pixels69y[i] = 1

    return pixels69, pixels69y


def weight_matrix(class1, class2, fig, ax=plt):
    """ Prends deux classes et plot la matrice de poids correspondante à la
        classification entre les deux.
        :param fig: figure de subplots
        :param ax=plt: ax sur lequel on veut afficher la figure
        :return: un dessin de la matrice de poids
    """

    # Extraction des données usps des classes passées
    datax, datay = extract_usps("USPS_train.txt", class1, class2)
    dataTx, dataTy = extract_usps("USPS_test.txt", class1, class2)
    # Apprentissage des données usps
    perceptron_usps = Perceptron(loss=hinge,loss_g=hinge_g,max_iter=1,
                                 eps=0.1,kernel=None)
    perceptron_usps.fit(datax, datay, gradient_descent="batch")
    print("======Entre les classes {} et {}======\n".format(class1, class2))
    err_learning = perceptron_usps.score(datax, datay)
    err_test = perceptron_usps.score(dataTx,dataTy)
    print("Erreur : train %f, test %f\n"% (err_learning,
                                           err_test))
    ax.set_title("{} vs {}".format(class1, class2))
    matrix = ax.imshow(perceptron_usps.w.reshape((16,16)),interpolation="nearest",cmap="YlGnBu")
    fig.colorbar(matrix, ax=ax)

    return err_learning, err_test


def matrix_one_vs_all(class1):
    """ Prends en entrée une seule classe (0..9) et affiche les différentes
        matrices de poids quand on entraine le perceptron contre toutes les
        autres classes. Au final 9 matrices de poids.
        :param class1: 0..9
        :return: 9 figures de matrice de poids.
    """

    # Matrice de poids Une class1 vs All
    nrows, ncols = 3, 3
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, sharex=True, sharey=True)
    plt.suptitle("Matrice de poids, {} vs All".format(class1))

    class2 = 0
    for i in range(nrows):
        for j in range(ncols):
            if class2 == class1: class2 += 1
            weight_matrix(6, class2, fig, ax[i, j])
            class2 += 1

    plt.savefig("weight_matrix_{}vsAll".format(class1))


def error_curves(class1):
    """ Plots the errors curves based on the number of iterations, in this case
        till 10000 iterations and class1 vs All.
        :return: 2 courbes d'erreurs, l'une sur l'apprentissage et l'autre sur
        le test.
    """

    # Définitions des itérations
    x_iter = [1, 100, 200]

    # Now, let's plot
    cols, marks = ["red", "green", "blue", "orange", "black", "cyan", "yellow",
                   "magenta", "green"],[".","+","*","o","x","^", ',', 'v', '^']

    fig , (ax1, ax2) = plt.subplots(ncols=2, sharex=True)
    plt.suptitle("Courbes d’erreurs en apprentissage et en test en\n fonction du "
                 "nombre d'itérations")


    ax1.set_title("Apprentissage")
    ax2.set_title("Test")

    for i in range(9):
        if i == class1: continue
        # Extraction des données usps des classes passées
        datax, datay = extract_usps("USPS_train.txt", class1, i)
        dataTx, dataTy = extract_usps("USPS_test.txt", class1, i)

        # Calcul des erreurs en learning et en test pour les diff iterations
        err_learning, err_test = [], []
        for iter in x_iter:
            perceptron_usps = Perceptron(loss=hinge,loss_g=hinge_g,max_iter=iter,
                                       eps=0.1,bias=False)
            perceptron_usps.fit(datax, datay, gradient_descent="batch")
            err_learning.append(perceptron_usps.score(datax, datay))
            err_test.append(perceptron_usps.score(dataTx, dataTy))

        ax1.plot(x_iter, err_learning, c=cols[i], marker=marks[i],
                 label='{} vs {}'.format(class1, i))
        ax2.plot(x_iter, err_test, c=cols[i], marker=marks[i],
                 label='{} vs {}'.format(class1, i))

    ax1.legend(loc='lower left', bbox_to_anchor=(1.04, 0),
               ncol=1, fancybox=True, shadow=True)
    ax2.legend(loc='lower left', bbox_to_anchor=(1.04, 0),
               ncol=1, fancybox=True, shadow=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig("error_curves_{}vsAll".format(class1))
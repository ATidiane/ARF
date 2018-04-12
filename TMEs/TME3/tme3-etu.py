# -*- coding: utf-8 -*-
""" ARF - TME3, Descente de gradient
"""

import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from math import inf
from matplotlib import cm
from logisticRegression import *
from mpl_toolkits.mplot3d import Axes3D
from sklearn.naive_bayes import GaussianNB


def make_grid(data=None, xmin=-5, xmax=5, ymin=-5, ymax=5, step=20):
    """ Cree une grille sous forme de matrice 2d de la liste des points
    :return: une matrice 2d contenant les points de la grille, la liste x, la
    liste y
    """
    if data is not None:
        xmax, xmin, ymax, ymin = np.max(data[:,0]), np.min(data[:,0]),\
                                 np.max(data[:,1]), np.min(data[:,1])
    x, y = np.meshgrid(np.arange(xmin, xmax, (xmax - xmin) * 1./step),
                       np.arange(ymin, ymax, (ymax - ymin) * 1./step))
    grid = np.c_[x.ravel(), y.ravel()]
    return grid, x, y


def load_usps(filename):
    with open(filename,"r") as f:
        f.readline()
        data =[ [float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp = np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)


def optimize(fonc, dfonc, xinit, eps, max_iter=100):
    """ Applique l'algorithme du gradient.
    :param: fonc, la fonction à optimiser.
    :param: dfonc, le gradient de cette fonction.
    :param: xinit, le point initial.
    :param: eps, le pas du gradient.
    :param: max_iter, le nombre d'itérations.
    :return: un triplet (x_histo, f_histo, grad_histo).
    """

    x_histo, f_histo, grad_histo = [xinit], [fonc(*xinit)], [dfonc(*xinit)]

    for t in range(max_iter):
        # Mise à jour de x à l'instant t
        epsDeltaFx = eps * np.array(grad_histo[-1])
        x = np.array(x_histo[-1]) - epsDeltaFx

        # Ajout x, f(x) et dfonc(x)
        x_histo.append(x)
        f_histo.append(fonc(*x))
        grad_histo.append(dfonc(*x))

    return np.array(x_histo), np.array(f_histo), np.array(grad_histo)


def _test_1(x):
    return x * np.cos(x)


def _dtest_1(x):
    return np.cos(x) - (x * np.sin(x))


def _test_2(x):
    return -np.log(x) + x**2


def _dtest_2(x):
    return (2 * x) - (1 / x)


def _test_3(x1, x2):
    return 100 * ((x2 - (x1 ** 2)) ** 2) + ((1 - x1)**2)


def _dtest_3(x1, x2):
    return (-2 * (1 - x1) - 400 * x1 * (-x1**2 + x2)), (200 * (-x1**2 + x2))


def _plot_2D_val_grad_f(title, x_histo, f_histo, grad_histo):

    fig, ax = plt.subplots()

    ax.set_title(title)
    ax.plot(x_histo, f_histo, 'ro-', label="f(x)")
    ax.plot(x_histo, grad_histo, '*-', label=u"∇f(x)")
    ax.set_xlabel("x")
    ax.set_ylabel(u"f(x) et ∇f(x)")
    ax.legend(loc=2)

    plt.show()


def _plot_2D_compare(title, fonc=_test_1, dfonc=_dtest_1, xinit1=[2], xinit2=[2],
                     eps1=0.1, eps2=0.7):

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, figsize=(9,7))

    x_histo, f_histo, grad_histo = optimize(fonc, dfonc, xinit1, eps1)

    x = np.linspace(0, 2*np.pi, 30)

    plt.suptitle(title)

    ax1.plot(x, fonc(x), 'c-', label="fonction f")
    ax1.plot(x_histo, fonc(x_histo), c='orange', marker='.',
             label="trajectoire de\nl'optimisation, eps={}".format(eps1), linewidth=0.3)
    ax1.legend()

    x_histo, f_histo, grad_histo = optimize(fonc, dfonc, xinit2, eps2)

    ax2.plot(x, fonc(x), 'c-', label="fonction f")
    ax2.plot(x_histo[0], fonc(x_histo[0]), c='orange', marker='.',
             label="trajectoire de\nl'optimisation, eps={}".format(eps2), linewidth=0.3)

    # Uncomment the following code fot the live plot animation
    for i in range(0, x_histo.shape[0], 2):
        ax2.plot(x_histo[i:i+2], fonc(x_histo[i:i+2]), c='orange', marker='.', linewidth=0.3)
        plt.pause(0.01)
        pass

    ax2.legend()

    plt.show()


def _plot_courbe(title, x_histo, max_iter):

    xetoile = x_histo[-1]
    logDif = [np.log(np.abs(xt - xetoile)) for xt in x_histo[:-1]]

    fig, ax = plt.subplots()
    plt.suptitle(title)
    ax.plot(range(max_iter), logDif, 'b:')
    plt.show()


def _plot_3D(x_histo, f_histo, grad_histo, fonc):

    # Construction du référentiel 3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title("Rosenbrock")

    # Grille de discretisation
    grid, X, Y = make_grid(xmin=-1, xmax=1, ymin=-1, ymax=1)

    # Affichage 2D
    plt.contourf(X, Y, fonc(grid[:,0], grid[:,1]).reshape(X.shape))

    surf = ax.plot_surface(X, Y, fonc(X, Y), rstride=1, cstride=1,
                           cmap=cm.gist_rainbow, linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)

    # not working
    # ax.plot(x_histo[:, 0], x_histo[:, 1], f_histo.ravel(), color="black")
    plt.savefig("banana.png")


def main():

    #-------------------------- Fist 1D function

    max_iter = 100
    x_histo, f_histo, grad_histo = optimize(fonc=_test_1, dfonc=_dtest_1,
                                             xinit=[2], eps=0.1, max_iter=100)

    _plot_2D_val_grad_f("xcos(x)\nen fonction du nombre d'itération,\n"
                        "les valeurs de f et du gradient de f",
                        x_histo, f_histo, grad_histo)

    _plot_2D_compare(title="xcos(x)\nla fonction f et la trajectoire de\n"
                     "l'optimisation(les valeurs successives de f(x))",
                     fonc=_test_1, dfonc=_dtest_1, xinit1=[2], xinit2=[2],
                     eps1=0.1, eps2=0.8)

    _plot_courbe("xcos(x), courbe(t, log||xt - x*||)", x_histo, max_iter)


    #-------------------------- Second 1D function

    x_histo, f_histo, grad_histo = optimize(fonc=_test_2, dfonc=_dtest_2,
                                             xinit=[2], eps=0.1)

    _plot_2D_val_grad_f("-log(x)+x^2\nen fonction du nombre d'itération,\n"
                        "les valeurs de f et du gradient de f",
                        x_histo, f_histo, grad_histo)

    _plot_2D_compare(title="-log(x)+x^2\nla fonction f et la trajectoire de\n"
                     "l'optimisation(les valeurs successives de f(x))",
                     fonc=_test_2, dfonc=_dtest_2, xinit1=[2], xinit2=[2],
                     eps1=0.1, eps2=0.8)


    #-------------------------- 2d function Rosenbrock (or banana)

    x_histo, f_histo, grad_histo = optimize(fonc=_test_3, dfonc=_dtest_3,
                                             xinit=[0, 1], eps=0.1)

    _plot_2D_val_grad_f("Rosenbrock\nen fonction du nombre d'itération,\n"
                        "les valeurs de f et du gradient de f",
                        x_histo, f_histo, grad_histo)

    _plot_3D(x_histo, f_histo, grad_histo, _test_3)


    #--------------------------- Logistic Regression

    trainx, trainy = load_usps("USPS_train.txt")
    testx, testy = load_usps("USPS_test.txt")

    logisticReg = logisticRegression(loss_g=cost_f_g, max_iter=100,
                                     epsilon=0.1)
    #logisticReg.fit(trainx, trainy)

    # Matrice de poids 6 vs 9 and 1 vs 8
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True)
    plt.suptitle("Matrice de poids")
    weight_matrix(6, 9, fig, logisticReg, ax1)
    weight_matrix(1, 8, fig, logisticReg, ax2)
    plt.savefig("weight_matrix_qqlexs_LR")

    # Matrice de poids 6 vs All
    matrix_one_vs_all(6, logisticReg)

    # Courbes d'erreurs 6 vs All
    error_curves(6)

    print("Erreur : train %f, test %f\n"% (logisticReg.score(trainx,trainy),
                                           logisticReg.score(testx,testy)))

    #--------------------------- Naïve Bayes

    clf = GaussianNB()

    clf.fit(trainx, trainy)

    print("Erreur : train %f, test %f\n"% (1 - clf.score(trainx, trainy),
                                           1 - clf.score(testx,testy)))


if __name__ == '__main__':
    main()

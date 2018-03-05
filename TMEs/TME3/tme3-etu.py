# -*- coding: utf-8 -*-
""" ARF - TME3, Descente de gradient

Auteurs:
* BIZZOZZERO Nicolas
* BALDE Ahmed
"""

import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


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
    x_histo, f_histo, grad_histo = [xinit], [fonc(*xinit)], [[dfonc(*xinit)]]

    for _ in range(max_iter):
        # Calcul du gradient
        # x = x_histo[-1] - (eps * grad_histo[-1])
        x = (np.array(x_histo[-1]) - np.array([(np.array(x) * eps).tolist() for x in grad_histo[-1]])).tolist()
        # Mise à jour
        x_histo.append(x)
        f_histo.append(fonc(*x))
        grad_histo.append([dfonc(*x)])

    return np.array(x_histo), np.array(f_histo), np.array(grad_histo)


def _test_1(x):
    return x * np.cos(x)


def _dtest_1(x):
    return np.cos(x) - (x * np.sin(x))


def _test_2(x1, x2):
    return 100 * ((x2 - (x1 ** 2)) ** 2) + ((1 - x1)**2)


def _dtest_2(x1, x2):
    return (-2 * (1 - x1) - 400 * x1 * (-x1**2 + x2)), (200 * (-x1**2 + x2))


def _test_plot_2D(x_histo, f_histo, grad_histo, xlim, ylim):
    plt.plot(x_histo, f_histo, '+', label="f(x)")
    plt.plot(x_histo, grad_histo, '*', label=u"∇f(x)")
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xlabel("x")
    plt.ylabel(u"f(x) et ∇f(x)")
    plt.legend(loc=2)
    plt.show()


def _test_plot_3D(x_histo, f_histo, grad_histo):
    # Grille de discretisation
    grid, xx, yy = make_grid(xmin=-1, xmax=1, ymin=-1, ymax=1)

    # Affichage 2D
    plt.contourf(xx, yy, grid.reshape(xx.shape))
    fig = plt.figure()

    # Construction du référentiel 3D
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xx, yy, mafonction(grid).reshape(xx.shape),
                           rstride=1, cstride=1, cmap=cm.gist_rainbow,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf)
    ax.plot(x_histo[:, 0], x_histo[:, 1], f_histo.ravel(), color="black")
    plt.show()


def main():
    x_histo, f_histo, grad_histo = optimize(fonc=_test_1, dfonc=_dtest_1,
                                            xinit=[2], eps=0.1)
    # x_histo, f_histo, grad_histo = optimize(fonc=_test_2, dfonc=_dtest_2,
    #                                         xinit=[0, 0], eps=0.1)

    _test_plot_2D(x_histo, f_histo, grad_histo, (1, 4), (-4, 0.5))


if __name__ == '__main__':
    main()

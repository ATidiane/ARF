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
from math import inf


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


def optimize2(fonc, dfonc, xinit, eps, max_iter=100):
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
    
    x_histo, f_histo, grad_histo = optimize2(fonc, dfonc, xinit1, eps1)
    
    x = np.linspace(0, 2*np.pi, 30)

    plt.suptitle(title)
    
    ax1.plot(x, fonc(x), 'c-', label="fonction f")
    ax1.plot(x_histo, fonc(x_histo), c='orange', marker='.', label="trajectoire de\nl'optimisation, eps={}".format(eps1), linewidth=0.3)
    ax1.legend()

    x_histo, f_histo, grad_histo = optimize2(fonc, dfonc, xinit2, eps2)

    ax2.plot(x, fonc(x), 'c-', label="fonction f")
    ax2.plot(x_histo[0], fonc(x_histo[0]), c='orange', marker='.', label="trajectoire de\nl'optimisation, eps={}".format(eps2), linewidth=0.3)
    ax2.legend()
    for i in range(0, x_histo.shape[0], 2):
        ax2.plot(x_histo[i:i+2], fonc(x_histo[i:i+2]), c='orange', marker='.', linewidth=0.3)
        plt.pause(0.1)
    
    plt.show()



def _plot_courbe(title, x_histo, max_iter):

    xetoile = x_histo[-1]
    logDif = [np.log(np.abs(xt - xetoile)) for xt in x_histo[:-1]]

    fig, ax = plt.subplots()
    plt.suptitle(title)
    ax.plot(range(max_iter), logDif, 'b:')
    plt.show()
    


def _plot_3D(x_histo, f_histo, grad_histo, fonc):
    
    # Grille de discretisation
    grid, xx, yy = make_grid(xmin=-1, xmax=1, ymin=-1, ymax=1)

    # Bidouiller by me
    n = int(grid.shape[0]/2)
    grid = grid[0:n]

    # Affichage 2D
    plt.contourf(xx, yy, fonc(grid).reshape(xx.shape))
    fig = plt.figure()
    
    # Construction du référentiel 3D
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xx, yy, fonc(grid).reshape(xx.shape),
                           rstride=1, cstride=1, cmap=cm.gist_rainbow,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf)
    ax.plot(x_histo[:, 0], x_histo[:, 1], f_histo.ravel(), color="black")
    plt.show()

    
def main():

    #-------------------------- Fist 1D function

    max_iter = 100
    x_histo, f_histo, grad_histo = optimize2(fonc=_test_1, dfonc=_dtest_1,
                                             xinit=[2], eps=0.1, max_iter=100)

    _plot_2D_val_grad_f("xcos(x)\nen fonction du nombre d'itération,\n"
                        "les valeurs de f et du gradient de f",
                        x_histo, f_histo, grad_histo)
    
    _plot_2D_compare(title="xcos(x)\nla fonction f et la trajectoire de\n"
                     "l'optimisation(les valeurs successives de f(x))",
                     fonc=_test_1, dfonc=_dtest_1, xinit1=[2], xinit2=[2],
                     eps1=0.1, eps2=0.8)

    _plot_courbe("xcos(x), courbe(t, log||xt - x*||)", x_histo, max_iter)
    
    #_plot_3D(x_histo, f_histo, grad_histo, _test_1)

    #-------------------------- Second 1D function

    x_histo, f_histo, grad_histo = optimize2(fonc=_test_2, dfonc=_dtest_2,
                                             xinit=[2], eps=0.1)

    _plot_2D_val_grad_f("-log(x)+x^2\nen fonction du nombre d'itération,\n"
                        "les valeurs de f et du gradient de f",
                        x_histo, f_histo, grad_histo)
    
    _plot_2D_compare(title="-log(x)+x^2\nla fonction f et la trajectoire de\n"
                     "l'optimisation(les valeurs successives de f(x))",
                     fonc=_test_2, dfonc=_dtest_2, xinit1=[2], xinit2=[2],
                     eps1=0.1, eps2=0.8)

    #_plot_3D(x_histo, f_histo, grad_histo, _test_2)
    
    #-------------------------- 2d function Rosenbrock (or banana)
        
    # x_histo, f_histo, grad_histo = optimize2(fonc=_test_3, dfonc=_dtest_3,
    #                                          xinit=[2], eps=0.1)

    # _plot_2D_val_grad_f("Rosenbrock\nen fonction du nombre d'itération,\n"
    #                     "les valeurs de f et du gradient de f",
    #                     x_histo, f_histo, grad_histo)
    
    # _plot_2D_compare(title="Rosenbrock\nla fonction f et la trajectoire de\n"
    #                  "l'optimisation(les valeurs successives de f(x))",
    #                  fonc=_test_3, dfonc=_dtest_3, xinit1=[2], xinit2=[2, 0],
    #                  eps1=0.1, eps2=0.7)

    #_plot_3D(x_histo, f_histo, grad_histo, _test_3)
    

    
if __name__ == '__main__':
    main()

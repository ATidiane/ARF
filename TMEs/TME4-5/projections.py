# -*- coding: utf-8 -*-

##########################################################################
#-------------------------------- Kernel Trick --------------------------------#
##########################################################################

import numpy as np
from math import *
from sklearn.metrics.pairwise import rbf_kernel


def allow_bias(datax):
    """ Ajoute une colonne de remplie de 1 à datax et retourne la nouvelle
        matrice correspondante.
    """

    adjust = np.ones(datax.shape[0]).reshape(-1, 1)
    return np.hstack((datax, adjust))


def kernel_poly(datax):
    """ Projette les données dans un espace polynomial, 6 dimensions """

    x1carre = (datax[:, 0] * datax[:, 0]).reshape(-1, 1)
    x2carre = (datax[:, 1] * datax[:, 1]).reshape(-1, 1)
    x1x2 = (datax[:, 0] * datax[:, 1]).reshape(-1, 1)

    return np.hstack((datax, x1carre, x2carre, x1x2))


def kernel_gaussian(datax, trainx, sigma=1):
    """ Projection gaussienne de toutes les données """

    rowd, col = datax.shape
    rowt, col = trainx.shape
    gassMatrix = np.zeros((rowd, rowt))

    for i, x in enumerate(datax):
        for j, z in enumerate(trainx):
            gassMatrix[i, j] = gaussian(x.T, z.T, sigma)

    return gassMatrix


def kernel_gauss(datax, datay=None, gamma=None):
    """ Calculate gaussian kernel using sklearn

    :param datax:
    :param datay:
    :param gamma:
    :returns:
    :rtype:

    """

    return rbf_kernel(datax, datay, gamma)


def gaussian(x, z, sigma):
    """ Calcule la projection gaussienne pour 2 arrays """
    return np.exp(-np.linalg.norm(x - z, 2)**2 / (2. * sigma**2))

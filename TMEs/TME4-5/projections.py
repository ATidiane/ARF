# -*- coding: utf-8 -*-

################################################################################
#-------------------------------- Kernel Trick --------------------------------#
################################################################################

import numpy as np
from math import *
def allow_bias(datax):
    """ Ajoute une colonne de remplie de 1 à datax et retourne la nouvelle
        matrice correspondante.
    """

    adjust = np.ones(datax.shape[0]).reshape(-1, 1)
    return np.hstack((datax, adjust))


def kernel_poly(datax):
    """ Projette les données dans un espace polynomial, 6 dimensions """

    x1carre = (datax[:,0]*datax[:,0]).reshape(-1, 1)
    x2carre = (datax[:,1]*datax[:,1]).reshape(-1, 1)
    x1x2 = (datax[:,0]*datax[:,1]).reshape(-1, 1)

    return np.hstack((datax, x1carre, x2carre, x1x2))


def kernel_gaussian(datax, sigma=1):
    """ Projection gaussienne de toutes les données """

    row, col = datax.shape
    gassMatrix = np.zeros((row, row))

    for i, x in enumerate(datax):
        for j, z in enumerate(datax):
            gassMatrix[i,j] = gaussian(x.T, z.T, sigma)

    return gassMatrix


def gaussian(x,z,sigma):
    """ Calcule la projection gaussienne pour 2 arrays """
    return np.exp(-np.linalg.norm(x - z, 2)**2/(2.*sigma**2))

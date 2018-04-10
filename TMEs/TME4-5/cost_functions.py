# -*- coding: utf-8 -*-

import numpy as np


################################################################################
#-------------------------------Cost functions --------------------------------#
################################################################################


def decorator_vec(fonc):
    def vecfonc(datax,datay,w,*args,**kwargs):
        if not hasattr(datay,"__len__"):
            datay = np.array([datay])
        datax,datay,w =  datax.reshape(len(datay),-1),datay.reshape(-1,1),w.reshape((1,-1))
        return fonc(datax,datay,w,*args,**kwargs)
    return vecfonc


@decorator_vec
def mse(datax, datay, w):
    """ retourne la moyenne de l'erreur aux moindres carres """
    # (w.x - y)² np.mean(((datax * w) - datay)**2)
    return np.mean((np.dot(datax, w.T) - datay)**2)


@decorator_vec
def mse_g(datax, datay, w):
    """ retourne le gradient moyen de l'erreur au moindres carres """
    # 2 x (w.x - y)²
    return np.mean(2 * ((np.dot(datax, w.T)) - datay)**2)


@decorator_vec
def hinge(datax, datay, w):
    """ retourn la moyenne de l'erreur hinge """
    # Erreur moyenne de -f(x)y
    moinsyfx = -datay * np.dot(datax, w.T)
    return np.maximum(0, moinsyfx).mean()


# @decorator_vec
# def hinge_g(datax, datay, w):
#     """ retourne le gradient moyen de l'erreur hinge """
#     h = np.array((datay * np.dot(datax, w.T)) < 0)
#     return np.mean(h * datax * datay)

@decorator_vec
def hinge_g(datax, datay, w, activation=np.sign):
    """ Retourne le gradient de l'erreur hinge """
    cost = -activation(hinge(datax, datay, w)) * datax * datay
    return (np.sum(cost, axis=0) / len(datax))  # Normalisation



def stochastic(vectorx, vectory, w):
    """ Retourne l'erreur aux moindres carres pour UN exemple de data,
        cette pratique de calculer la descente de gradient juste pour un ex
        à chaque itération est appelée descente de gradient stochastique ou
        Stochastic gradient descent.
    """
    return ((np.dot(vectorx.reshape(1,-1), w.T) - vectory)**2)/2


def stochastic_g(vectorx, vectory, w):
    """ Retourne le gradient de l'erreur aux moindres carres pour UN exemple de
        datax.
    """
    return (np.dot(vectorx.reshape(1,-1), w.T) - vectory)

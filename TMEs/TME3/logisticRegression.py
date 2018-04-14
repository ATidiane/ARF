# -*- coding: utf-8 -*-

import numpy as np


def sigmoid(z):
    """ Retourne la fonction sigmoide """

    return 1 / (1 + np.exp(-z))


def cost_f_g(datax, datay, w):
    """ Calcule le gradient du coût pour la regression logistique """

    z = w.dot(datax.T)
    sig = sigmoid(z)
    return (sig - datay).dot(datax)


class logisticRegression:
    """ Implémentation de la logistique regression
    """

    def __init__(self, loss_g=cost_f_g, max_iter=1000, epsilon=0.1):
        """ Initialiser les paramètres de la logistique regression
            :param loss_g: gradient de la fonction de coût
            :param max_iter: nombre maximum d'itération
            :param epsilon: pas d'apprentissage
        """

        self.loss_g = loss_g
        self.max_iter = max_iter
        self.epsilon = epsilon

    def fit(self, datax, datay):
        """ Apprentissage de la logistique regression sur les données de train,
            Optimisation du paramètre w par la descente de gradient.
            :param datax: Exemples du datasetExemples du dataset
            :param datay: Labels du dataset
        """

        self.w = np.random.random((1, datax.shape[1]))
        for _ in range(self.max_iter):
            self.w -= self.epsilon * cost_f_g(datax, datay, self.w)

    def predict(self, datax):
        """ Prediction des données

        :param datax: Exemples du dataset

        """

        D = datax.shape[0]
        fx = np.zeros((D, 1))

        for i, x in enumerate(datax):
            z = self.w.dot(x.T)
            fx[i] = 1 if sigmoid(z) >= 0.5 else -1
        return fx

    def score(self, datax, datay):
        """ Evaluation de la logistic Regression

        :param datax: Exemples du dataset
        :param datay: Labels du dataset

        """

        return (1 - np.mean((self.predict(datax) == datay[:, np.newaxis])))

# -*- coding: utf-8 -*-

from arftools import *
from cost_functions import *
from projections import *
import numpy as np

##########################################################################
# --------------------------------- Perceptron ------------------------- #
##########################################################################


class Perceptron(object):
    def __init__(self, loss=mse, loss_g=mse_g, max_iter=1000,
                 eps=0.01, kernel="bias"):
        """ Initialisation des paramètres du perceptron

        :param loss: fonction de coût
        :param loss_g: gradient de la fonction coût
        :param max_iter: nombre maximum d'itération de la fonction coût
        :param eps: pas du gradient
        :param kernel: kernel utilisé

        """

        self.max_iter, self.eps = max_iter, eps
        self.loss, self.loss_g = loss, loss_g
        self.kernel = kernel

    def projection_decorator_fit(fonc):
        def adjust_datax(self, datax, *args, **kwargs):
            """ Decorator for fit

            :param datax: contient tous les exemples du dataset
            :returns: transformed dataset
            :rtype: numpy array

            """

            self.trainx = datax
            if self.kernel == "bias":
                datax = allow_bias(datax)
            elif self.kernel == "polynomial":
                datax = kernel_poly(datax)
            elif self.kernel == "gaussian":
                #datax = kernel_gaussian(datax, datax, 1.5)
                datax = kernel_gauss(datax, datax, 1.5)

            return fonc(self, datax, *args, **kwargs)

        return adjust_datax

    def projection_decorator_predict(fonc):
        def adjust_datax(self, datax, *args, **kwargs):
            """ Decorator for predict

            :param datax: contient tous les exemples du dataset
            :returns: transformed dataset
            :rtype: numpy array

            """

            if self.kernel == "bias":
                datax = allow_bias(datax)
            elif self.kernel == "polynomial":
                datax = kernel_poly(datax)
            elif self.kernel == "gaussian":
                datax = kernel_gauss(datax, self.trainx, 1.5)

            return fonc(self, datax, *args, **kwargs)

        return adjust_datax

    def batch_fit(self, datax, datay):
        """ Classic gradient descent Learning

        :param datax: contient tous les exemples du dataset
        :param datay: labels du dataset

        """

        print("=================Batch=================\n")
        for i in range(self.max_iter):
            self.w = self.w - (self.eps * self.loss_g(datax, datay, self.w))

    def stochastic_fit(self, datax, datay):
        """ Stochastic gradient descent Learning

        :param datax: contient tous les exemples du dataset
        :param datay: labels du dataset

        """

        print("===============Stochastic==============\n")
        self.loss_g = stochastic_g
        data = np.hstack((datax, datay))
        # It's a good thing to shuffle the datax.
        np.random.shuffle(data)
        datax, datay = data[:, :2], data[:, -1]

        # plt.figure()
        grid, x, y = make_grid(datax, 200)
        step = 0
        for _ in range(self.max_iter):
            for vectorx, vectory in zip(datax, datay):
                self.w -= (self.eps * self.loss_g(vectorx, vectory, self.w))

    def stochastic_fit_animation(self, datax, datay):
        """ Stochastic gradient descent Learning

        :param datax: contient tous les exemples du dataset
        :param datay: labels du dataset

        """

        print("===============Stochastic==============\n")
        self.loss_g = stochastic_g
        data = np.hstack((datax, datay))
        # It's a good thing to shuffle the datax.
        np.random.shuffle(data)
        datax, datay = data[:, :2], data[:, -1]

        # plt.figure()
        grid, x, y = make_grid(datax, 200)
        step = 0
        for _ in range(self.max_iter):
            for vectorx, vectory in zip(datax, datay):
                self.w -= (self.eps * self.loss_g(vectorx, vectory, self.w))

                # Show video of learning
                plt.title("Stochastic animation, step %d" % (step))
                plt.contourf(x, y, self.predict(grid).reshape(x.shape),
                             colors=('gray', 'blue'), levels=[-1, 0, 1])
                plot_data(datax, datay)
                plt.pause(0.0000000000000001)
                step += 1
                plt.close()

    def minibatch_fit(self, datax, datay, batch_size=10):
        """ Mini-Batch gradient descent Learning

        :param datax: contient tous les exemples du dataset
        :param datay: labels du dataset
        :param batch_size: nb d'exemples sur lesquels itérés en un.

        """

        print("===============Mini-Batch==============\n")
        for _ in range(self.max_iter):
            for i in range(0, datax.shape[0], batch_size):
                # On prend seulement batch_size données sur toutes les données.
                batchx, batchy = datax[i:i +
                                       batch_size], datay[i:i + batch_size]
                # Et on essaye de progresser avec cela.
                self.w -= (self.eps * self.loss_g(batchx, batchy, self.w))

    @projection_decorator_fit
    def fit(self, datax, datay, testx=None, testy=None,
            gradient_descent="batch", batch_size=10):
        """ Apprentissage

        :param datax: contient tous les exemples du dataset
        :param datay: labels du dataset
        :param testx: contient les exemples de la base de test
        :param testy: contient les labels de la base de test
        :param gradient_descent: type de descente de gradient utilisé
        :param batch_size: nb d'exemples sur lesquels itérés en un.

        """

        self.datax = datax
        N = len(datay)
        datay = datay.reshape(-1, 1)

        if len(datax.shape) == 1:
            D = 1
        else:
            D = datax.shape[1]

        self.w = np.random.random((1, D))

        # Let's learn !
        if gradient_descent == "batch":
            self.batch_fit(datax, datay)
        elif gradient_descent == "stochastic":
            self.stochastic_fit(datax, datay)
        elif gradient_descent == "stochastic_animation":
            self.stochastic_fit_animation(datax, datay)
        else:
            self.minibatch_fit(datax, datay, batch_size)

    @projection_decorator_predict
    def predict(self, datax):
        """ Predict labels

        :param datax: contient tous les exemples du dataset
        :returns: predicted labels
        :rtype: numpy array

        """
        if len(datax.shape) == 1:
            datax = datax.reshape(-1, 1)

        return np.sign(datax.dot(self.w.T))

    def score(self, datax, datay):
        """ Evaluate de classification

        :param datax: contient les exemples du dataset
        :param datay: labels du dataset
        :returns: score des erreurs
        :rtype: float

        """

        return (1 - np.mean((self.predict(datax) == datay[:, np.newaxis])))

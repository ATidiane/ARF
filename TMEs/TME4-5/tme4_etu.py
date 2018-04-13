import matplotlib.pyplot as plt
import numpy as np

from arftools import *
from cost_functions import *
from perceptron import *
from projections import *
from usps import *

##########################################################################
# ---------------------- Plotting errors and 2D perceptron ------------- #
##########################################################################


def plot_error(datax, datay, f, step=10):
    """ Tracer des isocourbes de l'erreur

    :param datax: Contient les différents exemples des données
    :param datay: Labels des exemples
    :param f:
    :param step:

    """

    grid, x1list, x2list = make_grid(
        data=datax, xmin=-4, xmax=4, ymin=-4, ymax=4)
    plt.contourf(x1list, x2list, np.array(
        [f(datax, datay, w) for w in grid]).reshape(x1list.shape), 25)
    plt.colorbar()
    plt.show()


def learn_plot_perceptron2D(perceptron, trainx, trainy, testx, testy,
                            gradient_descent, title="Batch gradient descent"):
    """FIXME! briefly describe function

    :param perceptron: Perceptron utilisé pour afficher les données
    :param trainx: Données d'apprentissage
    :param trainy: Labels des données d'apprentissage
    :param testx: Données de test
    :param testy: Labels des données de test
    :param gradient_descent: type de gradient descent utilisé
    :param title: titre du graphique
    :returns: affiche un graphique
    :rtype: void

    """

    perceptron.fit(trainx, trainy, gradient_descent=gradient_descent)
    print("Erreur : train %f, test %f\n" % (perceptron.score(trainx, trainy),
                                            perceptron.score(testx, testy)))
    plot_frontiere(testx, perceptron.predict, 20)
    plot_data(testx, testy, title=title)
#    plt.show()
    plt.savefig("batch03m.png")


def plot_cost_erreur(datax, datay):
    """ Plots the error for the two cost functions in the same time given data

    :param datax: Contient les différents exemples des données
    :param datay: Labels des exemples

    """
        func_cost = [mse, hinge]
    for i, cost in enumerate(func_cost):
        plt.figure()
        plot_error(datax, datay, cost)
        # plt.savefig("error0_{}".format(i))


##########################################################################
#------------------------------------- Main -----------------------------#
##########################################################################

def main():
    """ Tracer des isocourbes de l'erreur """

    # plt.ion
    trainx, trainy = gen_arti(nbex=1000, data_type=2, epsilon=0)
    testx, testy = gen_arti(nbex=1000, data_type=2, epsilon=0.001)
    grid, x1list, x2list = make_grid(xmin=-4, xmax=4, ymin=-4, ymax=4)

    # Plot error for test
    # plot_cost_erreur(testx, testy)

    # Batch gradient descent
    perceptron = Perceptron(loss=mse, loss_g=mse_g, max_iter=100, eps=0.1,
                            kernel=None)
    learn_plot_perceptron2D(perceptron, trainx, trainy, testx, testy,
                            gradient_descent="batch",
                            title="Batch gradient descent")

    # perceptron_poly = Perceptron(loss=hinge,loss_g=hinge_g,max_iter=100,eps=0.1,
    #                         kernel="polynomial")
    # learn_plot_perceptron2D(perceptron_poly, trainx, trainy, testx, testy,
    #                       gradient_descent = "batch",
    #                       title = "Batch gradient descent")

    # perceptron_gaussian = Perceptron(loss=hinge,loss_g=hinge_g,max_iter=100,eps=0.1,
    #                         kernel="gaussian")
    # learn_plot_perceptron2D(perceptron_gaussian, trainx, trainy, testx, testy,
    #                       gradient_descent = "batch",
    #                       title = "Batch gradient descent")

    # # # Stochastic gradient descent
    # perceptron_stochastic = Perceptron(loss=stochastic,loss_g=stochastic_g,
    #                                    max_iter=50, eps=0.1, kernel=None)
    # learn_plot_perceptron2D(perceptron_stochastic, trainx, trainy, testx, testy,
    #                       gradient_descent = "stochastic",
    #                       title = "Stochastic gradient descent")

    # # # Mini-Batch gradient descent
    # perceptron_minibash = Perceptron(loss=hinge,loss_g=hinge_g,max_iter=100,
    #                          eps=0.1, kernel=None)
    # learn_plot_perceptron2D(perceptron_minibash, trainx, trainy, testx, testy,
    #                       gradient_descent = "minibatch",
    #                       title = "Mini-Batch gradient descent")

    # Stochastic gradient descent Animation
    perceptron_stoch_anim = Perceptron(loss=stochastic, loss_g=stochastic_g,
                                       max_iter=1, eps=0.1, kernel=None)
    # learn_plot_perceptron2D(perceptron_stoch_anim, trainx, trainy, testx, testy,
    #                      gradient_descent = "stochastic_animation",
    #                      title = "Stochastic gradient descent")

    ##########################################################################
    #------------------------------------ USPS ------------------------------#
    ##########################################################################

    perceptron_usps = Perceptron(loss=hinge, loss_g=hinge_g, max_iter=500,
                                 eps=0.1, kernel=None)

    # Matrice de poids 6 vs 9 and 1 vs 8
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True)
    plt.suptitle("Matrice de poids")
    weight_matrix(6, 9, fig, perceptron_usps, ax1)
    weight_matrix(1, 8, fig, perceptron_usps, ax2)
    plt.savefig("weight_matrix_qqlexs")

    # Matrice de poids 6 vs All
    matrix_one_vs_all(6, perceptron_usps)

    # Courbes d'erreurs 6 vs All
    error_curves(6)


if __name__ == "__main__":

    main()

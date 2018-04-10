from arftools import *
from cost_functions import *
from projections import *
from perceptron import *
from usps import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import time



################################################################################
#---------------------- Plotting errors and 2D perceptron ---------------------#
################################################################################


def plot_error(datax,datay,f,step=10):
    """ Tracer des isocourbes de l'erreur """

    grid,x1list,x2list=make_grid(data=datax, xmin=-4,xmax=4,ymin=-4,ymax=4)
    plt.contourf(x1list,x2list,np.array([f(datax,datay,w) for w in grid]).reshape(x1list.shape),25)
    plt.colorbar()
    # plt.show()


def learn_plot_perceptron2D(perceptron, trainx, trainy, testx, testy, gradient_descent,
                          title="Batch gradient descent"):

    perceptron.fit(trainx,trainy, gradient_descent=gradient_descent)
    print("Erreur : train %f, test %f\n"% (perceptron.score(trainx,trainy),
                                           perceptron.score(testx,testy)))
    plt.figure()
    plot_frontiere(testx,perceptron.predict,200)
    plot_data(testx,testy, title=title)
    plt.show()
    plt.savefig("gradent_descent.png")


def plot_cost_erreur(datax, datay):
    """ Plots the error for the two cost functions in the same time given data """

    func_cost = [mse, hinge]
    for i, cost in enumerate(func_cost):
        plt.figure()
        plot_error(datax, datay, cost)
        # plt.savefig("error0_{}".format(i))



################################################################################
#------------------------------------ Main ------------------------------------#
################################################################################

def main():
    """ Tracer des isocourbes de l'erreur """

    # plt.ion()
    trainx,trainy =  gen_arti(nbex=1000,data_type=1 ,epsilon=0.3)
    testx,testy =  gen_arti(nbex=1000,data_type=1,epsilon=0.3)
    grid,x1list,x2list=make_grid(xmin=-4,xmax=4,ymin=-4,ymax=4)

    # Plot error for test
    # plot_cost_erreur(testx, testy)

    # Batch gradient descent
    perceptron = Perceptron(loss=hinge,loss_g=hinge_g,max_iter=10000,eps=0.1,
                            kernel="gaussian")
    learn_plot_perceptron2D(perceptron, trainx, trainy, testx, testy,
                          gradient_descent = "batch",
                          title = "Batch gradient descent")


    # # Stochastic gradient descent
    # perceptron2 = Perceptron(loss=stochastic,loss_g=stochastic_g,max_iter=10,
    #                        eps=0.1,bias=False)
    # learn_plot_perceptron2D(perceptron2, trainx, trainy, testx, testy,
    #                       gradient_descent = "stochastic",
    #                       title = "Stochastic gradient descent")


    # # Mini-Batch gradient descent
    # perceptron3 = Perceptron(loss=hinge,loss_g=hinge_g,max_iter=1000,
    #                        eps=0.1,bias=False)
    # learn_plot_perceptron2D(perceptron3, trainx, trainy, testx, testy,
    #                       gradient_descent = "minibatch",
    #                       title = "Mini-Batch gradient descent")


    # # Stochastic gradient descent Animation
    # perceptron2 = Perceptron(loss=stochastic,loss_g=stochastic_g,max_iter=1,
    #                        eps=0.1,bias=False)
    # # learn_plot_perceptron2D(perceptron2, trainx, trainy, testx, testy,
    # #                      gradient_descent = "stochastic_animation",
    # #                      title = "Stochastic gradient descent")


    #############################################################################
    #------------------------------------ USPS ---------------------------------#
    #############################################################################

    # Matrice de poids 6 vs 9 and 1 vs 8
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True)
    plt.suptitle("Matrice de poids")
    weight_matrix(6, 9, fig, ax1)
    weight_matrix(1, 8, fig, ax2)
    plt.savefig("weight_matrix_qqlexs")

    # Matrice de poids 6 vs All
    matrix_one_vs_all(6)

    # Courbes d'erreurs 6 vs All
    error_curves(6)


if __name__=="__main__":

    main()

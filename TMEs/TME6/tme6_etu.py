# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '../TME4-5')

import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection, multiclass, svm
from sklearn.linear_model import Perceptron
from collections import Counter

from arftools import make_grid, gen_arti
from usps import weight_matrix, error_curves, matrix_one_vs_all, load_usps


def plotAll_in_subplots(testx, testy, f, ax=plt):
    """ Plot la frontière ainsi que les données sur l'axe ax

    :param testx: Contient les exemples de la base de test
    :param testy: Labels de la base de test
    :param f: fonction de prédiction
    :param ax: axe sur lequel affiché le graphique.

    """

    # Plot frontière
    grid, x, y = make_grid(data=testx, step=50)
    ax.contourf(x, y, f(grid).reshape(x.shape), 255)

    # Plot data
    cols = ["red", "green", "blue", "orange", "black", "cyan"]
    marks = [".", "+", "*", "o", "x", "^"]
    for i, l in enumerate(sorted(list(set(testy.flatten())))):
        ax.scatter(testx[testy == l, 0], testx[testy == l, 1],
                   c=cols[i], marker=marks[i])


def linear_model(perceptron, data_type=0, epsilon=0.3, ax=plt):
    """ Utilise le perceptron passé en paramètre en l'occurence celui de
        sklearn.
    :param perceptron: Perceptron initialisé, soit pour les données
                       artificiels, soit, pour les données USPS.
    :param data_type: Différents types de données: 0, 1, 2
    :param epsilon: Bruit dans les données
    :param ax: axe sur lequel affiché le graphique.

    """

    trainx, trainy = gen_arti(
        nbex=1000, data_type=data_type, epsilon=epsilon)
    testx, testy = gen_arti(nbex=1000, data_type=data_type, epsilon=epsilon)

    # Learning
    perceptron.fit(trainx, trainy)

    # erreurs
    err_train = round(1 - perceptron.score(trainx, trainy), 3)
    err_test = round(1 - perceptron.score(testx, testy), 3)

    plotAll_in_subplots(testx, testy, perceptron.predict, ax)

    ax.set_title("linear_model_perceptron \non data_type {}".format(data_type))


def comparaison_usps(perceptron):
    """ Genere les courbes d'erreurs en apprentissage et en test pour la classe

    :param perceptron: Perceptron initialisé, soit pour les données
                       artificiels, soit pour les données USPS.

    """
    # Matrice de poids 6 vs 9 and 1 vs 8
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True)
    plt.suptitle("Matrice de poids")
    weight_matrix(6, 9, fig, perceptron, ax1)
    weight_matrix(1, 8, fig, perceptron, ax2)
    # plt.savefig("weight_matrix_qqlexs_tme6")

    # Matrice de poids 6 vs All
    matrix_one_vs_all(6, perceptron)

    # Courbes d'erreurs 6 vs All
    error_curves(6)


##########################################################################
# ------------------------------ SVM et GridSearch --------------------- #
##########################################################################

def plot_frontiere_proba(data, f, step=20):
    """FIXME! briefly describe function

    :param data: Données pour lesquels affichées la frontière.
    :param f: fonction de prédiction
    :param step:
    :returns:
    :rtype:

    """

    grid, x, y = make_grid(data=data, step=step)
    plt.contourf(x, y, f(grid).reshape(x.shape), 255)


def SVM(data_type=0, epsilon=0.3, C=10, kernel='linear',
        probability=True, max_iter=100, ax=plt):
    """ Plot ls différents kernels appliqués sur les différents types de données

    :param data_type: Différents types de données: 0, 1, 2
    :param epsilon: Bruit dans les données
    :param C: pénalité
    :param kernel: kernel utilisé
    :param probability: Booléen permettant l'utilisation de la frontière ou nn.
    :param max_iter: Maximum d'itérations
    :param ax: axe sur lequel affiché le graphique.

    """

    trainx, trainy = gen_arti(nbex=1000, data_type=data_type, epsilon=epsilon)
    testx, testy = gen_arti(nbex=1000, data_type=data_type, epsilon=epsilon)

    s = svm.SVC(C=C, kernel=kernel, probability=probability, max_iter=max_iter)

    s.fit(trainx, trainy)

    err_train = round(1 - s.score(trainx, trainy), 3)
    err_test = round(1 - s.score(testx, testy), 3)

    print("Erreur : train %f, test %f\n" % (err_train, err_test))

    ax.set_title("SVM {} kernel \non data_type {}".format(kernel, data_type))

    if probability:
        def f(x): return s.predict_proba(x)[:, 0]
    else:
        def f(x): return s.decision_function(x)

    plotAll_in_subplots(testx, testy, f, ax)

    ax.legend(["err_train: {}\nerr_test: {}".format(err_train, err_test)],
              loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2),
              ncol=1, fancybox=True, shadow=True)


def svm_gridSearch(trainx, trainy, testx, testy):
    """ Recherche des meilleurs paramètres à appliquer sur ces données

    :param trainx: Contient les exemples de la base d'apprentissage
    :param trainy: Labels de la base d'apprentissage
    :param testx: Contient les exemples de la base de test
    :param testy: Labels de la base de test

    """

    parameters = {'C': [1, 5, 10, 20, 40, 80, 150, 250, 500],
                  'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                  'gamma': [0.0001, 0.001, 0.01, 0.1],
                  'max_iter': [1, 10, 100, 500]}

    svc = svm.SVC()
    clf = model_selection.GridSearchCV(svc, parameters, n_jobs=-1)
    clf.fit(trainx, trainy)
    clf.predict(testx)

    print("Best parameters set found on development set:")
    print(clf.best_params_)

    # print("Grid scores on development set:")

    # For more details, uncomment !
    # for params, mean_score, scores in clf.grid_scores_:
    #     print("%0.3f (+/-%0.03f) for %r"
    #           % (mean_score, scores.std() * 2, params))


##########################################################################
# -------------------------- Apprentissage multi-classe ---------------- #
##########################################################################

def multiClass(trainx, trainy, testx, testy):
    """ Traitement des cas multiclasses à partir de classifieurs binaires

    :param trainx: Contient les exemples de la base d'apprentissage
    :param trainy: Labels de la base d'apprentissage
    :param testx: Contient les exemples de la base de test
    :param testy: Labels de la base de test

    """

    Unvs1 = multiclass.OneVsOneClassifier(svm.LinearSVC(random_state=0))
    UnvsAll = multiclass.OneVsRestClassifier(svm.LinearSVC(random_state=0))

    Unvs1.fit(trainx, trainy)
    err_Unvs1_train = round(1 - Unvs1.score(trainx, trainy), 3)
    err_Unvs1_test = round(1 - Unvs1.score(testx, testy), 3)

    UnvsAll.fit(trainx, trainy)
    err_UnvsAll_train = round(1 - UnvsAll.score(trainx, trainy), 3)
    err_UnvsAll_test = round(1 - UnvsAll.score(testx, testy), 3)

    print("Err_1vs1 : train %f, test %f\n" % (err_Unvs1_train, err_Unvs1_test))
    print("Err_1vsAll : train %f, test %f\n" % (err_UnvsAll_train,
                                                err_UnvsAll_test))

    print("========== Prediction UnvsUn ==============")
    print(Unvs1.predict(testx).shape)

    print("========= Prediction UnvsRest =============")
    print(UnvsAll.predict(testx).shape)


##########################################################################
# --------------------------------- String Kernel ---------------------- #
##########################################################################


def string_kernel(s1, s2):
    w1, w2 = s1.split(), s2.split()
    c1, c2 = Counter(w1), Counter(w2)
    num = sum([c1[k] * c2[k] for k in set(w1).intersection(w2)])
    den = (len(w1) + len(w2))
    return num / den


##########################################################################
# ------------------------------------- Main --------------------------- #
##########################################################################

def main():
    """ Fonction main """

    # Creating the model
    perceptron = Perceptron(max_iter=500, n_jobs=-1)

    # Perceptron du linear model sur les différents types de données
    fig_lm, ax_lm = plt.subplots(ncols=3, figsize=(10, 6))
    for dtype in range(3):
        linear_model(perceptron, data_type=dtype, epsilon=0.3, ax=ax_lm[dtype])

    # plt.savefig("linear_model_perceptron.png")

    # Application du perceptron de linear model sur les données usps
    comparaison_usps(perceptron)

    # Application des différents kernels sur l'ensemble des types de données
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    fig_svm, ax_svm = plt.subplots(
        ncols=4, nrows=3, sharex=True, figsize=(25, 15))
    for dtype in range(3):
        for j, kernel in enumerate(kernels):
            SVM(data_type=dtype, epsilon=0, C=1, kernel=kernel,
                probability=False, max_iter=-1, ax=ax_svm[dtype][j])

    fig_svm.tight_layout()
    # plt.savefig("svm_data_kernel_rapport.png")
    plt.show()

    # ------------------------------- Grid Search

    # Données 2D artificiels
    trainx, trainy = gen_arti(nbex=1000, data_type=0, epsilon=0.3)
    testx, testy = gen_arti(nbex=1000, data_type=0, epsilon=0.3)
    # svm_gridSearch(trainx, trainy, testx, testy)

    # Données Usps
    trainux, trainuy = load_usps("USPS_train.txt")
    testux, testuy = load_usps("USPS_test.txt")
    # svm_gridSearch(trainux, trainuy, testux, testuy)

    # ------------------------------- Multi Class

    print(np.unique(testuy))
    multiClass(trainux, trainuy, testux, testuy)

    # ------------------------------- String Kernel


if __name__ == "__main__":
    main()

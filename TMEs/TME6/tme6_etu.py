# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '../TME4-5')

from arftools import *
from usps import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn import svm, model_selection, multiclass


def plotAll_in_subplots(testx, testy, f, ax=plt):
    """ Plot la frontière ainsi que les données sur l'axe ax """

    # Plot frontière
    grid,x,y=make_grid(data=testx,step=50)
    ax.contourf(x, y, f(grid).reshape(x.shape), 255)

    # Plot data
    cols = ["red", "green", "blue", "orange", "black", "cyan"]
    marks = [".","+","*","o","x","^"]
    for i,l in enumerate(sorted(list(set(testy.flatten())))):
        ax.scatter(testx[testy==l,0],testx[testy==l,1],c=cols[i],marker=marks[i])


def linear_model(perceptron, data_type=0, epsilon=0.3, ax=plt):
    """ Utilise le perceptron passé en paramètre en l'occurence celui de
        linear_model et applique sur les data_type 0..2
    """

    trainx,trainy =  gen_arti(nbex=1000,data_type=data_type,epsilon=epsilon)
    testx,testy =  gen_arti(nbex=1000,data_type=data_type,epsilon=epsilon)

    # Learning
    perceptron.fit(trainx, trainy)

    # erreurs
    err_train = round(1 - perceptron.score(trainx, trainy), 3)
    err_test = round(1 - perceptron.score(testx, testy), 3)

    plotAll_in_subplots(testx, testy, perceptron.predict, ax)

    ax.set_title("linear_model_perceptron \non data_type {}".format(data_type))


def comparaison_usps(perceptron):
    """ Genere les courbes d'erreurs en apprentissage et en test pour la classe
        6 vs All, les différentes matrices de poids correspondantes ainsi que
        la matrice de poids 6 vs 9 et 1 vs 8
        :param perceptron:  perceptron utilisé
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


#################################################################################
#------------------------------ SVM et GridSearch ------------------------------#
#################################################################################

def plot_frontiere_proba(data, f, step=20):
    grid, x, y = make_grid(data=data, step=step)
    plt.contourf(x, y, f(grid).reshape(x.shape), 255)


def SVM(data_type=0, epsilon=0.3, C=10, kernel='linear',
        probability=True, max_iter=100, ax=plt):
    """ Plot ls différents kernels appliqués sur les différents types de données
    """

    trainx,trainy =  gen_arti(nbex=1000,data_type=data_type,epsilon=epsilon)
    testx,testy =  gen_arti(nbex=1000,data_type=data_type,epsilon=epsilon)

    s = svm.SVC(C=C, kernel=kernel, probability=probability, max_iter=max_iter)
    
    s.fit(trainx, trainy)

    err_train = round(1 - s.score(trainx, trainy), 3)
    err_test = round(1 - s.score(testx, testy), 3)

    print("Erreur : train %f, test %f\n"% (err_train, err_test))

    ax.set_title("SVM {} kernel \non data_type {}".format(kernel, data_type))

    f = lambda x: s.predict_proba(x)[:, 0]
    plotAll_in_subplots(testx, testy, f, ax)

    ax.legend(["err_train: {}\nerr_test: {}".format(err_train, err_test)],
              loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2),
               ncol=2, fancybox=True, shadow=True)


def svm_gridSearch(trainx, trainy, testx, testy):
    """ Recherche des meilleurs paramètres à appliquer sur ces données """

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



#################################################################################
#-------------------------- Apprentissage multi-classe -------------------------#
#################################################################################

def multiClass(trainx, trainy, testx, testy):
    """ Traitement des cas multiclasses à partir de classifieurs binaires """

    Unvs1 = multiclass.OneVsOneClassifier(svm.LinearSVC(random_state=0))
    UnvsAll = multiclass.OneVsRestClassifier(svm.LinearSVC(random_state=0))

    Unvs1.fit(trainx, trainy)
    err_Unvs1_train = round(1 - Unvs1.score(trainx, trainy), 3)
    err_Unvs1_test = round(1 - Unvs1.score(testx, testy), 3)

    UnvsAll.fit(trainx, trainy)
    err_UnvsAll_train = round(1 - UnvsAll.score(trainx, trainy), 3)
    err_UnvsAll_test = round(1 - UnvsAll.score(testx, testy), 3)

    
    print("Err_1vs1 : train %f, test %f\n"% (err_Unvs1_train, err_Unvs1_test))
    print("Err_1vsAll : train %f, test %f\n"% (err_UnvsAll_train,
                                               err_UnvsAll_test))

    print("========== Prediction UnvsUn ==============")
    print(Unvs1.predict(testx).shape)

    print("========= Prediction UnvsRest =============")
    print(UnvsAll.predict(testx).shape)
    

#################################################################################
#--------------------------------- String Kernel -------------------------------#
#################################################################################

def sous_sequences(s):
    s = [i for i in range(1, len(s) + 1)]
    return [list(combinations(s, i) for i in range(1, len(s) + 1))]


def l(i):
    """ Longueur d'une sous-sequence """
    return i[-1] - i[0] + 1


def proj(s, lmda=1):
    """ Projection pour une sous-séquence """
    if lmda > 1:
        raise ValueError

    list_sous_sequences, slamda = sous_sequences(s), 0
    for i in list_sous_sequences:
        s_lmda += lmda**l(i)

    return s_lmda


def kernel_string(sigma_n, s, t, lmda=1):
    """ Kernel string
        :param lmda: lambda <= 1
        :param sigma_n: ensemble de mots
        :param s: mot s
    """

    sous_seqi = sous_sequences(s)
    sous_seqj = sous_sequences(t)

    rep = 0
    for i in range(len(sous_seqi)):
        for j in range(sous_seqj):
            rep += lmda**l(i) + lmda**l(j)

    return rep


#################################################################################
#------------------------------------- Main ------------------------------------#
#################################################################################

def main():
    """ Fonction main """

    # Creating the model
    perceptron = Perceptron(max_iter=500, n_jobs=-1)

    # Perceptron du linear model sur les différents types de données
    fig_lm, ax_lm = plt.subplots(ncols=3, figsize=(10,6))
    for dtype in range(3):
        linear_model(perceptron, data_type=dtype, epsilon=0.3, ax=ax_lm[dtype])

    # plt.savefig("linear_model_perceptron.png")

    # Application du perceptron de linear model sur les données usps
    comparaison_usps(perceptron)

    # Application des différents kernels sur l'ensemble des types de données
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    fig_svm, ax_svm = plt.subplots(ncols=4, nrows=3, sharex=True, figsize=(25,15))
    for dtype in range(3):
        for j, kernel in enumerate(kernels):
            SVM(data_type=dtype, epsilon=0.0001, C=50, kernel=kernel, probability=True,
                max_iter=100, ax=ax_svm[dtype][j])

    fig_svm.tight_layout()
    # plt.savefig("svm_data_kernel.png")
    plt.show()


    #------------------------------- Grid Search

    # Données 2D artificiels
    trainx, trainy = gen_arti(nbex=1000, data_type=0, epsilon=0.3)
    testx, testy = gen_arti(nbex=1000, data_type=0, epsilon=0.3)
    # svm_gridSearch(trainx, trainy, testx, testy)

    # Données Usps
    trainux, trainuy = load_usps("USPS_train.txt")
    testux, testuy = load_usps("USPS_test.txt")
    # svm_gridSearch(trainux, trainuy, testux, testuy)

    #------------------------------- Multi Class

    multiClass(trainx, trainy, testx, testy)
    multiClass(trainux, trainuy, testux, testuy)

    #------------------------------- String Kernel





if __name__=="__main__":
    main()

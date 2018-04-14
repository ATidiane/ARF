# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import collections
import pickle as pkl
import pydot
from math import *
from decisiontree import *
# data : tableau (films,features), id2titles : dictionnaire id -> titre
# film, fields : id

with open('imdb_extrait.pkl', 'rb') as f:
    [data, id2titles, fields] = pkl.load(f, encoding='latin1')
# la derniere colonne est le vote


datax = data[:, :32]
datay = np.array([1 if x[33] > 6.5 else -1 for x in data])

print(fields, len(datay))
print(data[:, :33])


def attributes():
    entropie, entropie_cond, dif = [], [], []
    for num in fields.keys():
        entropie.append(entropy(data[:, num]))
        entropie_cond.append(entropy_cond([data[:, num], datay]))
        dif.append(entropie[num] - entropie_cond[num])

    return (entropie, entropie_cond, dif)


def affichage():
    """Affiche l'attribut, l'entropie, l'entropie conditionnel ainsi que
    la différence entre ces deux dernieres."""
    ent, entc, dif = attributes()
    i = 0
    for e, ec, d in zip(ent, entc, dif):
        print(
            "attr: {}, {}, entro: {}, entro_cond: {}, dif: {}".format(
                i,
                fields[i],
                e,
                ec,
                d))
        i += 1


def partitionnement(datax, app):
    """
    Cette fonction prend en parametre le pourcentage de la base d'apprentissage
    et puis renvoie un liste de tuple contenant une base d'exemples d'apprentissage
    de ce pourcentage et une base d'exemples de test pour le reste(1-app).
    """

    indices = np.arange(len(datax))
    np.random.shuffle(indices)

    nb_ex_app = int(len(datax) * app)

    inds_app = indices[:nb_ex_app]
    inds_test = indices[:nb_ex_app]

    data_app, datay_app = datax[inds_app], datay[inds_app]
    data_test, datay_test = datax[inds_test], datay[inds_test]

    return [data_app, datay_app], [data_test, datay_test]


def plot_error_curves(datax):

    fig, ax = plt.subplots(ncols=3, sharex=True)

    for i, p in enumerate([0.2, 0.5, 0.8]):
        app, test = partitionnement(datax, p)
        data_app, datay_app = app
        data_test, datay_test = test

        dt = DecisionTree()

        err_learning, err_test = [], []
        profondeurs = range(0, 30, 2)
        for p in profondeurs:
            dt.max_depth = p
            dt.fit(data_app, datay_app)
            err_learning.append(1 - dt.score(data_app, datay_app))
            err_test.append(1 - dt.score(data_test, datay_test))

        print(err_learning)
        print(err_test)

        ax[i].plot(
            profondeurs,
            err_learning,
            c='magenta',
            label="err_learning")
        ax[i].plot(profondeurs, err_test, c='red', label="err_test")
        ax[i].legend()

    plt.suptitle("Courbe d'erreurs en apprentissage et en test \n"
                 "en fonction de la profondeur de l'arbre")
    plt.show()

###############################################################################
#----------------------------------Main---------------------------------------#
###############################################################################


affichage()

dt = DecisionTree()
dt.max_depth = 1                        # Taille de l'arbre
dt.min_samples_split = 2

# Apprentissage
dt.fit(datax, datay)

# Prédiction
dt.predict(datax[:100, :])


print("\nscore:", dt.score(datax, datay), "\n")

# Pour dessiner l'arbre dans un fichier pdf

#dt.to_pdf("test_tree2{}.pdf".format(dt.max_depth), fields)
print(str(dt))
# print(_)


# Partionnement, Sur et sous apprentissage

plot_error_curves(datax)

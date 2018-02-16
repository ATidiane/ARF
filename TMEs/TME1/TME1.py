# -*- coding: utf-8 -*-

import numpy as np
#import matplotlib.pyplot as plt
import collections
import pickle as pkl
from math import *
from decisiontree import *
# data : tableau (films,features), id2titles : dictionnaire id -> titre film, fields : id 

with open('imdb_extrait.pkl', 'rb') as f:
    [data, id2titles, fields] = pkl.load(f, encoding='latin1') 
# la derniere colonne est le vote
    

datax=data[:,:32]
datay=np.array([1 if x[33]>6.5 else -1 for x in data])

print(fields, len(datay))
print(data[:,:33])

def attributes():
    entropie, entropie_cond, dif = [], [], []
    for num in fields.keys():
        entropie.append(entropy(data[:,num]))
        entropie_cond.append(entropy_cond([data[:,num], datay]))
        dif.append(entropie[num] - entropie_cond[num])
        
    return (entropie, entropie_cond, dif)
    
def affichage():
    """Affiche l'attribut, l'entropie, l'entropie conditionnel ainsi que 
    la différence entre ces deux dernieres."""
    ent, entc, dif = attributes()
    i = 0
    for e, ec, d in zip(ent, entc, dif):
        print("attr: {}, {}, entro: {}, entro_cond: {}, dif: {}".format(i, fields[i], e, ec, d))
        i += 1

def partionnement(app):
    """
    Cette fonction prend en parametre le pourcentage de la base d'apprentissage 
    et puis renvoie un liste de tuple contenant une base d'exemples d'apprentissage
    de ce pourcentage et une base d'exemples de test pour le reste(1-app).
    """
    nb_ex_app = app * len(datax)
    print(nb_ex_app)
    data_app, datay_app = datax[:nb_ex_app], datay[:nb_ex_app]
    data_test, datay_test = datax[nb_ex_app:], datay[:nb_ex_app]

    return [data_app, datay_app], [data_test, datay_test]
    

###############################################################################
#----------------------------------Main---------------------------------------#
###############################################################################

affichage()
        
dt = DecisionTree()
dt.max_depth = 10                        # Taille de l'arbre
dt.min_samples_split = 2
dt.fit(datax, datay)
dt.predict(datax[:5,:])
print("\nscore:", dt.score(datax, datay), "\n")
#dt.to_pdf("~/Bureau/test_tree.pdf", fields) # Pour dessiner l'arbre dans un fichier pdf
str(dt)
print(_)


app, test = partionnement(0.8)
data_app, datay_app = app
data_test, datay_test = test

dtp = DecisionTree()
dtp.max_depth = 5
dt.min_samples_split = 2
dt.fit(data_app, datay_app)
dt.predict()
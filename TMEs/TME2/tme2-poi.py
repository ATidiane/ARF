# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


####################################################################
# -------------------------- Initialisation ---------------------- #
####################################################################

plt.ion()
parismap = mpimg.imread('data/paris-48.806-2.23--48.916-2.48.jpg')

## coordonnees GPS de la carte
xmin,xmax = 2.23,2.48   ## coord_x min et max
ymin,ymax = 48.806,48.916 ## coord_y min et max

poidata = pickle.load(open("data/poi-paris.pkl","rb"))


def show_map():
    plt.imshow(parismap,extent=[xmin,xmax,ymin,ymax],aspect=1.5)
    ## extent pour controler l'echelle du plan



def chooseTypePoi(typepoi):
    ## Creation de la matrice des coordonnees des POI
    geo_mat = np.zeros((len(poidata[typepoi]),2))
    for i,(k,v) in enumerate(poidata[typepoi].items()):
        geo_mat[i,:]=v[0]

    return geo_mat


def affichage(xmin, xmax, ymin, ymax, steps, modele, points=False):
    # Affichage de l'estimation par densité
    
    xx,yy = np.meshgrid(np.linspace(xmin,xmax,steps),np.linspace(ymin,ymax,steps))
    grid = np.c_[xx.ravel(),yy.ravel()]

    res = modele.predict(grid).reshape(steps,steps)
    plt.figure(figsize = (12,7))
    show_map()
    plt.imshow(res,extent=[xmin,xmax,ymin,ymax],interpolation='none',\
                   alpha=0.3,origin = "lower")
    plt.colorbar()
    if points:
        plt.scatter(geo_mat[:,0],geo_mat[:,1],alpha=0.1)
    return res


class ModeleHistogramme:
    
    def __init__(self, xmin, xmax, ymin, ymax, steps=50, resolution=5e-8):
        self.xmin = xmin
        self.ymin = ymin
        self.steps = steps
        self.dx = (xmax) - xmin
        self.dy = (ymax) - ymin
        self.resolution = resolution
        self.base = np.zeros((self.steps, self.steps))

    def get_indices(self, point):
        """ retourne les indices associés au point dans la base """
        
        return int((point[0]-self.xmin)/self.dx*(self.steps) - self.resolution), \
            int((point[1]-self.ymin)/self.dy*(self.steps) - self.resolution)

    def fit(self, data):
        self.nb_points = len(data)
        for point in data:
            indx, indy = self.normalise(point)
            self.base[indx][indy] += 1
        
    def predict(self, grid):
        res = np.zeros(len(grid))   
        for i,point in enumerate(grid):
            indx, indy = self.get_indices(point)
            res[i] = self.base[indx][indy]
        return res / self.nb_points
    


## liste des types de point of interest (poi)
print("Liste des types de POI" , ", ".join(poidata.keys()))

## Choix d'un poi
typepoi = "restaurant"
geo_mat = chooseTypePoi(typepoi)

print(geo_mat)

## Affichage brut des poi
#show_map()

## alpha permet de regler la transparence, s la taille
plt.scatter(geo_mat[:,1],geo_mat[:,0],alpha=0.8,s=3)


###################################################

# discretisation pour l'affichage des modeles d'estimation de densite
steps = 20
res = ModeleHistogramme(xmin, xmax, ymin, ymax, steps)
res.fit(geo_mat)
r = affichage(xmin, xmax, ymin, ymax, steps, res)

# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


plt.ion()
parismap = mpimg.imread('data/paris-48.806-2.23--48.916-2.48.jpg')

## coordonnees GPS de la carte
xmin,xmax = 2.23,2.48   ## coord_x min et max
ymin,ymax = 48.806,48.916 ## coord_y min et max

def show_map():
    plt.imshow(parismap,extent=[xmin,xmax,ymin,ymax],aspect=1.5)
    ## extent pour controler l'echelle du plan


poidata = pickle.load(open("data/poi-paris.pkl","rb"))


def chooseTypePoi(typepoi):
    ## Creation de la matrice des coordonnees des POI
    geo_mat = np.zeros((len(poidata[typepoi]),2))
    for i,(k,v) in enumerate(poidata[typepoi].items()):
        geo_mat[i,:]=v[0]

    return geo_mat


class ModeleHistogramme:
    def __init__(self, xmin=0, ymin=0, xmax=100, ymax=100, nbsteps=10):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.nbsteps = nbsteps
        
    def fit(self, data):
        """ On divise en cases, et pour chacune, on calcule l'estimation de 
        densité """
        xstep = (self.xmax - self.xmin) // self.nbsteps
        ystep = (self.ymax - self.ymin) // self.nbsteps
        mat = np.zeros((self.nbsteps))
        data2 = np.argsort(data)
        
        cpt = 0
        for i, j in zip(range(self.xmin, self.xmax, xstep), range(self.ymin, self.ymax, ystep)):
            pass
    
        print(mat)
    
    def predict(self, grid):
        """ On passe une nouvelle liste de points et on veut prédire,
        l'estimation de densité pour chaque point"""
        pass




## liste des types de point of interest (poi)
print("Liste des types de POI" , ", ".join(poidata.keys()))

## Choix d'un poi
typepoi = "restaurant"
geo_mat = chooseTypePoi(typepoi)

print(geo_mat)
## Affichage brut des poi
show_map()
## alpha permet de regler la transparence, s la taille
plt.scatter(geo_mat[:,1],geo_mat[:,0],alpha=0.8,s=3)


###################################################

# discretisation pour l'affichage des modeles d'estimation de densite
steps = 100
xx,yy = np.meshgrid(np.linspace(xmin,xmax,steps),np.linspace(ymin,ymax,steps))
grid = np.c_[xx.ravel(),yy.ravel()]

# A remplacer par res = monModele.predict(grid).reshape(steps,steps)
# res = np.random.random((steps,steps))
res = ModeleHistogramme()
res.fit(geo_mat)

plt.figure()
show_map()
plt.imshow(res,extent=[xmin,xmax,ymin,ymax],interpolation='none',\
               alpha=0.3,origin = "lower")
plt.colorbar()
plt.scatter(geo_mat[:,0],geo_mat[:,1],alpha=0.3)


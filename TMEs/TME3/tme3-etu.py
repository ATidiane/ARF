import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def make_grid(xmin=-5,xmax=5,ymin=-5,ymax=5,step=20,data=None):
    """ Cree une grille sous forme de matrice 2d de la liste des points
    :return: une matrice 2d contenant les points de la grille, la liste x, la liste y
    """
    if data is not None:
        xmax,xmin,ymax,ymin = np.max(data[:,0]),np.min(data[:,0]),\
                              np.max(data[:,1]),np.min(data[:,1])
    x,y = np.meshgrid(np.arange(xmin,xmax,(xmax-xmin)*1./step),
                      np.arange(ymin,ymax,(ymax-ymin)*1./step))
    grid=np.c_[x.ravel(),y.ravel()]
    return grid, x, y

def load_usps(filename):
    with open(filename,"r") as f:
        f.readline()
        data =[ [float(x) for x in l.split()] for l in f if len(l.split())>2 ]
    tmp = np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

mafonction = ... 
grid,xx,yy = make_grid(-1,3,-1,3,20)
plt.figure()
plt.contourf(xx,yy,mafonction(grid).reshape(xx.shape))



fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(xx, yy, mafonction(grid).reshape(xx.shape),rstride=1,cstride=1,\
	  cmap=cm.gist_rainbow, linewidth=0, antialiased=False)
fig.colorbar(surf)
plt.show()

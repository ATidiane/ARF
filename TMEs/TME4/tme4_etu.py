from arftools import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def decorator_vec(fonc):
    def vecfonc(datax,datay,w,*args,**kwargs):
        if not hasattr(datay,"__len__"):
            datay = np.array([datay])
        datax,datay,w =  datax.reshape(len(datay),-1),datay.reshape(-1,1),w.reshape((1,-1))
        return fonc(datax,datay,w,*args,**kwargs)
    return vecfonc

@decorator_vec
def mse(datax,datay,w):
    """ retourne la moyenne de l'erreur aux moindres carres """
    
    # (w.x - y)² np.mean(((datax * w) - datay)**2)
    return np.mean((np.dot(datax, w.T) - datay)**2)

@decorator_vec
def mse_g(datax,datay,w):
    """ retourne le gradient moyen de l'erreur au moindres carres """
    
    # 2 x (w.x - y)²
    return np.mean(2 * ((np.dot(datax, w.T)) - datay)**2)

@decorator_vec
def hinge(datax,datay,w):
    """ retourn la moyenne de l'erreur hinge """
    
    # Erreur moyenne de -f(x)y
    param2 = -datay * np.dot(datax, w.T)
    return np.maximum(0, param2).mean()

@decorator_vec
def hinge_g(datax,datay,w):
    """ retourne le gradient moyen de l'erreur hinge """
    h = datay * np.dot(datax, w.T)
    h = h < 0
    return np.mean(h * datax * datay)

    
class Lineaire(object):
    def __init__(self,loss=mse,loss_g=mse_g,max_iter=1000,eps=0.01):
        """ :loss: fonction de cout
            :loss_g: gradient de la fonction de cout
            :max_iter: nombre d'iterations
            :eps: pas de gradient
        """
        self.max_iter, self.eps = max_iter,eps
        self.loss, self.loss_g = loss, loss_g

    def fit(self,datax,datay,testx=None,testy=None):
        """ :datax: donnees de train
            :datay: label de train
            :testx: donnees de test
            :testy: label de test
        """
        
        datay = datay.reshape(-1,1)
        N = len(datay)
        datax = datax.reshape(N,-1)
        D = datax.shape[1]
        self.w = np.random.random((1,D))
        
        for i in range(self.max_iter):
            self.w = self.w - (self.eps * self.loss_g(datax, datay, self.w))

    def predict(self,datax):
        if len(datax.shape)==1:
            datax = datax.reshape(1,-1)
            
        return np.sign(np.dot(datax, self.w.T))
                
        

    def score(self,datax,datay):
        nbFound = 0
        for y1, y2 in zip(self.predict(datax), datay):
            if y1*y2 > 0: nbFound += 1

        return (1 - (nbFound / datay.shape[0]))


def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    # Retourne la seconde colonne jusqu'à la dernière tmp[:,1:], ainsi que 
    # la première colonne tmp[:,0]
    return tmp[:,1:],tmp[:,0].astype(int)

def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")



def plot_error(datax,datay,f,step=10):
    grid,x1list,x2list=make_grid(data=datax, xmin=-4,xmax=4,ymin=-4,ymax=4)
    print(x1list.shape, x2list.shape)
    print(np.array([f(datax,datay,w) for w in grid]).shape)
    plt.contourf(x1list,x2list,np.array([f(datax,datay,w) for w in grid]).reshape(x1list.shape),25)
    plt.colorbar()
    plt.show()


def extract_usps(f, firstclass, secondclass):
    pixels, numbers = load_usps(f)
    indexes69 = np.hstack(np.where(numbers == i)[0] for i in (firstclass, secondclass))
    pixels69 = pixels[indexes69]
    numbers69 = numbers[indexes69]
    pixels69y = np.ones(pixels69.shape[0])
    
    for i in range(numbers69.shape[0]):
        if numbers69[i] == firstclass: 
            pixels69y[i] = -1
        else:
            pixels69y[i] = 1
            
    return pixels69, pixels69y

if __name__=="__main__":
    """ Tracer des isocourbes de l'erreur """
    
   
    
    plt.ion()
    trainx,trainy =  gen_arti(nbex=1000,data_type=0,epsilon=1)
    testx,testy =  gen_arti(nbex=1000,data_type=0,epsilon=1)
    grid,x1list,x2list=make_grid(xmin=-4,xmax=4,ymin=-4,ymax=4)
    mse_test = mse(trainx, trainy, np.array([2,1])) 
    mseg_test = mse_g(trainx, trainy, np.array([2,1]))
    hinge_test = hinge(trainx, trainy, np.array([2,1]))
    print(mse_test, mseg_test, hinge_test)
    
    plt.figure()
    plot_error(trainx,trainy,mse)
    plt.figure()
    plot_error(trainx,trainy,hinge)
    perceptron = Lineaire(hinge,hinge_g,max_iter=1000,eps=0.1)
    perceptron.fit(trainx,trainy)
    print("Erreur : train %f, test %f"% (perceptron.score(trainx,trainy),perceptron.score(testx,testy)))
    plt.figure()
    plot_frontiere(trainx,perceptron.predict,200)
    plot_data(trainx,trainy)


    # USPS

    datax, datay = extract_usps("USPS_train.txt", 1, 8) 
    dataTx, dataTy = extract_usps("USPS_test.txt", 1, 8)
     
    show_usps(datax[1])
    
    #plt.figure()
    #plot_error(datax,datay,mse)
    plt.figure()
    plot_error(datax,datay,hinge)
    
    perceptron_usps = Lineaire(hinge,hinge_g,max_iter=1000,eps=0.1)
    perceptron_usps.fit(datax,datay)
    print("Erreur : train %f, test %f"% (perceptron_usps.score(datax, datay),perceptron_usps.score(dataTx,dataTy)))
    plt.figure()
    plot_frontiere(trainx,perceptron_usps.predict,200)
    plot_data(datax,datay)

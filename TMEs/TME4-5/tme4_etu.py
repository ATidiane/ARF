from arftools import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import time


def decorator_vec(fonc):
    def vecfonc(datax,datay,w,*args,**kwargs):
        if not hasattr(datay,"__len__"):
            datay = np.array([datay])
        datax,datay,w =  datax.reshape(len(datay),-1),datay.reshape(-1,1),w.reshape((1,-1))
        return fonc(datax,datay,w,*args,**kwargs)
    return vecfonc


def allow_bias(datax):
    """ Ajoute une colonne de remplie de 1 à datax et retourne la nouvelle
        matrice correspondante.
    """
    adjust = np.ones(datax.shape[0]).reshape(-1, 1)
    return np.hstack((datax, adjust))


@decorator_vec
def mse(datax, datay, w):
    """ retourne la moyenne de l'erreur aux moindres carres """
    # Ajouter une colonne de remplie de 1 dans datax
    if w.shape[1] > datax.shape[1]: datax = allow_bias(datax)
    # (w.x - y)² np.mean(((datax * w) - datay)**2)
    return np.mean((np.dot(datax, w.T) - datay)**2)


@decorator_vec
def mse_g(datax, datay, w):
    """ retourne le gradient moyen de l'erreur au moindres carres """
    if w.shape[1] > datax.shape[1]: datax = allow_bias(datax)
    # 2 x (w.x - y)²
    return np.mean(2 * ((np.dot(datax, w.T)) - datay)**2)

@decorator_vec
def hinge(datax, datay, w):
    """ retourn la moyenne de l'erreur hinge """
    if w.shape[1] > datax.shape[1]: datax = allow_bias(datax)
    # Erreur moyenne de -f(x)y
    param2 = -datay * np.dot(datax, w.T)
    return np.maximum(0, param2).mean()


@decorator_vec
def hinge_g(datax, datay, w):
    """ retourne le gradient moyen de l'erreur hinge """

    if w.shape[1] > datax.shape[1]: datax = allow_bias(datax)
    h = np.array((datay * np.dot(datax, w.T)) < 0)
    return np.mean(h * datax * datay)


def stochastic(vectorx, vectory, w):
    """ Retourne l'erreur aux moindres carres pour UN exemple de data,
        cette pratique de calculer la descente de gradient juste pour un ex
        à chaque itération est appelée descente de gradient stochastique ou
        Stochastic gradient descent.
    """

    return ((np.dot(vectorx.reshape(1,-1), w.T) - vectory)**2)/2


def stochastic_g(vectorx, vectory, w):
    """ Retourne le gradient de l'erreur aux moindres carres pour UN exemple de
        datax.
    """

    return (np.dot(vectorx.reshape(1,-1), w.T) - vectory)



class Lineaire(object):
    def __init__(self,loss=mse,loss_g=mse_g,max_iter=1000,eps=0.01, bias=False):
        """ :loss: fonction de cout
            :loss_g: gradient de la fonction de cout
            :max_iter: nombre d'iterations
            :eps: pas de gradient
        """
        self.max_iter, self.eps = max_iter,eps
        self.loss, self.loss_g = loss, loss_g
        self.bias = bias


    def bias_decorator(fonc):
        def adjust_datax(self, datax, *args, **kwargs):
            if self.bias:
                datax = allow_bias(datax)
            return fonc(self, datax, *args, **kwargs)

        return adjust_datax


    def batch_fit(self, datax, datay):
        """ Classic gradient descent Learning """
        print("=================Batch=================\n")
        self.loss_g = hinge_g
        for i in range(self.max_iter):
            self.w = self.w - (self.eps * self.loss_g(datax, datay, self.w))

    def stochastic_fit(self, datax, datay):
        """ Stochastic gradient descent Learning """
        print("===============Stochastic==============\n")
        self.loss_g = stochastic_g
        data = np.hstack((datax, datay))
        # It's a good thing to shuffle the datax.
        np.random.shuffle(data)
        datax, datay = data[:,:2], data[:,-1]


        #plt.figure()
        grid,x,y=make_grid(datax,200)
        step = 0
        for _ in range(self.max_iter):
            for vectorx, vectory in zip(datax, datay):
                self.w -= (self.eps * self.loss_g(vectorx, vectory, self.w))


    def stochastic_fit_animation(self, datax, datay):
        """ Stochastic gradient descent Learning """
        print("===============Stochastic==============\n")
        self.loss_g = stochastic_g
        data = np.hstack((datax, datay))
        # It's a good thing to shuffle the datax.
        np.random.shuffle(data)
        datax, datay = data[:,:2], data[:,-1]

        #plt.figure()
        grid,x,y=make_grid(datax,200)
        step = 0
        for _ in range(self.max_iter):
            for vectorx, vectory in zip(datax, datay):
                self.w -= (self.eps * self.loss_g(vectorx, vectory, self.w))

                # Show video of learning
                plt.title("Stochastic animation, step %d"%(step))
                plt.contourf(x,y,self.predict(grid).reshape(x.shape),colors=('gray','blue'),levels=[-1,0,1])
                plot_data(datax, datay)
                plt.pause(0.0000000000000001)
                step += 1
                #plt.close()

                #if t == 0: line, = plt.plot(range(len(self.w[0])), self.w[0], 'b-')
                #else: line.set_ydata(self.w[0])


    def minibatch_fit(self, datax, datay, batch_size=10):
        """ Mini-Batch gradient descent Learning """
        print("===============Mini-Batch==============\n")
        for _ in range(self.max_iter):
            for i in range(0, datax.shape[0], batch_size):
                # On prend seulement batch_size données sur toutes les données.
                batchx, batchy = datax[i:i+batch_size], datay[i:i+batch_size]
                # Et on essaye de progresser avec cela.
                self.w -= (self.eps * self.loss_g(batchx, batchy, self.w))


    @bias_decorator
    def fit(self,datax,datay,testx=None,testy=None,gradient_descent="batch",
            batch_size=10):
        """ :datax: donnees de train
            :datay: label de train
            :testx: donnees de test
            :testy: label de test
            :method: can be either, batch_fit, stochastic_fit or mini_batch_fit.
        """

        N = len(datay)
        datay = datay.reshape(-1,1)
        D = datax.shape[1]
        self.w = np.random.random((1,D))

        # Let's learn !
        if gradient_descent == "batch": self.batch_fit(datax, datay)
        elif gradient_descent == "stochastic": self.stochastic_fit(datax, datay)
        elif gradient_descent == "stochastic_animation":
            self.stochastic_fit_animation(datax, datay)
        else: self.minibatch_fit(datax, datay, batch_size)


    @bias_decorator
    def predict(self,datax):
        if len(datax.shape)==1:
            datax = datax.reshape(1,-1)

        return np.sign(np.dot(datax, self.w.T))


    def score(self,datax,datay):
        return (1 - np.mean((self.predict(datax) == datay[:, np.newaxis])))


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
    plt.contourf(x1list,x2list,np.array([f(datax,datay,w) for w in grid]).reshape(x1list.shape),25)
    plt.colorbar()
    plt.show()


def extract_usps(fichier, firstclass, secondclass):
    """ Permet de construire un ensemble de données datax, datay avec les
        données USPS, juste en lui spécifiant en paramètre:
        :param fichier: le fichier,
        :param firstclass: le premier chiffre dont on veut extraire les données
        :param secondclass: le second chiffre.
        :return: datax, datay
    """

    pixels, numbers = load_usps(fichier)
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



def learn_plot_perceptron(perceptron, trainx, trainy, testx, testy, gradient_descent,
                          title="Batch gradient descent"):
    perceptron.fit(trainx,trainy, gradient_descent=gradient_descent)
    print("Erreur : train %f, test %f\n"% (perceptron.score(trainx,trainy),perceptron.score(testx,testy)))
    plt.figure()
    plot_frontiere(testx,perceptron.predict,200)
    plot_data(testx,testy, title=title)
    plt.savefig(gradient_descent+".png")



def main():
    """ Tracer des isocourbes de l'erreur """

    plt.ion()
    trainx,trainy =  gen_arti(nbex=1000,data_type=0,epsilon=0.3)
    testx,testy =  gen_arti(nbex=1000,data_type=0,epsilon=0.3)
    grid,x1list,x2list=make_grid(xmin=-4,xmax=4,ymin=-4,ymax=4)

    # Plot error for test
    plt.figure()
    plot_error(trainx,trainy,mse)
    plt.savefig("errormse")
    plt.figure()
    plot_error(trainx,trainy,hinge)
    plt.savefig("errorhinge")

    # Batch gradient descent
    perceptron = Lineaire(loss=mse,loss_g=mse_g,max_iter=10000,eps=0.1,
                          bias=False)
    learn_plot_perceptron(perceptron, trainx, trainy, testx, testy,
                          gradient_descent = "batch",
                          title = "Batch gradient descent")

    # Stochastic gradient descent
    perceptron2 = Lineaire(loss=stochastic,loss_g=stochastic_g,max_iter=1,
                           eps=0.1,bias=False)
    learn_plot_perceptron(perceptron2, trainx, trainy, testx, testy,
                          gradient_descent = "stochastic",
                          title = "Stochastic gradient descent")

    # Mini-Batch gradient descent
    perceptron3 = Lineaire(loss=mse,loss_g=mse_g,max_iter=1000,
                           eps=0.1,bias=False)
    learn_plot_perceptron(perceptron3, trainx, trainy, testx, testy,
                          gradient_descent = "minibatch",
                          title = "Mini-Batch gradient descent")

    # Stochastic gradient descent Animation
    perceptron2 = Lineaire(loss=stochastic,loss_g=stochastic_g,max_iter=1,
                           eps=0.1,bias=False)
    # learn_plot_perceptron(perceptron2, trainx, trainy, testx, testy,
    #                      gradient_descent = "stochastic_animation",
    #                      title = "Stochastic gradient descent")

    # USPS DATA

    datax, datay = extract_usps("USPS_train.txt", 6, 9)
    dataTx, dataTy = extract_usps("USPS_test.txt", 6, 9)

    # Affichage des données USPS
    show_usps(datax[1])

    #usps_perceptron = Lineaire(loss=hinge,loss_g=hinge_g,max_iter=1,
    #                       eps=0.1,bias=False)
    #learn_plot_perceptron(usps_perceptron, datax, datay, dataTx, dataTy,
    #                      gradient_descent = "minibatch",
    #                      title = "Minibatch gradient descent")

    #perceptron_usps = Lineaire(hinge,hinge_g,max_iter=1000,eps=0.1)
    #perceptron_usps.fit(datax,datay, gra)
    #print("Erreur : train %f, test %f"% (perceptron_usps.score(datax, datay),perceptron_usps.score(dataTx,dataTy)))
    #plt.figure()
    #plot_frontiere(trainx,perceptron_usps.predict,200)
    #plot_data(datax,datay)



if __name__=="__main__":

    main()

# -*- coding: utf-8 -*-

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
            plt.contourf(x,y,self.predict(grid).reshape(x.shape),
                         colors=('gray','blue'),levels=[-1,0,1])
            plot_data(datax, datay)
            plt.pause(0.0000000000000001)
            step += 1
            plt.close()


def minibatch_fit(self, datax, datay, batch_size=10):
    """ Mini-Batch gradient descent Learning """
    print("===============Mini-Batch==============\n")
    for _ in range(self.max_iter):
        for i in range(0, datax.shape[0], batch_size):
            # On prend seulement batch_size données sur toutes les données.
            batchx, batchy = datax[i:i+batch_size], datay[i:i+batch_size]
            # Et on essaye de progresser avec cela.
            self.w -= (self.eps * self.loss_g(batchx, batchy, self.w))
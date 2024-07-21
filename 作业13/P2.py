# Homework 13 Problem2 : DFP
# different init

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import time
plt.style.use('default')

class dfp:
    def __init__(self,x0,alpha=0.2,beta=0.5,eta = 1e-5) -> None:
        self.x = x0
        self.eta = eta
        self.alpha = alpha 
        self.beta = beta
        self.pstar = 0
        self.deltay = []
        self.H0 = np.eye(2)
        
    def value(self,x):
        fx = x[0]**4/4+x[1]**2/2-x[0]*x[1]+x[0]-x[1]
        return fx
    
    def dx(self,x):
        dx = [x[0]**3-x[1]+1,x[1]-x[0]-1]
        return np.array(dx)    
        
    def gradientnorm(self,dx) -> float:
        factor = np.sqrt(np.sum(dx**2))
        return factor 
    
    def iteration(self):
        g = self.dx(self.x)
        H = self.H0
        d = -np.dot(self.H0,g)
        gs = [g]
        while self.gradientnorm(g) >= self.eta:
            t = 1.0
            while self.value(self.x+t*d) > self.value(self.x) + self.alpha*t*np.dot(self.dx(self.x),d):
                t *= self.beta
            self.deltay.append(self.value(self.x)-self.pstar)
            
            # update
            self.x += t*d
            g = self.dx(self.x) #k+1
            gs.append(g)
            deltax = (t * d)
            deltag = (gs[-1]-gs[-2])
            H +=  np.outer(deltax,deltax)/np.dot(deltax.T,deltag)-\
                np.outer((H@deltag),(H@deltag))/(deltag.T@H@deltag)
            d = -np.dot(H,g) #k+1
            if self.gradientnorm(g) < self.eta:
                break
        print(self.x)

x = [np.array([0.0,0.0]),np.array([1.5,1.0])]
for i in range(2):
    x0 = x[i]
    optimizer = dfp(x0=x0)
    optimizer.iteration()
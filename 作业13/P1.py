# Homework 13 Problem1 : conjugate gradient 
# compare different formula preformance
# alpha = 1; n = 100

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import time
plt.style.use('default')

class ConjugateNewton:
    def __init__(self,x0,method:str,a=1,alpha=0.3,beta=0.5,eta = 1e-5) -> None:
        self.x = x0
        self.eta = eta
        self.a = a
        self.alpha = alpha 
        self.beta = beta
        self.size = len(x0)
        self.pstar = 0
        self.deltay = []
        self.method = method
        
    def value(self,x):
        fx = 0
        for i in range(int(self.size/2)):
            fx += self.a*(x[2*i+1]-x[2*i]**2)**2+(1-x[2*i])**2
        return fx
    
    def dx(self,x):
        dx = []
        for i in range(int(self.size/2)):
            dx.append((4*x[2*i]**3-4*x[2*i]*x[2*i+1])*self.a+2*x[2*i]-2)
            dx.append(2*(x[2*i+1]-x[2*i]**2)*self.a)
        return np.array(dx)
    
    def gradientnorm(self,dx) -> float:
        factor = np.sqrt(np.sum(dx**2))
        return factor 
    
    def iteration(self):
        g = self.dx(self.x)
        d = -g
        gs = [g]
        while self.gradientnorm(g) >= self.eta:
            t = 1.0
            while self.value(self.x+t*d) > self.value(self.x) + self.alpha*t*np.dot(self.dx(self.x),d):
                t *= self.beta

            self.deltay.append(self.value(self.x)-self.pstar)
            # update
            self.x += t*d
            g = self.dx(self.x)
            gs.append(g)
            if self.gradientnorm(g) < self.eta:
                break
            if self.method == 'HS':
                beta = np.dot(gs[-1],gs[-1]-gs[-2])/(np.dot(d,gs[-1]-gs[-2]))
            elif self.method == 'PR':
                beta = np.dot(gs[-1],gs[-1]-gs[-2])/(np.dot(gs[-2],gs[-2]))
            else:
                beta = np.dot(gs[-1],gs[-1])/(np.dot(gs[-2],gs[-2]))
            d = -g+beta*d

        
    def plot(self):
        iterations = len(self.deltay)
        x = np.arange(iterations)
        plt.plot(x,np.log(np.array(self.deltay)),'-o')
        plt.ylabel(r'$log(f(x)-p^*)$')
        plt.xlabel('Iterations')
        plt.title(f'method {self.method}')   
        plt.savefig(f"P1_f1_{self.method}.png", bbox_inches='tight')
        plt.show()   

methods = ['HS','PR','FR']
for _ in range(len(methods)):
    x0 = np.array([-1.0]*100)
    method = methods[_]
    optimizer = ConjugateNewton(x0=x0.copy(),a=1,method=method)
    optimizer.iteration()
    optimizer.plot()

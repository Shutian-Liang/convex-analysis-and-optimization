# Homework 13 Problem4 : DFP + BFGS
# Compare different algorithms

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import copy
plt.style.use('default')

class optimizer:
    def __init__(self,x0,H0,alpha=0.6,beta=0.6,eta = 1e-5) -> None:
        self.x = x0
        self.eta = eta
        self.alpha = alpha 
        self.beta = beta
        self.pstar = 0
        self.deltay = []
        self.H = H0
        self.name = None
        self.positive = []
        
    def value(self,x):
        x1,x2,x3 = x[0],x[1],x[2]
        fx = (3-x1)**2+7*(x2-x1**2)**2+9*(x3-x1-x2**2)**2
        return fx 
    
    def dx(self,x):
        x1,x2,x3 = x[0],x[1],x[2]
        dx1 = -2*(3-x1)-28*x1*(x2-x1**2)-18*(x3-x1-x2**2)
        dx2 = 14*(x2-x1**2)-36*x2*(x3-x1-x2**2)
        dx3 = 18*(x3-x1-x2**2)
        dx = np.array([dx1,dx2,dx3])
        return dx  
    
    def gradientnorm(self,dx) -> float:
        factor = np.sqrt(np.sum(dx**2))
        return factor 
    
    def testpositive(self,x) -> int:
        eigenvalues = np.linalg.eigvals(x)
        eigenvalues.sort()
        if eigenvalues[0] > 0:
            positive = 1
        else:
            positive = 0
        return positive
    
    def plot(self):
        iterations = len(self.deltay)
        x = np.arange(iterations)
        fig,axes = plt.subplots(1,2,figsize=(12,6))
        axes[0].plot(x,np.log(np.array(self.deltay)),'-o')
        axes[0].set_ylabel(r'$log(f(x)-p^*)$')
        axes[0].set_xlabel('Iterations')
        axes[0].set_title(f'method {self.name}')  
        
        axes[1].plot(x,self.positive,'-o')
        axes[1].set_ylabel('If positive')
        axes[1].set_xlabel('Iterations')
        axes[1].set_title(f'method {self.name}')  
        plt.savefig(f"{self.name}.png", bbox_inches='tight')
        plt.show()   

class dfp(optimizer):
    def __init__(self, x0, H0, alpha=0.6, beta=0.6, eta=0.00001) -> None:
        super().__init__(x0, H0, alpha, beta, eta)
        self.name = 'dfp'
    
    def iteration(self):
        g = self.dx(self.x)
        d = -np.dot(self.H,g)
        gs = [g]
        while self.gradientnorm(g) >= self.eta:
            
            #backtracking
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
            self.H +=  np.outer(deltax,deltax)/(np.dot(deltag,deltax))-np.outer((self.H@deltag),(deltag@self.H))/(deltag@self.H@deltag)
            positive = self.testpositive(self.H)
            self.positive.append(positive)
            d = -np.dot(self.H,g) #k+1
            if self.gradientnorm(g) < self.eta:
                break

class bfgs(optimizer):
    def __init__(self, x0, H0, alpha=0.5, beta=0.6, eta=0.00001) -> None:
        super().__init__(x0, H0, alpha, beta, eta)
        self.name = 'bfgs'
        
    def iteration(self):
        g = self.dx(self.x)
        d = -np.dot(self.H,g)
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
            self.H +=  (1+(deltag@self.H@deltag)/(deltag@deltax))*\
                np.outer(deltax,deltax)/(deltax@deltag)-\
                (self.H@(np.outer(deltag,deltax))+(self.H@np.outer(deltag,deltax)).T)/(deltag@deltax)    
            positive = self.testpositive(self.H)
            self.positive.append(positive)
            d = -np.dot(self.H,g) #k+1
            if self.gradientnorm(g) < self.eta:
                break  

H0 = np.eye(3)
x0 = np.array([1.0,1.0,1.0])
dfp_optimizer = dfp(x0=x0.copy(),H0=H0)
bfgs_optimizer = bfgs(x0=x0.copy(),H0=H0)
dfp_optimizer.iteration()
dfp_optimizer.plot()
bfgs_optimizer.iteration()
bfgs_optimizer.plot()

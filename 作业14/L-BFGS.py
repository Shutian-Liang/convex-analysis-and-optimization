# Homework 14 Problem1 : L-BFGS
# compare different memory m
# alpha = 1; n = 100; 
# 在k时尚未update

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import time
import copy
plt.style.use('default')

class L_BFGS:
    def __init__(self,x0,size,a=1,alpha=0.8,beta=0.6,eta = 1e-5) -> None:
        self.x = x0
        self.eta = eta
        self.alpha = alpha 
        self.beta = beta
        self.a = a
        self.m = size
        self.memory = []
        self.size = len(x0)
        self.pstar = 0
        self.deltay = []
        
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
    
    def save(self,deltax,deltag):
        size = len(self.memory)
        if size >= self.m:
            self.memory.pop(0)
        self.memory.append([deltax,deltag])
        
    def gradientnorm(self,dx) -> float:
        factor = np.sqrt(np.sum(dx**2))
        return factor 
    
    def backtracking(self,x) -> float: 
        t = 1
        deltax = self.recursion(self.dx(x))
        while self.value(x+t*deltax) > self.value(x) + self.alpha*t*np.dot(self.dx(x).T,deltax):
            t *= self.beta
        return t
    
    def getdx(self,i):
        deltax = self.memory[i][0]
        return deltax 
    
    def getdg(self,i):
        deltag = self.memory[i][1]
        return deltag
    
    def recursion(self,dx): 
        q = dx
        m = len(self.memory)
        if m > 0:
            gamma = (self.getdx(-1)@self.getdg(-1))/(self.getdg(-1)@self.getdg(-1))
        else:
            gamma = 1    
        H = np.eye(self.size)*gamma
        alphas = []
        for i in range(m):
            deltax = self.getdx(-1-i)
            deltag = self.getdg(-1-i)
            rho = 1/(deltax@deltag)
            alpha = rho*deltax@q
            alphas.append(alpha)
            q = q -  alpha*deltag
        
        p = H@q 
        for i in range(m):
            deltax = self.getdx(i)
            deltag = self.getdg(i)
            rho = 1/(deltax@deltag)
            alpha = alphas[-1-i]
            beta = rho*deltag@p
            p += (alpha-beta)*deltax 
        return -p
        
    def iteration(self):
        g_new = self.dx(self.x)
        while self.gradientnorm(g_new) >= self.eta:
            g_old = self.dx(self.x)
            alpha = self.backtracking(self.x)
            p = self.recursion(g_old)
            self.x += alpha*p   
            
            # update
            self.deltay.append(self.value(self.x))
            g_new = self.dx(self.x)
            deltag = g_new - g_old
            deltax = alpha*p 
            self.save(deltax=deltax,deltag=deltag)


    def plot(self):
        iterations = len(self.deltay)
        x = np.arange(iterations)
        plt.plot(x,np.log(np.array(self.deltay)),'-o')
        plt.ylabel(r'$log(f(x)-p^*)$')
        plt.xlabel('Iterations')
        plt.title(f'L-BFGS')   
        plt.savefig(f"L-BFGS-{self.m}.png", bbox_inches='tight')
        plt.show()   

x0 = np.array([-1.0]*2000)
optimizer = L_BFGS(x0=x0,size=5)
optimizer.iteration()
optimizer.plot()
    
    
    
    
    
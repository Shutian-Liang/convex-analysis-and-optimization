# Homework 12 Problem1 : Damped Newton Method
# Methods: Damped Newton method 
# Goals: draw pictures of log(f-p*) V.S. iterations and running time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import time
plt.style.use('default')

class DampedN:
    def __init__(self,x,alpha,beta,eta) -> None:
        self.x = x
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.pstar = 0  # where x1 = x2 = 1
        self.deltay = [self.value(self.x)-self.pstar]
        self.time = [0]
        
    def value(self,x) -> float:
        formula = 100*(x[1]-x[0]**2)**2+(1-x[0])**2
        return formula
    
    def dx(self,x) -> np.array:
        dx1 = 400*x[0]**3-400*x[0]*x[1]+2*x[0]-2
        dx2 = 200*x[1]-200*x[0]**2
        dx = np.array([dx1,dx2])
        return dx  
    
    def hx(self,x) -> np.array:
        hx11 = 1200*x[0]**2-400*x[1]+2
        hx12 = -400*x[0]
        hx21 = -400*x[0]
        hx22 = 200
        hx = np.array([[hx11,hx12],[hx21,hx22]])
        return hx 
    
    def deltax(self,x) -> np.array:
        gx = self.dx(x)
        Fx = self.hx(x)
        inv_Fx = np.linalg.inv(Fx)
        deltax = -np.dot(inv_Fx, gx)
        return deltax
        
    def backtracking(self,x) -> float: 
        t = 1
        deltax = self.deltax(x)
        while self.value(x+t*deltax) > self.value(x) + self.alpha*t*np.dot(self.dx(x).T,deltax):
            t *= self.beta
        return t
    
    def gradientnorm(self,dx) -> float:
        factor = np.sqrt(np.sum(dx**2))
        return factor 
    
    def iteration(self):
        t = self.backtracking(self.x)
        dx = self.dx(self.x)
        start_time = time.perf_counter()
        while self.gradientnorm(dx) >= self.eta:
            deltax = self.deltax(self.x)
            self.x += t*deltax
            finish_time = time.perf_counter()
            self.time.append(finish_time-start_time)
            deltay = self.value(self.x) - self.pstar
            self.deltay.append(deltay)
            dx = self.dx(self.x)
            t = self.backtracking(self.x)
    
    def plot(self) -> plt:
        fig,axes = plt.subplots(1,2,figsize=(12,6))
        iterations = len(self.deltay)
        x = np.arange(iterations)
        axes[0].plot(x,np.log(np.array(self.deltay)),'-o')
        axes[0].set_xticks(np.arange(iterations))
        axes[0].set_ylabel(r'$log(f(x)-p^*)$')
        axes[0].set_xlabel('Iterations')
        axes[0].set_title(r'$log(f(x)-p^*)$ V.S. Iterations')
        
        axes[1].plot(self.time,np.log(np.array(self.deltay)),'-o')
        axes[1].set_ylabel(r'$log(f(x)-p^*)$')
        axes[1].set_xlabel('Time(s)')
        axes[1].set_title(r'$log(f(x)-p^*)$ V.S. Runnning Time')
        print(self.time)
        plt.savefig("P1_f1.png", bbox_inches='tight')
        plt.show()

# start experiment
x0 = np.array([2.0,2.0])
dn = DampedN(x0,alpha=0.2,beta=0.5,eta=1e-5)
dn.iteration()
dn.plot()

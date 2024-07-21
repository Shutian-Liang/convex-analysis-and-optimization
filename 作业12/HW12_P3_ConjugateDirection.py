# Homework 12 Problem3 : Conjugate Direction Methods
# Methods: Conjugate Direction Methods
# Goals: draw pictures of log(f-p*) V.S. iteration k 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import copy
plt.style.use('default')
np.random.seed(0)

class ConjugateNewton:
    def __init__(self,x0,Q:np.array,b:np.array,eta = 1e-5) -> None:
        self.Q = Q
        self.b = b
        self.x = x0
        self.eta = eta
        self.pstar = -1/2*self.b@(np.linalg.inv(Q)).T@self.b
        self.deltay = []
        
    def value(self,x):
        fx = 1/2*x.T@self.Q@x-x.T@self.b
        return fx
    
    def dx(self,x):
        dx = self.Q@x-self.b
        return dx
    
    def gradientnorm(self,dx) -> float:
        factor = np.sqrt(np.sum(dx**2))
        return factor 
    
    def alpha(self,g,d):
        alpha = -(g.T@d)/(d.T@self.Q@d)
        return alpha
    
    def beta(self,g,d):
        beta = (g.T@self.Q@d)/(d.T@self.Q@d)
        return beta
    
    def iteration(self):
        g = self.dx(self.x)
        d = -g
        while self.gradientnorm(g) >= self.eta:
            alpha = self.alpha(g,d)
            self.deltay.append(self.value(self.x)-self.pstar)
            # update
            self.x += alpha*d
            g = self.dx(self.x)
            if self.gradientnorm(g) < self.eta:
                break
            beta = self.beta(g,d)
            d = -g+beta*d
        
    def plot(self):
        iterations = len(self.deltay)
        x = np.arange(iterations)
        plt.plot(x,np.log(np.array(self.deltay)))
        plt.ylabel(r'$log(f(x)-p^*)$')
        plt.xlabel('Iterations')
        plt.title(r'Conjugate Method')   
        plt.savefig("P3_f1.png", bbox_inches='tight')
        plt.show()   

# generate symmetric positive definite matrix Q
def generate_symmetric_matrix(n):
    # Step 1: Generate a random positive diagonal matrix
    diag = np.random.uniform(1, 1000, size=n)  # Diagonal elements between 1 and 1000
    D = np.diag(diag)
    
    # Step 2: Generate a random orthogonal matrix
    Q, _ = np.linalg.qr(np.random.rand(n, n))
    
    # Step 3: Generate the symmetric matrix using similarity transformation
    A = np.matmul(np.matmul(Q, D), np.transpose(Q))
    
    return A

# Generate a symmetric matrix until its condition number > 100
condition_number = 0
while condition_number <= 100:
    Q = generate_symmetric_matrix(1000)
    condition_number = np.linalg.cond(Q)

# random generate a vector b
b = np.random.rand(1,1000).flatten()
x0 = np.random.randint(1,1000,1000).astype('float64')

conjugate = ConjugateNewton(x0=x0,Q=Q,b=b)
conjugate.iteration()
conjugate.plot()


import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

class GradientDecent:
    def __init__(self,a:np.array,x0:np.array,alpha: float,beta:float,eta: float) -> None:
        self.a = a
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.deltay = [] # save for the f - p*
        self.length = []  # save for the step length
        self.x = x0
        self.pstar = 2*self.a.shape[0]
        
    def value(self,x) -> float:
        return np.sum(np.exp(np.dot(a,x))+np.exp(-np.dot(a,x)))
    
    def gradient(self,x) -> np.array:
        gradient = np.dot(a.T,np.exp(np.dot(a,x))-np.exp(-np.dot(a,x)))
        return gradient
    
    def gradientnorm(self,gradient) -> float:
        factor = np.sqrt(np.sum(gradient**2))
        return factor
        
    def backtracking(self,x,gradient) -> float:
        t = 1
        while self.value(x-t*gradient) > self.value(x) - self.alpha*t*np.dot(gradient,gradient):
            t *= self.beta
        return t
    
    def iteration(self) -> np.array:
        gradient = self.gradient(self.x)
        t = self.backtracking(self.x,gradient)
        while self.gradientnorm(gradient) >= self.eta:
            deltay = self.value(self.x) - self.pstar
            self.deltay.append(deltay)
            self.length.append(self.gradientnorm(t*gradient))
            
            self.x -= t*gradient
            gradient = self.gradient(self.x)
            t = self.backtracking(self.x,gradient)
        return np.array(self.length),np.array(self.deltay),len(self.length)

m = 2
n = 2
np.random.seed(0)
a = np.random.rand(m,n)
x0 = np.ones(n)
descent = GradientDecent(a,x0,alpha = 0.2,beta= 0.5,eta=1e-5)
step,deltay,iter = descent.iteration() 

# draw pictures
x = np.arange(len(deltay))
fig,axes = plt.subplots(1,2,figsize=(12,6))
axes[0].plot(x,np.log(deltay))
axes[0].set_xlabel('Iterations')
axes[0].set_ylabel(r'$log(f-p^*)$')

axes[1].plot(x,step)
axes[1].set_xlabel('Iterations')
axes[1].set_ylabel('step length')
plt.savefig("P1_f1.png", bbox_inches='tight')
plt.show()

# Different parameter combinations
alphas = np.arange(0.1,0.31,0.05)
betas = np.arange(0.1,0.85,0.1)
def paracombine(alphas,betas,a,x0,iteration):
    num_a = len(alphas)
    num_b = len(betas)
    iterations = np.zeros(shape=(num_a,num_b))
    for t in range(iteration): 
        for i in range(num_a):
            for j in range(num_b):
                a = np.random.rand(m,n)
                x0 = np.ones(n)
                _,_,iter = GradientDecent(a,x0,alpha = alphas[i],beta= betas[j],eta=1e-5).iteration() 
                print(f"iter:{t} alpha: {alphas[i]}, beta: {betas[j]}, iterations: {iter}")
                iterations[i][j] += iter
    return iterations/iteration
iterations = paracombine(alphas,betas,a,x0,50)

f2 = sns.heatmap(iterations,cmap='viridis')
alphas_sorted = sorted(alphas)
f2.set_xticks(np.arange(len(betas)) + 0.5, ['%.3f' % b for b in betas])
f2.set_yticks(np.arange(len(alphas)) + 0.5, ['%.3f' % a for a in reversed(alphas)])
f2.set_xlabel(r'$\beta$')
f2.set_ylabel(r'$\alpha$')
plt.savefig("P1_f2.png", bbox_inches='tight')
plt.show()
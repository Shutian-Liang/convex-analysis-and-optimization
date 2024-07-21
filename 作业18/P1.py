import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

# Generate some data
m, n = 200, 300
A = 5+5*np.random.randn(m, n)
b = 5 + np.random.randn(m)
gamma = 1
x0 = np.zeros(n)

class penalty:
    def __init__(self,method, A,b,x0,gamma,eta=1e-2,alpha=0.3,beta=0.5):
        self.gamma = gamma
        self.A = A
        self.b = b
        self.x = x0
        self.eta = eta
        self.ys = []
        self.qs = []
        self.method = method
        self.alpha = alpha
        self.beta = beta
        self.xopt = self.A.T@np.linalg.inv(self.A@self.A.T)@self.b
        self.fstar = 0.5*np.linalg.norm(self.xopt,2)
    
    def f(self,x):
        return 0.5*np.sqrt(x@x)
    
    def p_abs(self,x):
        return self.f(x) + self.gamma*np.linalg.norm(self.A@x-b,1)
    
    def p_cb(self,x):
        return self.f(x) + self.gamma*np.linalg.norm(self.A@x-b,2)**2
    
    def dp_abs(self,x):
        return x/(2*np.linalg.norm(x,2)) + self.gamma * self.A.T @ (self.A @ x - self.b)

    def dp_cb(self,x):
        return x/(2*np.linalg.norm(x,2)) + self.gamma * 2 * self.A.T @ (self.A @ x - self.b)
    
    def iteration(self):
        while np.linalg.norm(self.x - self.xopt,2) > self.eta:
            if self.method == 'abs':
                direction = -self.dp_abs(self.x)/np.linalg.norm(self.dp_abs(self.x),2)
                t = 1
                while self.p_abs(self.x + t*direction) > self.p_abs(self.x) + self.alpha*t*np.dot(direction,direction):
                    t *= self.beta

            elif self.method == 'cb':
                t = 1
                direction = -self.dp_cb(self.x)/np.linalg.norm(self.dp_cb(self.x),2)
                while self.p_cb(self.x + t*direction) > self.p_cb(self.x) + self.alpha*t*np.dot(direction,direction):
                    t *= self.beta
            
            self.x = self.x + t*direction
            if self.method == 'abs':
                self.ys.append(self.f(self.x))
            else:
                self.ys.append(self.f(self.x))
            
            if len(self.ys) > 2000:
                break
            
            if (self.f(self.x) - self.fstar) < self.eta:
                break

    def plot(self):
        iterations = len(self.ys)
        x = np.arange(iterations)
        plt.plot(x,np.log(np.array(self.ys)-self.fstar),'-')
        plt.ylabel(r'$log(f(x)-f^*)$')
        plt.xlabel('Iterations')
        plt.title(f'{self.method} method')   
        plt.savefig(f"{self.method}.png", bbox_inches='tight')
        plt.show()  

gammas = [0.1,1,10,100,1000]   
for gamma in gammas: 
    abs_p = penalty('abs',A.copy(),b.copy(),np.ones(n),gamma)
    abs_p.iteration()
    print(abs_p.ys[-1])
    print(np.linalg.norm(abs_p.x - abs_p.xopt,2))
    abs_p.plot()

gammas = [0.1,1,10,100,1000]   
for gamma in gammas: 
    cb_p = penalty('cb',A.copy(),b.copy(),np.ones(n),gamma)
    cb_p.iteration()
    print(cb_p.ys[-1])
    print(np.linalg.norm(cb_p.x - cb_p.xopt,2))
    cb_p.plot()


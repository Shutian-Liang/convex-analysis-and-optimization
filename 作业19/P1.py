import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
np.random.seed(240620)

# Generate some data
n = 100
a = np.random.rand(n)
b = np.random.rand(n)
x0 = np.random.randn(100)
y0 = 2*np.random.randn(100)
lam = np.zeros(n)
initial_guess = np.concatenate((x0, y0))

# 定义目标函数
def objective(var):
    x = var[:n]
    y = var[n:]
    fx = np.linalg.norm(x - a, 2) + np.linalg.norm(y - b, 1)
    return fx

# 定义约束条件
def constraint(var):
    x = var[:n]
    y = var[n:]
    return x - y  # 这里是一个示例，您需要根据实际情况定义约束条件

# 定义约束条件类型
constraint_type = {'type': 'eq', 'fun': constraint}

# 求解最优值
res = minimize(objective, initial_guess, constraints=constraint_type)
fstar = res.fun
print(fstar)

class ADMM:
    def __init__(self, x0, y0, lamb,a, b, tau, beta,fstar, method='ADMM',eta1=1e-8,eta2=1e-8):
        self.a = a
        self.b = b
        self.beta = beta
        self.ys = []
        self.x = x0
        self.y = y0
        self.lamb = lamb
        self.tau = tau
        self.eta1 = eta1
        self.eta2 = eta2
        self.fstar = fstar
        self.method = method    
    
    def f(self, x, y):
        return np.linalg.norm(x-self.a, 2) + np.linalg.norm(y-self.b, 1)

    def update_x(self):
        d = self.y - (self.lamb)/self.beta - self.a
        dist = np.linalg.norm(d, 2)
        self.x = self.a + (dist-self.beta)*d/dist

    def update_y(self):
        b = self.x + self.lamb / self.beta - self.b
        self.y = np.sign(b) * np.maximum(np.abs(b) - 1 / self.beta, 0) + self.b

    def update_lamb(self):
        self.lamb += self.tau*self.beta*(self.x - self.y)
    
    def iteration(self):
        while (self.f(self.x, self.y) - self.fstar> self.eta1):
            self.ys.append(self.f(self.x, self.y))
            self.update_x()
            self.update_y()
            self.update_lamb()
            if len(self.ys) > 5000:
                break
    
    def plot(self):
        iterations = len(self.ys)
        x = np.arange(iterations)
        plt.plot(x,np.log(np.array(self.ys)-self.fstar),'-o')
        plt.ylabel(r'$log(f(x)-f^*)$')
        plt.xlabel('Iterations')
        plt.title(f'{self.method} method tau={self.tau} beta={self.beta}')   
        plt.savefig(f'{self.method}_tau={self.tau}_beta={self.beta}.png')
        plt.show()

if __name__ == '__main__':
    taus = [0.5,1,1.5]
    betas = [0.001,0.01,0.1]
    for i in range(len(taus)):
        for j in range(len(betas)):
            tau = taus[i]
            beta = betas[j]
            admm = ADMM(x0= x0.copy(), y0= y0.copy(), lamb = lam.copy(),a=a.copy(), b=b.copy(), tau = tau,beta = beta, fstar=fstar)
            admm.iteration()
            admm.plot()
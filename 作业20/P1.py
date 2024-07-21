import numpy as np
import matplotlib.pyplot as plt

# generate data
np.random.seed(202406)
m = 200
n = 300
x0 = 2+np.random.randn(200, 300)
y = np.random.randint(0, 2, size=300) * 2 - 1
w = np.array([0.0]*200)

class GD:
    def __init__(self, method,m, n, x0, y, w0, alpha=0.2, beta=0.5):
        self.method = method
        self.m = m  # m=1000,n=10000,X(m,n)
        self.n = n
        self.x = x0
        self.y = y
        self.w = w0
        self.alpha = alpha
        self.beta = beta
        self.ys = []
        self.eta = 1e-3

    # give me the logistic function
    def f(self, w):
        return np.sum(np.log(1 + np.exp(-(w @ self.x) * self.y))) / self.n

    # compute the gradient of the logistic function
    def df(self, w):
        return self.x @ (-self.y * (1 - 1 / (np.exp(-(w.T @ self.x) * self.y) + 1))) / self.n

    def backtracking(self, x, direction):
        t = 1.0
        df = self.df(x)
        while self.f(x + t * direction) > self.f(x) + self.alpha * t * np.dot(df, direction):
            t *= self.beta
        return t

    def iteration(self):
        while np.linalg.norm(self.df(self.w),2) > self.eta:
            d = -self.df(self.w)
            t = self.backtracking(self.w, d)
            self.w += t * d
            self.ys.append(self.f(self.w))
            if len(self.ys) > 1000:
                break
    
    def plot(self):
        iterations = len(self.ys)
        x = np.arange(iterations)
        plt.plot(x,np.array(self.ys),'-o')
        plt.ylabel(r'$log(f(x))$')
        plt.xlabel('Iterations')
        plt.title(f'{self.method} method')   
        plt.savefig(f"{self.method}.png", bbox_inches='tight')
        plt.show()  

gd = GD('GD', m=m, n=n, x0=x0.copy(), y=y.copy(), w0=w.copy())
gd.iteration()
gd.plot()

class pLADMPSAP:
    def __init__(self, method,m, n, x0, y, w0, beta=0.5, eta=1e-3):
        self.method = method
        self.m = m  
        self.n = n
        self.X = x0
        self.y = y
        self.w0 = w0
        self.W = None
        self.t0 = None
        self.T = None
        self.tau0 = None
        self.TAU = None
        self.eta0 = eta
        self.ETA = None
        self.ys = []
        self.beta = beta
        self.LAM = None

    def setParameter(self):
        self.T = np.linalg.norm(self.X, ord=2, axis=0) / (self.n * 4) + 1.0
        self.t0 = 1.0
        self.eta0 = (self.n)**2.5 + 1.0
        self.ETA = (self.n+5.0)*np.ones(self.n)
        self.tau0 = self.t0 + self.beta * self.eta0
        self.TAU = self.T + self.beta * self.ETA
        self.W = np.zeros((self.m, self.n))
        for i in range(self.n):
            self.W[:, i] = self.w0.copy()
        self.LAM = np.zeros((self.m, self.n))

    def f(self, w):
        return np.sum(np.log(1 + np.exp(-(w @ self.X) * self.y))) / self.n

    def df(self, w):
        return self.X @ (-self.y * (1 - 1 / (np.exp(-(w.T @ self.X) * self.y) + 1))) / self.n

    def df_i(self, w, i):
        y = self.y[i]
        return (1 - 1 / (np.exp(-(y*self.X[:, i]) @ w) + 1)) * (-y * self.X[:, i])

    def iteration(self):
        self.setParameter()
        while True:
            dLAM = self.LAM.copy()
            w0 = self.w0.copy()
            dw0 = (1 / self.tau0) * np.sum(self.LAM, axis=1)
            self.w0 += dw0
            y = self.f(self.w0)
            self.ys.append(y)
            for i in range(self.n):
                self.W[:, i] -= (1 / self.TAU[i]) * (self.LAM[:, i] + self.df_i(self.W[:, i], i))
                dLAM[:, i] = self.W[:, i] - w0
            self.LAM += self.beta * dLAM
            
            if len(self.ys) > 1000:
                break
    
    def plot(self):
        iterations = len(self.ys)
        x = np.arange(iterations)
        plt.plot(x,np.array(self.ys),'-o')
        plt.ylabel(r'$log(f(x))$')
        plt.xlabel('Iterations')
        plt.title(f'{self.method} method')   
        plt.savefig(f"{self.method}.png", bbox_inches='tight')
        plt.show()  

p = pLADMPSAP('pLADMPSAP', m=m, n=n, x0=x0.copy(), y=y.copy(), w0=w.copy())
p.iteration()
p.plot()
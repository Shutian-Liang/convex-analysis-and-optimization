import numpy as np
import matplotlib.pyplot as plt

np.random.seed(202406)
X = np.random.randn(1000, 10000)
y = np.random.randint(0, 2, size=10000) * 2 - 1
w = np.array([0.0]*1000)

class Stochastic:
    def __init__(self, m, n, X, y, w0, eta=0.01):
        self.m = m  
        self.n = n
        self.X = X
        self.y = y
        self.w = w0
        self.eta = eta
        self.ys = []

    def f(self, w):
        return np.sum(np.log(1 + np.exp(-(w @ self.X) * self.y))) / self.n

    def df(self, w):
        return self.X @ (-self.y * (1 - 1 / (np.exp(-(w.T @ self.X) * self.y) + 1))) / self.n

    def df_i(self, w, i):
        x_i = self.X[:, i]
        return (-self.y[i] * (1 - 1 / (np.exp(-(w.T @ x_i) * self.y[i]) + 1))) * x_i / self.n

    def df_j(self, w, j):
        z = -(w @ self.X) * self.y
        g = -self.y * (1 - 1 / (np.exp(z) + 1))
        return self.X[j] @ g / self.n
    
    def df_xi(self, w, i):
        return self.X[i] @ (-self.y * (1 - 1 / (np.exp(-(w.T @ self.X) * self.y) + 1))) / self.n
    
    def SGD_step(self, i):
        self.w -= self.df_i(self.w, i) * self.eta
        self.ys.append(self.f(self.w))

    def SGD_iteration(self):
        while True:
            i = np.random.randint(0, self.n-1)
            self.SGD_step(i)
            if len(self.ys) >= 1000:
                break

    def Momentum_step(self, i):
        self.b = self.b*self.gamma + self.eta * self.df_i(self.w, i)
        self.w -= self.b
        self.ys.append(self.f(self.w))

    def Momentum_iteration(self, gamma):
        self.gamma = gamma
        self.b = np.zeros(1000)
        while True:
            i = np.random.randint(0, self.n-1)
            self.Momentum_step(i)
            if len(self.ys) >= 1000:
                break
    
    def NAG_step(self, i):
        d = self.df_i(self.w - self.eta * self.b, i)
        self.b = self.b*self.gamma + self.eta * d
        self.w -= self.b
        self.ys.append(self.f(self.w))

    def NAG_iteration(self, gamma):
        self.gamma = gamma
        self.b = np.zeros(1000)
        while True:
            i = np.random.randint(0, self.n-1)
            self.NAG_step(i)
            if len(self.ys) >= 1000:
                break

    def Adagrad_step(self, i):
        d = self.df_i(self.w, i)
        self.G[i] += np.dot(d, d)
        self.w -= self.eta_adagrad / np.sqrt(1e-8 + self.G[i]) * d
        self.ys.append(self.f(self.w))

    def Adagrad_iteration(self, eta):
        self.G = np.zeros(10000)
        self.eta_adagrad = eta
        while True:
            i = np.random.randint(0, self.n-1)
            self.Adagrad_step(i)
            if len(self.ys) >= 1000:
                break

    def Adadelta_iteration(self, gamma):
        self.gamma = gamma
        self.Eg = np.zeros_like(self.w)
        self.dw = np.zeros_like(self.w)
        self.dwsq = np.zeros_like(self.w)
        while True:
            i = np.random.randint(0, self.n-1)
            self.Adadelta_step(i)
            if len(self.ys) >= 1000:
                break

    def Adadelta_step(self, i):
        d = self.df_i(self.w, i)
        self.Eg = self.gamma*self.Eg+(1-self.gamma)*d**2
        dw = np.sqrt(self.dwsq+1e-8) / np.sqrt(self.Eg+1e-8)*d
        self.dwsq = self.gamma*self.dwsq + (1-self.gamma)*dw ** 2
        self.w -= dw
        self.ys.append(self.f(self.w))

    def Adam_iteration(self):
        self.eta = 0.00001
        self.beta1 = 0.99
        self.eps = 1e-5
        self.beta2 = 0.001
        self.m = np.zeros_like(self.w)
        self.v = np.zeros_like(self.w)
        while True:
            i = np.random.randint(0, self.n-1)
            self.Adam_step(i)
            if len(self.ys) >= 1000:
                break

    def Adam_step(self, i):
        d = self.df_i(self.w, i)
        self.m = self.beta1 * self.m + (1 - self.beta1) * d
        self.v = self.beta2 * self.v + (1 - self.beta2) * d ** 2
        m_hat = self.m / (1 - self.beta1 ** (1+len(self.ws)))
        v_hat = self.v / (1 - self.beta2 ** (1+len(self.ws)))
        self.w -= self.eta / (np.sqrt(v_hat) + self.eps) * m_hat
        self.ys.append(self.f(self.w))

    def Adan_iteration(self):
        self.eta = 0.0001
        self.eps = 1e-8
        self.beta1 = 0.02
        self.beta2 = 0.1
        self.beta3 = 0.01
        self.lam = 0.05
        self.mk = np.zeros_like(self.w)
        self.vk = np.zeros_like(self.w)
        self.uk = np.zeros_like(self.w)
        self.nk = np.zeros_like(self.w)
        self.gk = np.zeros_like(self.w)
        while True:
            i = np.random.randint(0, self.n-1)
            self.Adan_step(i)
            if len(self.ys) >= 1000:
                break

    def Adan_step(self, i):
        g = self.df_i(self.w, i)
        self.vk = (1-self.beta2)*self.vk+self.beta2*(g-self.gk)
        self.mk = (1-self.beta1)*self.mk+self.beta1*g
        self.uk = self.mk+(1-self.beta2)*self.vk
        self.nk = (1-self.beta3)*self.nk+self.beta3*(g+(1-self.beta2)*(g-self.gk))**2
        self.gk = g
        self.w = (1/(1+self.lam*self.eta))*(self.w-self.eta*self.uk/np.sqrt(self.nk+self.eps))
        self.ys.append(self.f(self.w))

methods = ["SGD", "Momentum", "NAG", "Adagrad", "Adadelta", "Adam", "Adan"]
sgd = Stochastic(1000, 10000, X.copy(), y.copy(), w.copy())
sgd.SGD_iteration()
momentum = Stochastic(1000, 10000, X.copy(), y.copy(), w.copy())
momentum.Momentum_iteration(0.9)
nag = Stochastic(1000, 10000, X.copy(), y.copy(), w.copy())
nag.NAG_iteration(0.9)
adagrad = Stochastic(1000, 10000, X.copy(), y.copy(), w.copy())
adagrad.Adagrad_iteration(0.01)
adadelta = Stochastic(1000, 10000, X.copy(), y.copy(), w.copy())
adadelta.Adadelta_iteration(0.9)
adam = Stochastic(1000, 10000, X.copy(), y.copy(), w.copy())
adam.Adam_iteration()
adan = Stochastic(1000, 10000, X.copy(), y.copy(), w.copy())
adan.Adan_iteration()

optimizers = [sgd, momentum, nag, adagrad, adadelta, adam, adan]
for i in range(len(optimizers)):
    opt = optimizers[i]
    y = opt.ys
    x = np.arange(len(y))
    plt.plot(x, y, label=methods[i])
plt.ylabel("loss")
plt.xlabel("iteration")
plt.legend()
plt.show()

#### rcd
class Stochastic_rcd(Stochastic):
    def __init__(self, m, n, X, y, w0, gamma=1,eta=0.01):
        super().__init__(m, n, X, y, w0, eta)
        self.rowbeta = np.linalg.norm(X, axis=1) ** 2 / 4 / self.n
        self.gamma = gamma
        self.pbeta = self.rowbeta ** self.gamma / np.sum(self.rowbeta ** self.gamma)
    
    def RCD_step(self, j):
        self.w[j] -= self.df_j(self.w, j)/self.rowbeta[j]
        self.ys.append(self.f(self.w))

    def RCD_iteration(self):
        while True:
            j = np.random.choice(self.m, p=self.pbeta)
            self.RCD_step(j)
            if len(self.ys) >= 1000:
                break

np.random.seed(202406)
X = np.random.randn(10000, 1000)
y = np.random.randint(0, 2, size=1000) * 2 - 1
w = np.array([0.0]*10000)
rcd = Stochastic_rcd(10000, 1000, X.copy(), y.copy(), w.copy())
rcd.RCD_iteration()
rcd_01 = Stochastic_rcd(10000, 1000, X.copy(), y.copy(), w.copy(),gamma=0.1) 
rcd_01.RCD_iteration()
rcd_05 = Stochastic_rcd(10000, 1000, X.copy(), y.copy(), w.copy(),gamma=0.5) 
rcd_05.RCD_iteration()

rcd_09 = Stochastic_rcd(10000, 1000, X.copy(), y.copy(), w.copy(),gamma=0.9) 
rcd_09.RCD_iteration()
optimizers = []
optimizers.append(rcd)
optimizers.append(rcd_01)
optimizers.append(rcd_05)
optimizers.append(rcd_09)

methods= ["RCD","RCD_0.1","RCD_0.5", "RCD_0.9"]
for i in range(len(optimizers)):
    opt = optimizers[i]
    y = opt.ys
    x = np.arange(len(y))
    plt.plot(x, y, label=methods[i])
plt.ylabel("loss")
plt.xlabel("iteration")
plt.legend()
plt.show()
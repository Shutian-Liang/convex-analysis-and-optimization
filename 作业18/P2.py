# 引用numpy库和matplotlib库
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2024)
D = np.random.rand(200,300)
y = np.random.rand(200)

from scipy.optimize import minimize
# 定义目标函数
def objective(x):
    fx = np.linalg.norm(D @ x - y, 2) ** 2
    return fx

def constraint(x):
    return np.linalg.norm(x, np.inf) 

x0 = np.random.randn(300)
# 定义约束条件类型
constraint_type = {'type': 'ineq', 'fun': constraint}

# 求解最优值
res = minimize(objective, x0, constraints=constraint_type)
fstar = res.fun
print(fstar)

class FW:
    def __init__(self,method,y,D,x0,fstar,eta=1e-3) -> None:
        self.y = y
        self.D = D
        self.x = x0
        self.eta = eta
        self.ys = []
        self.fstar = fstar
        self.method = method
    
    def f(self, x: np.array):
        ydx = self.y-self.D@x
        return np.dot(ydx, ydx)

    def df(self, x: np.array):
        return 2*self.D.T@(-self.y+self.D@x)
    
    def iteration(self):
        while np.linalg.norm(self.df(self.x),2) > self.eta:
            delta = self.df(self.x)
            sk = -np.sign(delta)
            gamma = 2/(len(self.ys)+2)
            self.x = self.x + gamma*(sk-self.x)
            self.ys.append(self.f(self.x))
            if (self.f(self.x)-self.fstar) < self.eta:
                break
            if len(self.ys) > 200000:
                break
    
    def plot(self):
        iterations = len(self.ys)
        x = np.arange(iterations)
        plt.plot(np.log(np.array(self.ys)),'-')
        plt.ylabel(r'$log(f(x)-f^*)$')
        plt.xlabel('Iterations')
        plt.title(f'{self.method} method')   
        plt.show()  

fw = FW('fw',y=y.copy(),D=D.copy(),x0=x0.copy(),fstar=fstar)
fw.iteration()
fw.plot()

class PGD(FW):
    def __init__(self, method, y, D, x0, fstar, alpha=0.2,beta=0.5,eta=0.01) -> None:
        super().__init__(method, y, D, x0, fstar, eta)
        self.alpha = alpha
        self.beta = beta

    def backtracking(self,x,direction):
        t = 1
        dx = self.df(x)
        while self.f(x + t*direction) > self.f(x) + self.alpha*t*np.dot(dx,direction):
            t *= self.beta
        return t
    
    def iteration(self):
        while np.linalg.norm(self.df(self.x),2) > self.eta:
            direction = -self.df(self.x)
            t = self.backtracking(self.x,direction)
            self.x = self.x + t*direction

            # 投影到约束集合
            x1 = np.minimum(np.ones_like(self.x), self.x)
            x2 = np.maximum(-np.ones_like(x1), x1)
            self.x = x2
            self.ys.append(self.f(self.x))
            if (self.f(self.x)-self.fstar) < self.eta:
                break

pgd = PGD('pgd',y.copy(),D.copy(),x0.copy(),fstar)
pgd.iteration()
pgd.plot()
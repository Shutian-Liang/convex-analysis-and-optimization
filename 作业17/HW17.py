import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#scipy calculating the minimum of a scalar function
np.random.seed(202406)
p,n = 100, 500
A = 10*np.random.rand(p,n)+5
b = 10*np.random.rand(p)+5

# 定义目标函数
def objective(x):
    fx = np.log(np.sum(np.exp(x)))
    return fx

def constraint(x):
    return A @ x - b

x0 = 2*np.random.randn(n)
# 定义约束条件类型
constraint_type = {'type': 'eq', 'fun': constraint}

# 求解最优值
res = minimize(objective, x0, constraints=constraint_type)
fstar = res.fun
print(fstar)

class optimizer:
    def __init__(self,name,x0,A,b,alpha=0.3,beta=0.5,eta = 1e-4) -> None:
        self.x = x0
        self.A = A
        self.b = b
        self.name = name
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.P = np.eye(n) - A.T @ np.linalg.inv(A @ A.T) @ A
        self.ys = []
    
    def grad(self,x) -> float:
        return np.exp(x)/np.sum(np.exp(x))
    
    def gradgrad(self,x) -> float:
        exp_sum = np.sum(np.exp(x))
        dfdf = (np.exp(x)*exp_sum - np.exp(x)**2)/(exp_sum**2)
        # 把dfdf放在对角线上
        dfdf = np.diag(dfdf)
        return dfdf
    
    def value(self,x) -> float:
        return np.log(np.sum(np.exp(x)))
    
    def backtracking(self,x,direction) -> float: 
        t = 1
        while self.value(x+t*direction) > self.value(x) + self.alpha*t*np.dot(self.grad(x),direction):
            t *= self.beta
        return t
    
    def gradientnorm(self,dx) -> float:
        factor = np.sqrt(np.sum(dx**2))
        return factor 

    def plot(self):
        iterations = len(self.ys)
        x = np.arange(iterations)
        plt.plot(x,np.log(np.array(self.ys)-fstar),'-')
        plt.ylabel(r'$log(f(x)-f^*)$')
        plt.xlabel('Iterations')
        plt.title(f'{self.name} method')   
        plt.savefig(f"{self.name}.png", bbox_inches='tight')
        plt.show()  

class projection(optimizer):
    def __init__(self,name,x0,A,b,alpha=0.8,beta=0.6,eta = 1e-3):
        super().__init__('projection',x0,A,b,alpha=0.3,beta=0.5,eta = 1e-3)

    def proj(self,x) -> np.array:
        return -self.P @ self.grad(x)
    
    def iteration(self):
        while self.gradientnorm(self.proj(self.x)) > self.eta:
            t = self.backtracking(self.x,self.proj(self.x))
            self.x = self.x + t*self.proj(self.x)
            self.ys.append(self.value(self.x))
            if len(self.ys) > 1000:
                break

p = projection('projection',x0.copy(),A,b)
p.iteration()
p.plot()

class newton(optimizer):
    def __init__(self,name,x0,A,b,alpha=0.2,beta=0.5,eta = 1e-3):
        super().__init__('newton',x0,A,b,alpha=0.3,beta=0.5,eta = 1e-3)
    
    def direction(self,x) -> np.array:
        df = self.grad(x)
        dfdf = self.gradgrad(x)
        Left_matrix_upper = np.concatenate([dfdf, self.A.T], axis=1)
        Left_matrix_lower = np.concatenate([self.A, np.zeros((self.A.shape[0],self.A.shape[0]))], axis=1)
        Left_matrix = np.concatenate([Left_matrix_upper, Left_matrix_lower], axis=0)

        Right_matrix = np.concatenate([-df, np.zeros(self.A.shape[0])], axis=0)
        dxw = np.linalg.solve(Left_matrix, Right_matrix)
        dx = dxw[:500]
        return dx
    
    def iteration(self):
        direction = self.direction(self.x)
        lambdax = (direction@self.gradgrad(self.x)@direction)**(0.5)
        while 0.5*lambdax**2 > self.eta:
            direction = self.direction(self.x)
            t = self.backtracking(self.x,direction)
            self.x = self.x + t*direction
            self.ys.append(self.value(self.x))
            lambdax = (direction@self.gradgrad(self.x)@direction)**(0.5)
            if len(self.ys) > 1000:
                break

new = newton('Dampled Newton',x0.copy(),A,b)
new.iteration()
new.plot()

class elimate(optimizer):
    def __init__(self,name,x0,A,b,alpha=0.2,beta=0.5,eta = 1e-5):
        super().__init__('elimate',x0,A,b,alpha=0.3,beta=0.5,eta = 1e-5)
        self.xhat = A.T @ np.linalg.inv(A @ A.T) @ b
    
    def value_old(self, x) -> float:
        return super().value(x)
    
    def value(self,x) -> float:
        return self.value_old(self.P @ x + self.xhat)
    
    def backtracking(self,x,direction) -> float: 
        t = 1
        while self.value(x+t*direction) > self.value(x) + self.alpha*t*np.dot(direction,direction):
            t *= self.beta
        return t
    
    def direction(self,x) -> np.array:
        return -self.P.T @ self.grad(self.P @ x + self.xhat)
    
    def iteration(self) -> np.array:
        print(abs(self.value(self.x)-fstar))
        while self.gradientnorm(self.direction(self.x)) > self.eta:
            direction = self.direction(self.x)
            t = self.backtracking(self.x,direction)
            self.x = self.x + t*direction
            self.ys.append(self.value(self.x))
            if len(self.ys) > 1000:
                break

x1 = np.zeros(n)
e = elimate('elimate',x0.copy(),A,b)
e.iteration()
e.plot()

class dual(optimizer):
    def __init__(self, name, x0, A, b,gamma,alpha=0.3, beta=0.5, eta=0.01) -> None:
        super().__init__(name, x0, A, b, alpha=0.3, beta=0.5, eta=0.01)
        self.lamb = np.ones_like(b)
        self.gamma = gamma
            
    def L(self, x):
        Axb = self.A@x-self.b
        return self.value(self.x) + np.dot(self.lamb, Axb) + 0.5*self.gamma*np.dot(Axb, Axb)

    def dL(self, x):
        return self.grad(x) + self.A.T@self.lamb + self.gamma * self.A.T@(self.A@x-b)

    def iteration(self): 
        while True:
            self.x = minimize(self.L, self.x, jac=self.dL).x
            self.lamb += self.gamma*(self.A@self.x - self.b)
            y = self.value(self.x)
            self.ys.append(y)
            if len(self.ys) > 1000:
                break
            if self.gradientnorm(self.grad(self.x)) < self.eta:
                break


x1 = np.zeros(n)
d = dual('dual',x0.copy(),A,b,gamma=0.1,alpha=0.3,beta=0.5,eta=1e-4)
d.iteration()
d.plot()
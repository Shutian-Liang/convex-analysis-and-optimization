import numpy as np
import matplotlib.pyplot as plt
np.random.seed(20240601)

# Generate some data
lamb = 1.0
D0 = np.random.randn(200, 300)
E0 = np.zeros((200, 300))
Z0 = np.zeros((300, 300))
LAM0 = np.zeros((201, 300))
class LADMAP:
    def __init__(self, method,n, p, lamb, D, Z0, E0, LAM, beta, beta_m, rho, eps1, eps2):
        self.n = n
        self.p = p
        self.lamb = lamb
        self.method = method
        self.D = D
        self.Z = Z0
        self.E = E0
        self.A1 = np.vstack((self.D, np.ones(D.shape[1])))
        self.A2 = np.vstack((np.eye(D.shape[0]), np.zeros(D.shape[0])))
        self.b = np.vstack((self.D, np.ones(D.shape[1])))
        self.eta1 = np.linalg.norm(self.A1, 2)**2 + 1
        self.eta2 = np.linalg.norm(self.A2, 2)**2 + 1

        self.beta = beta
        self.beta_m = beta_m
        self.rho = rho
        self.eps1 = eps1
        self.eps2 = eps2
        self.ys = []
        self.cs = []
        self.bs = []
        self.LAM = LAM
        self.flag1 = False
        self.flag2 = False

    # value function and prox
    def f1(self, Z: np.array):
        return np.linalg.norm(Z, "nuc")

    def f2(self, E: np.array):
        return self.lamb * np.linalg.norm(np.linalg.norm(E, 2, axis=0), 1)

    def f1_prox(self, Z, beta):
        U, sigma,VT = np.linalg.svd(Z)
        sigma = np.maximum(sigma - 1/beta, np.zeros_like(sigma))
        return U @ np.diag(sigma) @ VT

    def f2_prox(self, E, beta):
        Xv = np.linalg.norm(E,2, axis=0)
        P = np.zeros_like(E)
        for i in range(P.shape[1]):
            P[:, i] = max((1 - self.lamb / (beta * Xv[i])), 0) * E[:, i]
        return P

    def update_LAM(self):
        self.LAM += self.beta * (self.A1 @ self.Z + self.A2 @ self.E - self.b)

    def update_beta(self):
        rho = 1
        con2= self.beta * max(np.sqrt(self.eta1) * np.linalg.norm(self.dZ), np.sqrt(self.eta2) * np.linalg.norm(self.dE)) / np.linalg.norm(self.b)
        if con2< self.eps2:
            rho = self.rho
        self.beta = min(self.beta_m, rho*self.beta)

    def set_criterion(self):
        self.criteria_1 = False
        self.criteria_2 = False
        if np.linalg.norm(self.A1 @ self.Z + self.A2 @ self.E - self.b)/np.linalg.norm(self.b) < self.eps1:
            self.criteria_1 = True
        if self.beta * max(np.sqrt(self.eta1) * np.linalg.norm(self.dZ), np.sqrt(self.eta2) * np.linalg.norm(self.dE)) / np.linalg.norm(self.b) < self.eps2:
            self.criteria_2 = True

    def iteration(self):
        while True:
            y = self.f1(self.Z) + self.f2(self.E)
            self.ys.append(y)
            constraint = np.linalg.norm(self.A1@self.Z + self.A2@self.E - self.b)
            self.bs.append(self.beta)
            self.cs.append(constraint)
            
            # update z, e, lam, beta
            W = self.Z - (1 / (self.beta * self.eta1)) * self.A1.T @ (self.LAM + self.beta * (self.A1 @ self.Z + self.A2 @ self.E -self.b))
            Z = self.f1_prox(W, self.beta * self.eta1)

            # update E
            W = self.E - (1 / (self.beta * self.eta2)) * self.A2.T @ (self.LAM + self.beta * (self.A1 @ self.Z + self.A2 @ self.E -self.b))
            E = self.f2_prox(W, self.beta * self.eta2)
            self.dE = E-self.E
            self.dZ = Z-self.Z
            self.E = E.copy()
            self.Z = Z.copy()

            self.update_LAM()
            self.update_beta()
            self.set_criterion()
            if len(self.ys)%200 == 0:
                print(f'iteration {len(self.ys)} finished')
            if len(self.ys) > 1000:
                break
            if self.criteria_1 and self.criteria_2:
                return

    def plot(self):
        iterations = len(self.ys)
        x = np.arange(iterations)
        fig,axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].plot(x, self.ys, '-')
        axes[1].plot(x, self.bs, '-')
        axes[2].plot(x, self.cs, '-')
        axes[0].set_ylabel(r'$f(Z, E)$')
        axes[1].set_ylabel(r'$beta$')
        axes[2].set_ylabel(r'$constriant$')
        for ax in axes:
            ax.set_xlabel('Iterations')
            ax.set_title(f'{self.method} method')
        plt.subplots_adjust(wspace=0.5)
        plt.savefig(f'{self.method}.png', bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    ladmap = LADMAP(method='LADMAP',n=200, p=300, lamb=1.1, D=D0.copy(), Z0=Z0.copy(), E0=E0.copy(), LAM=LAM0.copy(), beta=1e-3, beta_m=100, rho=1.8, eps1=1e-3, eps2=1e-3)
    ladmap.iteration()
    ladmap.plot()
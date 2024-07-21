from P2 import LADMAP
from multiprocessing import Process, Manager
import numpy as np 
import matplotlib.pyplot as plt
np.random.seed(20240601)

# Generate some data
lamb = 1.0
D0 = np.random.randn(200, 300)
E0 = np.zeros((200, 300))
Z0 = np.zeros((300, 300))
LAM0 = np.zeros((201, 300))
class LADMPSAP(LADMAP):
    def __init__(self, method,n, p, lamb, D, Z0, E0, LAM, beta, beta_m, rho, eps1, eps2):
        super().__init__(method,n,p, lamb, D, Z0, E0, LAM, beta, beta_m, rho, eps1, eps2)
    
    def parallel_update(self):
        with Manager() as manager:
            dZ = manager.dict()
            dE = manager.dict()
            Zs = manager.dict()
            Es = manager.dict()

            p1 = Process(target=self.update_Z, args=(dZ,Zs))
            p2 = Process(target=self.update_E, args=(dE,Es))

            p1.start()
            p2.start()

            p1.join()
            p2.join()

            self.dZ = dZ['value']
            self.dE = dE['value']
            self.Z = Zs['value']
            self.E = Es['value']

    def update_Z(self, dZ, Zs):
        W = self.Z - (1 / (self.beta * self.eta1)) * self.A1.T @ (self.LAM + self.beta * (self.A1 @ self.Z + self.A2 @ self.E - self.b))
        Z = self.f1_prox(W, self.beta * self.eta1)
        self.dZ = Z-self.Z
        self.Z = Z.copy()
        dZ['value'] = self.dZ  # update the value in the shared dictionary
        Zs['value'] = self.Z

    def update_E(self, dE, Es):
        W = self.E - (1 / (self.beta * self.eta2)) * self.A2.T @ (self.LAM + self.beta * (self.A1 @ self.Z + self.A2 @ self.E - self.b))
        E = self.f2_prox(W, self.beta * self.eta2)
        self.dE = E-self.E
        self.E = E.copy()
        dE['value'] = self.dE  # update the value in the shared dictionary
        Es['value'] = self.E

    def iteration(self):
        while True:
            y = self.f1(self.Z) + self.f2(self.E)
            self.ys.append(y)
            constraint = np.linalg.norm(self.A1@self.Z + self.A2@self.E - self.b)
            self.bs.append(self.beta)
            self.cs.append(constraint)
            
            # update Z and E
            self.parallel_update()

            self.update_LAM()
            self.update_beta()
            self.set_criterion()
            print(len(self.ys))
            if len(self.ys)%200 == 0:
                print(f'iteration {len(self.ys)} finished')
            if self.criteria_1 and self.criteria_2:
                return
            if len(self.ys) > 100:
                break

if __name__ == '__main__':
    ladmpsap = LADMPSAP(method='LADMPSAP',n=200, p=300, lamb=1.1, D=D0.copy(), Z0=Z0.copy(), E0=E0.copy(), LAM=LAM0.copy(), beta=1e-3, beta_m=100, rho=1.8, eps1=1e-3, eps2=1e-3)
    ladmpsap.iteration()
    ladmpsap.plot()
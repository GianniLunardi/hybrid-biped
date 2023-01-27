import numpy as np

class HybridLipm:

    def __init__(self, dt, tau0, omega, r_bar, v_bar, T, L, K):
        self.dt = dt
        self.tau = tau0
        self.omega = omega
        self.r_bar = r_bar
        self.v_bar = v_bar
        self.T = T
        self.L = L
        self.K = K
        self.x0 = np.array([-r_bar, v_bar])
        self.A_d = np.array([[np.cosh(omega * dt), (1 / omega) * np.sinh(omega * dt)],
                             [omega * np.sinh(omega * dt), np.cosh(omega * dt)]])
        self.B_d = np.array([1 - np.cosh(omega * dt), - omega * np.sinh(omega * dt)])

    def flow(self, x, u):
        self.tau += self.dt
        return self.A_d.dot(x) + self.B_d * u

    def jump(self, x):
        self.tau = self.tau - self.T
        return x - np.array([2 * self.r_bar, 0])

    def referenceWithTimer(self):
        expA = np.array([[np.cosh(self.omega * self.tau), (1 / self.omega) * np.sinh(self.omega * self.tau)],
                         [self.omega * np.sinh(self.omega * self.tau), np.cosh(self.omega * self.tau)]])
        return expA.dot(self.x0)

    def linearSat(self, u):
        return np.min([np.max([u, -self.L]), self.L])

    def saturatedFb(self, eps):
        return self.linearSat(self.K.dot(eps))

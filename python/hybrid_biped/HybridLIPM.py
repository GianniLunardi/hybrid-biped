import numpy as np
from hybrid_biped.Params import BipedParams

class HybridLipm:

    def __init__(self, dt, tau0, params = None):
        if params is not None:
            self.params = params
        else:
            self.params = BipedParams('.')
        self.dt = dt
        self.tau = tau0
        self.omega = params.omega
        self.r_bar = params.r_bar
        self.v_bar = params.v_bar
        self.T = params.T
        self.K = params.K
        self.x_sat = params.x_sat
        self.x_ref_0 = np.array([-params.r_bar, params.v_bar])
        self.A_d = np.array([[np.cosh(self.omega * dt), (1 / self.omega) * np.sinh(self.omega * dt)],
                             [self.omega * np.sinh(self.omega * dt), np.cosh(self.omega * dt)]])
        self.B_d = np.array([1 - np.cosh(self.omega * dt), - self.omega * np.sinh(self.omega * dt)])

    def flow(self, x, u):
        self.tau += self.dt
        return self.A_d.dot(x) + self.B_d * u

    def jump(self, x):
        self.tau = self.tau - self.T
        return x - np.array([2 * self.r_bar, 0])

    def referenceWithTimer(self):
        expA = np.array([[np.cosh(self.omega * self.tau), (1 / self.omega) * np.sinh(self.omega * self.tau)],
                         [self.omega * np.sinh(self.omega * self.tau), np.cosh(self.omega * self.tau)]])
        return expA.dot(self.x_ref_0)

    def linearSat(self, u):
        return np.min([np.max([u, -self.x_sat]), self.x_sat])

    def saturatedFb(self, eps):
        return self.linearSat(self.K.dot(eps))


def rolloutLipmDynamics(x0, tau0, dt, params):
    """ Roll-out of the Hybrid LIPM dynamics """
    i = 0
    hs_lipm = HybridLipm(dt, tau0)
    x_hb = x0

    while x_hb[0] < params.r_bar:
        x_hb_ref = hs_lipm.referenceWithTimer()
        eps = x_hb - x_hb_ref
        u = hs_lipm.saturatedFb(eps)
        x_hb = hs_lipm.flow(x_hb, u)
        i += 1
    return i
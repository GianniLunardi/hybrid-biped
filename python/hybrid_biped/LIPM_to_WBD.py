import numpy as np
import lmpc_walking.second_order as lw
from hybrid_biped import rolloutLipmDynamics


class LipmToWbc:

    def __init__(self, params):
        self.dt = params.dt
        self.g = params.omega
        self.h_com = params.h_com
        self.omega = params.omega
        self.r_bar = params.r_bar
        self.T = params.T
        self.foot_h = params.step_height
        self.params = params

    def computeCoMAcceleration(self, c_p, u):
        return self.g/self.h_com * (c_p - u)

    def fifthOrderPolynomial(self, t, xy_init, xy_fin):
        a = xy_init
        d = -10 * (xy_init - xy_fin) / self.T**3
        e = 15 * (xy_init - xy_fin) / self.T**4
        f = -6 * (xy_init - xy_fin) / self.T**5
        poly = a + d*t**3 + e*t**4 + f*t**5
        poly_dot = (3*d*t**2 + 4*e*t**3 + 5*f*t**4) * self.alpha
        poly_ddot = (6*d*t + 12*e*t**2 + 20*f*t**3) * self.alpha**2
        return np.array([poly, poly_dot, poly_ddot]).T

    def sixthOrderPolynomial(self, t, z_max):
        d = 64 * z_max/ self.T**3
        e = -192 * z_max / self.T**4
        f = 192 * z_max / self.T**5
        g = -64 * z_max / self.T**6
        poly = d*t**3 + e*t**4 + f*t**5 + g*t**6
        poly_dot = (3*d*t**2 + 4*e*t**3 + 5*f*t**4 + 6*g*t**5) * self.alpha
        poly_ddot = (6*d*t + 12*e*t**2 + 20*f*t**3 + 30*g*t**4) * self.alpha**2
        return np.array([poly, poly_dot, poly_ddot]).T

    def footTrajectoryFromHS(self, foot_init, foot_fin, foot_h = None):
        if foot_h is None:
            foot_h = self.foot_h
        foot_pos = np.zeros((3,3))
        self.alpha = self.T /(self.t + self.t_left)
        foot_pos[:2,:] = self.fifthOrderPolynomial(self.alpha * self.t, foot_init, foot_fin)
        foot_pos[2,:] = self.sixthOrderPolynomial(self.alpha * self.t, foot_h)
        return foot_pos

    def integrateCoMLateralState(self, y, u_y, j):
        A_d, B_d = lw.discrete_LIP_dynamics((j+1) * self.dt, self.omega)
        return A_d.dot(y) + B_d * u_y

    def set_time(self, t_hs, x, tau):
        self.t = t_hs
        self.t_left = self.dt * rolloutLipmDynamics(x, tau, self.dt, self.params)

    def set_time_offline(self, t_hs, t_pred):
        self.t = t_hs
        self.t_left = t_pred - t_hs
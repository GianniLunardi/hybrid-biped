import numpy as np
import scipy.io

class BipedParams:

    def __init__(self, mat_path, filename = None):
        self.h_com = 0.58
        self.g = 9.81
        self.omega = np.sqrt(self.g / self.h_com)

        self.nb_stpes_des = 5
        self.T = 1.2
        self.dt = 1e-3
        self.dt_mpc = 3 * 1e-2
        self.ratio_mpc = int(self.dt_mpc / self.dt)

        self.r_bar = 0.15
        self.v_bar = -self.omega * self.r_bar * (1 + np.exp(self.omega * self.T)) / (1 - np.exp(self.omega * self.T))

        self.foot_length = 0.15
        self.foot_width = 0.1
        self.x_sat = self.foot_length / 2
        self.y_sat = self.foot_width / 2

        data = scipy.io.loadmat(mat_path + '/data.mat')
        self.K = data['K']

        # ICs
        self.x0 = data['x0'][:-1].T
        self.tau0 = data['x0'][-1].item()
        self.foot_step_0 = np.array([0., -0.05])
        # TODO use only one among x0 and x_hat_0, it creates confusion
        self.x_hat_0 = np.array([0.0, 0.0])
        self.y_hat_0 = np.array([-0.04, 0.0])

        self.step_length = self.r_bar * 2
        self.step_width = 2 * np.absolute(self.y_hat_0[0])
        self.step_height = 0.05

        # Weights
        self.alpha = 1e2
        self.beta = 1e2
        self.gamma = 1e2
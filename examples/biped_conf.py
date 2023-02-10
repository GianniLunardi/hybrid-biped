import numpy as np
import scipy.io

dt = 1e-3
h_com = 0.58
g = 9.81
omega = np.sqrt(g/h_com)
N_footsteps = 4

T = 1.2

dt_mpc = 1e-2
horizon = 18

r_bar = 0.15
v_bar = -omega*r_bar*(1 + np.exp(omega*T))/(1 - np.exp(omega*T))

foot_length = 0.15
foot_width = 0.1
x_sat = foot_length/2
y_sat = foot_width/2

step_length = r_bar * 2
nb_des_steps = 4
nb_plan_steps = nb_des_steps + 2

data = scipy.io.loadmat('../data/data.mat')
K = data['K']

# ICs
x0 = data['x0'][:-1].T
tau0 = data['x0'][-1].item()
foot_step_0 = np.array([0., -0.09])

x_hat_0 = np.array([0.0, 0.0])
y_hat_0 = np.array([-0.09, 0.0])
step_width = 2 * np.absolute(y_hat_0[0])
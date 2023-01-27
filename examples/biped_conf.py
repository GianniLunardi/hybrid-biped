import numpy as np
import scipy.io

dt = 1e-3
h_com = 0.58
g = 9.81
omega = np.sqrt(g/h_com)
N_footsteps = 2

T = 1.2
r_bar = 0.15
v_bar = -omega*r_bar*(1 + np.exp(omega*T))/(1 - np.exp(omega*T))
L = 0.075
data = scipy.io.loadmat('data.mat')
K = data['K']

# ICs
x0 = data['x0'][:-1].T
tau0 = data['x0'][-1].item()
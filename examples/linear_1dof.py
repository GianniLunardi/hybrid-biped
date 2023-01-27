import numpy as np
import matplotlib.pyplot as plt
from hybrid_biped import HybridLipm
import biped_conf as conf


hs_lipm = HybridLipm(conf.dt, conf.tau0, conf.omega, conf.r_bar,
                     conf.v_bar, conf.T, conf.L, conf.K)

n_steps = int(conf.T/conf.dt) * conf.N_footsteps

x = np.zeros((n_steps+1, 2))
x_r = np.zeros((n_steps, 2))
eps = np.zeros((n_steps, 2))
u = np.zeros(n_steps)
x[0,:] = conf.x0

for i in range(n_steps):
    if x[i,0] >= conf.r_bar:
        x[i,:] = hs_lipm.jump(x[i,:])
    x_r[i,:] = hs_lipm.referenceWithTimer()
    eps[i,:] = x[i,:] - x_r[i,:]
    u[i] = hs_lipm.saturatedFb(eps[i,:])
    x[i+1,:] = hs_lipm.flow(x[i,:], u[i])

t = np.arange(0, n_steps) * conf.dt
fig, ax = plt.subplots(2, 1)
ax[0].plot(t, x[:-1,0], color='blue', label='x')
ax[0].plot(t, x_r[:,0], color='red', label='x_r')
ax[0].legend()
ax[1].plot(t, x[:-1,1], color='blue', label='x')
ax[1].plot(t, x_r[:,1], color='red', label='x_r')
ax[1].legend()
plt.show()
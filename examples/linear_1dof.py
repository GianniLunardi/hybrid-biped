import numpy as np
import matplotlib.pyplot as plt
from hybrid_biped import HybridLipm, BipedParams


def integrateCoMAcc(p, v, a, dt):
    p += v * dt + 0.5 * a * dt ** 2
    v += a * dt
    return p, v

params = BipedParams()
hs_lipm = HybridLipm(params.dt, params.tau_hat_0, params)

n_steps = int(params.T/params.dt) * params.nb_steps

x = np.zeros((n_steps+1, 2))
x_r = np.zeros((n_steps, 2))
eps = np.zeros((n_steps, 2))
x_ddot = np.zeros(n_steps)
x_inter = np.zeros((n_steps+1, 2))
u = np.zeros(n_steps)
x[0,:] = params.x_hat_0
x_inter[0,:] = params.x_hat_0

for i in range(n_steps):
    if x[i,0] >= params.r_bar:
        x[i,:] = hs_lipm.jump(x[i,:])
    x_r[i,:] = hs_lipm.referenceWithTimer()
    eps[i,:] = x[i,:] - x_r[i,:]
    u[i] = hs_lipm.saturatedFb(eps[i,:])
    x[i+1,:] = hs_lipm.flow(x[i,:], u[i])
    x_ddot[i] = params.g / params.h_com * (x[i,0] - u[i])

    # Compute the integration
    x_inter[i+1,0], x_inter[i+1,1] = integrateCoMAcc(x_inter[i,0], x_inter[i,1], x_ddot[i], params.dt)

t = np.arange(0, n_steps) * params.dt
fig, ax = plt.subplots(3, 1)
ax[0].plot(t, x[:-1,0], color='blue', label='$x$')
ax[0].plot(t, x_r[:,0], color='red', label='$x_r$')
ax[0].plot(t, x_inter[:-1,0], color='lightblue', linestyle='--', label='$x_{inter}$')
ax[0].legend()
ax[1].plot(t, x[:-1,1], color='blue', label=r'$\dot x$')
ax[1].plot(t, x_r[:,1], color='red', label=r'$\dot x_r$')
ax[1].plot(t, x_inter[:-1,1], color='lightblue', linestyle='--', label='$\dot x_{inter}$')
ax[1].legend()
ax[2].plot(t, u, color='blue', label='u')
ax[2].legend()

plt.figure()
plt.plot(t, x_ddot, color='blue', label=r'$\ddot x$')

plt.show()
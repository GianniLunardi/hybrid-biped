import numpy as np
import matplotlib.pyplot as plt
from hybrid_biped import BipedParams, HybridLipm, LipmMPC


params = BipedParams('../data')
hs = HybridLipm(params.dt, params.tau0, params)
mpc = LipmMPC(params.x_hat_0, params.y_hat_0, params)
foot_placement = np.copy(mpc.foot_steps_des)

n_steps = mpc.walk_time_des * params.ratio_mpc

# CoM state
x = np.zeros((n_steps+1, 2))
x[0,:] = params.x0
y = np.zeros((n_steps+1, 2))
y[0,:] = params.y_hat_0
# CoM_x reference
x_r = np.zeros((n_steps, 2))
# Error on x direction
eps = np.zeros((n_steps, 2))
# CoP
u_x = np.zeros(n_steps)
u_y = np.zeros(n_steps)

j = 0
for i in range(n_steps):
    # x direction
    if x[i,0] >= params.r_bar:
        x[i,:] = hs.jump(x[i,:])
    x_r[i,:] = hs.referenceWithTimer()
    eps[i,:] = x[i,:] - x_r[i,:]
    u_x[i] = hs.saturatedFb(eps[i,:])
    x[i+1,:] = hs.flow(x[i,:], u_x[i])
    # y direction
    if i % params.ratio_mpc == 0:
        mpc.step(j)
        mpc.updateCurrentState(hs.tau, x[i,:], y[i,:])
        j += 1

t = np.arange(0, len(x)-1) * params.dt
t_mpc = np.arange(0, len(mpc.Y_total)-1) * params.dt_mpc
y_mpc = np.copy(mpc.Y_total)
t_z = np.arange(0, len(mpc.Z_ref_k)) * params.dt_mpc

fig, ax = plt.subplots(2, 1)
ax[0].plot(t, x_r[:,0], color='DarkOrange', label='$x_r$')
ax[0].plot(t, x[:-1,0], color='blue', label='$x$')
ax[0].set_ylabel('CoM x HS position (m)')
ax[0].grid()
ax[0].legend()
ax[1].plot(t_mpc, y_mpc[:-1,0], color='blue', label='$y$')
ax[1].plot(t_mpc, mpc.Z_y_total, color='green', label='$u_y$')
ax[1].plot(t_mpc, mpc.Z_y_ref - params.foot_width/2, color='coral', linestyle='dashed', label='$u_{min}$')
ax[1].plot(t_mpc, mpc.Z_y_ref, color='red', linestyle='dashed', label='$u_{ref}$')
ax[1].plot(t_mpc, mpc.Z_y_ref + params.foot_width/2, color='darkred', linestyle='dashed', label='$u_{max}$')
ax[1].grid()
ax[1].legend()
ax[1].set_ylabel('CoM and CoP y positions (m)')
ax[1].set_xlabel('Time (s)')

plt.figure()
plt.plot(t, u_x, color='darkblue', label='$u_x$')
plt.legend()
plt.grid()
plt.ylabel('CoP x HS position (m)')
plt.xlabel('Time (s)')

plt.show()

# Save the results
np.savez('../data/linear_mpc.npz', x_com=x, y_com=y_mpc, u_x=u_x, u_y=mpc.Z_y_total, u_ref=mpc.Z_y_ref,
         foot_steps=foot_placement)

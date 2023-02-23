import numpy as np
import matplotlib.pyplot as plt
from hybrid_biped import BipedParams, HybridLipm, LipmMPC, rolloutLipmDynamics


params = BipedParams('../data')
hs = HybridLipm(params.dt, params.tau_hat_0, params)
mpc = LipmMPC(params)
foot_placement = np.copy(mpc.foot_steps_des)

nb_steps_per_T = int(params.T/params.dt_mpc)
walk_time_first = rolloutLipmDynamics(params.x_hat_0, params.tau_hat_0, mpc.dt_mpc, params)
walk_time_des = (params.nb_steps - 1) * nb_steps_per_T + walk_time_first
n_steps = walk_time_des * params.ratio

# CoM state
x = np.zeros((n_steps+1, 2))
x[0,:] = params.x_hat_0[0]
y = np.zeros((walk_time_des+1, 2))
y[0,:] = params.y_hat_0
# CoM_x reference
x_r = np.zeros((n_steps, 2))
# Error on x direction
eps = np.zeros((n_steps, 2))
# CoP
u_x = np.zeros(n_steps)
u_y = np.zeros(walk_time_des)
z_y = np.zeros(walk_time_des)

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
    if i % params.ratio == 0:
        mpc.updateCurrentState(hs.tau, x[i,:], y[j,:])
        mpc.step()
        y[j+1,:] = mpc.y_next
        u_y[j] = mpc.U_y_current
        z_y[j] = mpc.Z_y
        j += 1

t = np.arange(0, len(x)-1) * params.dt
t_mpc = np.arange(0, walk_time_des) * params.dt_mpc

fig, ax = plt.subplots(2, 1)
ax[0].plot(t, x_r[:,0], color='DarkOrange', label='$x_r$')
ax[0].plot(t, x[:-1,0], color='blue', label='$x$')
ax[0].set_ylabel('CoM x HS position (m)')
ax[0].grid()
ax[0].legend()
ax[1].plot(t_mpc, y[:-1,0], color='blue', label='$y$')
ax[1].plot(t_mpc, u_y, color='green', label='$u_y$')
ax[1].plot(t_mpc, z_y - params.foot_width/2, color='coral', linestyle='dashed', label='$u_{min}$')
ax[1].plot(t_mpc, z_y, color='red', linestyle='dashed', label='$u_{ref}$')
ax[1].plot(t_mpc, z_y + params.foot_width/2, color='darkred', linestyle='dashed', label='$u_{max}$')
ax[1].grid()
ax[1].legend()
ax[1].set_ylabel('CoM and CoP y positions (m)')
ax[1].set_xlabel('Time (s)')

# plt.figure()
# plt.plot(t, u_x, color='darkblue', label='$u_x$')
# plt.legend()
# plt.grid()
# plt.ylabel('CoP x HS position (m)')
# plt.xlabel('Time (s)')

plt.show()

# # Save the results
np.savez('../data/linear_mpc.npz', x_com=x, y_com=y, u_x=u_x, u_y=u_y, u_ref = z_y,
         foot_steps=foot_placement)

import numpy as np
import os
import matplotlib.pyplot as plt
from hybrid_biped import BipedParams, HybridLipm, LipmMPC, LinearMPC, rolloutLipmDynamics


CLASSIC = 0
SAVE_PLOT = 0
PLOT_COM_POS = 1
PLOT_COM_VEL = 0
PLOT_COP_HISTORY = 0

params = BipedParams()
hs = HybridLipm(params.dt, params.tau_hat_0, params)
if CLASSIC:
    mpc = LinearMPC(params)
else:
    mpc = LipmMPC(params)

foot_placement = np.copy(mpc.foot_steps_des)

nb_steps_per_T = int(params.T/params.dt_mpc)
walk_time_first = rolloutLipmDynamics(params.x_hat_0, params.tau_hat_0, params.dt_mpc, params)
if CLASSIC:
    walk_time_des = mpc.walk_time_des
else:
    walk_time_des = (params.nb_steps - 1) * nb_steps_per_T + walk_time_first
n_steps = walk_time_des * params.ratio

# CoM state
x = np.zeros((n_steps+1, 2))
x[0,:] = params.x_hat_0
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
    # Verify the jump condition along the longitudinal direction
    if x[i,0] >= params.r_bar:
        x[i,:] = hs.jump(x[i,:])
    # Compute the CoP along x
    x_r[i,:] = hs.referenceWithTimer()
    eps[i,:] = x[i,:] - x_r[i,:]
    u_x[i] = hs.saturatedFb(eps[i,:])
    # Compute the rollout (expected time for the next footstep)
    # i_left = rolloutLipmDynamics(x[i,:], hs.tau, params.dt, params)
    # Compute the MPC for the CoP along y
    if i % params.ratio == 0:
        mpc.updateCurrentState(hs.tau, x[i,:], y[j,:])
        # mpc.setTimeLeft(i_left)
        mpc.step()
        y[j+1,:] = mpc.y_next
        u_y[j] = mpc.U_y_current
        z_y[j] = mpc.Z_y
        j += 1
    # Perform the integration (flow dynamics)
    x[i+1,:] = hs.flow(x[i,:], u_x[i])


t = np.arange(0, len(x)-1) * params.dt
t_mpc = np.arange(0, walk_time_des) * params.dt_mpc

if PLOT_COM_POS:
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
    ax[1].set_ylim(bottom =-0.15, top= 0.15)
    ax[1].grid()
    ax[1].legend()
    ax[1].set_ylabel('CoM and CoP y positions (m)')
    ax[1].set_xlabel('Time (s)')

if PLOT_COM_VEL:
    plt.figure()
    plt.plot(t_mpc, y[:-1,1], color='darkblue', label='$\dot y$')
    plt.legend()
    plt.grid()
    plt.ylabel('CoP y velocity (m/s)')
    plt.xlabel('Time (s)')

if PLOT_COP_HISTORY:
    mpc_time_vec = np.arange(mpc.horizon) * params.dt_mpc
    if SAVE_PLOT:
        os.system('rm ../plots/*.png')
        for i in range(len(mpc.horizon_data)):
            fig, ax = plt.subplots(3, 1)
            ax[0].plot(mpc_time_vec, mpc.horizon_data[i]['Y_k'][:,0], label='com y pos ' + str(i))
            ax[0].set_ylabel('CoM y position (m)')
            ax[0].set_ylim(bottom =-0.15, top= 0.15)
            ax[0].grid(), ax[0].legend()
            ax[1].plot(mpc_time_vec, mpc.horizon_data[i]['Y_k'][:,1], label='com y vel ' + str(i))
            ax[1].set_ylabel('CoM y velocity (m/s)')
            ax[1].set_ylim(bottom =-0.5, top= 0.5)
            ax[1].grid(), ax[1].legend()
            ax[2].plot(mpc_time_vec, mpc.horizon_data[i]['zmp_reference'][:,0], label='zmp ref x ' + str(i), linestyle='dashed', color='orange')
            ax[2].plot(mpc_time_vec, mpc.horizon_data[i]['Z_x_k'], label='zmp x ' + str(i), color='orange')
            ax[2].plot(mpc_time_vec, mpc.horizon_data[i]['zmp_reference'][:,1], label='zmp ref y ' + str(i), linestyle='dashed', color='blue')
            ax[2].plot(mpc_time_vec, mpc.horizon_data[i]['Z_y_k'], label='zmp y ' + str(i), color='blue')
            ax[2].set_ylabel('ZMP reference (m)')
            ax[2].set_xlabel('Time (s)')
            ax[2].grid(), ax[2].legend()
            plt.legend()

            plt.savefig('../plots/lipm_mpc_horizon_' + str(i) + '.png', dpi=300)
            plt.close(fig)

if not SAVE_PLOT:
    plt.show()

# Save the results
np.savez('../data/linear_mpc.npz', x_com=x, y_com=y, u_x=u_x, u_y=u_y, u_ref=z_y,
         foot_steps=foot_placement)

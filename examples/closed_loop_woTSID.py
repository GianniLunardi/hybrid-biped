import matplotlib.pyplot as plt
import numpy as np
from hybrid_biped import *
from tqdm import tqdm
import plot_utils

PLOT_X = 1
PLOT_Y = 1
PLOT_ACC = 1
REAL_BOOL = 1
PLOT_COP_HISTORY = 0
SAVE_PLOT = 0

def integrateCoMAcc(p, v, a, dt):
    p += v * dt + 0.5 * a * dt ** 2
    v += a * dt
    return p, v

params = BipedParams()
hs = HybridLipm(params.dt, params.tau_hat_0, params)
mpc = LipmMPC(params)
inter = LipmToWbc(params)

nb_steps_per_T = int(params.T/params.dt_mpc)
walk_time_first = rolloutLipmDynamics(params.x_hat_0, params.tau_hat_0, mpc.dt_mpc, params)
walk_time_des = (params.nb_steps - 1) * nb_steps_per_T + walk_time_first
n_steps = walk_time_des * params.ratio

# CoM state
x = np.zeros((n_steps+1, 2))
x_ddot = np.zeros(n_steps)
x[0,:] = params.x_hat_0
y = np.zeros((n_steps+1,2))
y_ddot = np.zeros(n_steps)
y[0,:] = params.y_hat_0
y_mpc = np.zeros((walk_time_des+1, 2))
y_mpc[0,:] = params.y_hat_0
# CoM_x reference
x_r = np.zeros((n_steps, 2))
# Error on x direction
eps = np.zeros((n_steps, 2))
# CoP
u_x = np.zeros(n_steps)
u_y = np.zeros(walk_time_des)
z_y = np.zeros(walk_time_des)
# Foot steps
foot_steps = np.vstack([[-params.step_length, params.step_width/2],
                        np.copy(mpc.foot_steps_des)])

t_init = 1/params.omega * np.log(
    (-params.x_hat_0[0] - np.sqrt(params.v_bar** 2 / params.omega** 2 + params.x_hat_0[0]** 2 - params.r_bar**2)) /
    (params.r_bar - params.v_bar / params.omega))
hs.t_HS = round(t_init/params.dt) * params.dt

x_real = np.zeros((n_steps+1, 2))
y_real = np.zeros((n_steps+1, 2))
x_real[0,:] = params.x_hat_0
y_real[0,:] = params.y_hat_0

com_vel_int = np.zeros((n_steps+1, 2))
com_pos_int = np.zeros((n_steps+1, 2))

### SIMUALTION ###
j, k = 0, 1
stance, swing = 'right', 'left'
for i in tqdm(range(n_steps)):
    # x direction
    if REAL_BOOL:
        x[i,0] = x_real[i,0] - (k - 1) * params.step_length
        x[i,1] = x_real[i,1]
    if x[i, 0] >= params.r_bar:
        x[i,:] = hs.jump(x[i,:])
        stance, swing = swing, stance
        k += 1
    x_r[i,:] = hs.referenceWithTimer()
    eps[i,:] = x[i,:] - x_r[i,:]
    u_x[i] = hs.saturatedFb(eps[i,:])
    # y direction
    if i % params.ratio == 0:
        if REAL_BOOL:
            mpc.updateCurrentState(hs.tau, x[i,:], y_real[i,:])
        else:
            mpc.updateCurrentState(hs.tau, x[i,:], y_mpc[j,:])
        mpc.step()
        y_mpc[j+1,:] = mpc.y_next
        u_y[j] = mpc.U_y_current
        z_y[j] = mpc.Z_y
        j += 1
    x[i+1,:] = hs.flow(x[i,:], u_x[i])
    # Compute the accelerations
    x_ddot[i] = inter.computeCoMAcceleration(x[i,0], u_x[i])    # THIS SEEMS TO BE UNCORRECT
    if REAL_BOOL:
        y_ddot[i] = inter.computeCoMAcceleration(y_real[i,0], u_y[j-1])
    else:
        y_ddot[i] = inter.computeCoMAcceleration(y_mpc[j-1,0], u_y[j-1])
    # Integrate the acceleration
    if REAL_BOOL:
        # x_real[i+1,0], x_real[i+1,1] = integrateCoMAcc(x_real[i,0], x_real[i,1], x_ddot[i], params.dt)
        # y_real[i+1,0], y_real[i+1,1] = integrateCoMAcc(y_real[i,0], y_real[i,1], y_ddot[i], params.dt)
        x_real[i+1,1] = x_real[i,1] + x_ddot[i] * params.dt
        y_real[i+1,1] = y_real[i,1] + y_ddot[i] * params.dt
        x_real[i+1,0] = x_real[i,0] + x_real[i,1] * params.dt + 0.5 * x_ddot[i] * params.dt**2
        y_real[i+1,0] = y_real[i,0] + y_real[i,1] * params.dt + 0.5 * y_ddot[i] * params.dt**2
    else:
        y[i+1,:] = inter.integrateCoMLateralState(y[i,:], u_y[j-1], 0)

com_vel_int[0,0], com_vel_int[0,1] = x[0,1], y[0,1]
com_pos_int[0,0], com_pos_int[0,1] = x[0,0], y[0,0]

# for i in range(n_steps):
#     # INTEGRATION
#     com_vel_int[i+1,0] += com_vel_int[i,0] + x_ddot[i] * params.dt
#     com_vel_int[i+1,1] += com_vel_int[i,1] + y_ddot[i] * params.dt
#     com_pos_int[i+1,0] += com_pos_int[i,0] + com_vel_int[i,0] * params.dt + 0.5 * x_ddot[i] * params.dt**2
#     com_pos_int[i+1,1] += com_pos_int[i,1] + com_vel_int[i,1] * params.dt + 0.5 * y_ddot[i] * params.dt**2
#     # com_pos_int[i+1,0], com_vel_int[i+1,0] = integrateCoMAcc(com_pos_int[i,0], com_vel_int[i,0], x_ddot[i], params.dt)
#     # com_pos_int[i+1,1], com_vel_int[i+1,1] = integrateCoMAcc(com_pos_int[i,1], com_vel_int[i,1], y_ddot[i], params.dt)

com_acc_ref = np.zeros((n_steps, 2))
for i in range(n_steps):
    com_acc_ref[i,0] = x_ddot[i]
    com_acc_ref[i,1] = y_ddot[i]
    com_pos_int[i+1,:], com_vel_int[i+1,:] = integrateCoMAcc(com_pos_int[i,:], com_vel_int[i,:], com_acc_ref[i,:], params.dt)


### PLOTS ###
t = np.arange(0, n_steps) * params.dt
t_mpc = np.arange(0, walk_time_des) * params.dt_mpc

if not SAVE_PLOT:
    plt.rc('text', usetex = True)
    plt.rc('font', family ='serif')

if PLOT_X:
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t, x[:-1,0], color='green',label='$x$')
    if REAL_BOOL:
        ax[0].plot(t, x_real[:-1,0], color='red',linestyle='dashed', label='$x$ real')
    else:
        ax[0].plot(t, com_pos_int[:-1, 0], color='coral', label='$x$ int', linestyle='dashed')
    ax[0].legend()
    ax[0].set_ylabel('CoP x position (m)')
    ax[1].plot(t, x[:-1,1], color='green', label='$\dot x$')
    if REAL_BOOL:
        ax[1].plot(t, x_real[:-1,1], color='red', linestyle = 'dashed', label='$\dot x$ real')
    else:
        ax[1].plot(t, com_vel_int[:-1, 0], color='coral', label='$\dot x$ int', linestyle='dashed')
    ax[1].set_ylabel('CoM x velocity (m/s)')
    ax[1].set_xlabel('Time (s)')
    ax[1].legend()

if PLOT_Y:
    fig1, ax = plt.subplots(2, 1)
    ax[0].set_ylim([-0.18, 0.18])
    ax[0].plot(t_mpc, u_y, color='purple', label='$u_y$')
    if REAL_BOOL:
        ax[0].plot(t, y_real[:-1,0], color='red', linestyle='dashed', label='$y_{real}$')
    else:
        ax[0].plot(t, y[:-1, 0], color='green', label='$y$')
        ax[0].plot(t, com_pos_int[:-1,1], color='coral', label='$y$ int', linestyle='dashed')
    ax[0].plot(t_mpc, z_y - params.foot_width/2, color='darkblue', linestyle='dashed', label='$u_{right}$')
    ax[0].plot(t_mpc, z_y, color='blue', linestyle='dashed', label='$u_{ref}$')
    ax[0].plot(t_mpc, z_y + params.foot_width/2, color='turquoise', linestyle='dashed', label='$u_{left}$')
    ax[0].legend()
    ax[0].set_ylabel('CoM and CoP y positions (m)')
    if REAL_BOOL:
        ax[1].plot(t, y_real[:-1,1], color='red', linestyle='dashed', label='$\dot y$ real')
    else:
        ax[1].plot(t, y[:-1, 1], color='green', label='$\dot y$')
        ax[1].plot(t, com_vel_int[:-1,1], color='coral', linestyle='dashed', label='$\dot y$ int')
    ax[1].set_ylabel('CoM y velocity (m/s)')
    ax[1].set_xlabel('Time (s)')
    ax[1].legend()

if PLOT_ACC:
    fig2, ax = plt.subplots(2, 1)
    ax[0].plot(t, x_ddot, color='blue', label='$\ddot x$')
    ax[0].set_ylabel('CoM x acceleration ($m/s^2$)')
    ax[0].legend()
    ax[1].plot(t, y_ddot, color='blue', label='$\ddot y$')
    ax[1].legend()
    ax[1].set_ylabel('CoM y acceleration ($m/s^2$)')
    ax[1].set_xlabel('Time (s)')


if PLOT_COP_HISTORY:
    mpc_time_vec = np.arange(mpc.horizon) * params.dt_mpc
    if SAVE_PLOT:
        os.system('rm ../plots/*.png')
        print("SAVING PLOTS")
        for i in tqdm(range(40)):
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
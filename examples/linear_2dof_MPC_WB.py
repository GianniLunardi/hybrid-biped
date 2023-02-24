import matplotlib.pyplot as plt
from hybrid_biped import *


params = BipedParams('../data')
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
foot_steps = np.vstack([[-0.3, 0.08],
                        np.copy(mpc.foot_steps_des)])
feet_traj = dict()
feet_traj['left'] = np.zeros((n_steps, 3, 3))
feet_traj['right'] = np.zeros((n_steps, 3, 3))


### SIMUALTION ###
j, k = 0, 1
count = 1
stance, swing = 'right', 'left'
for i in range(n_steps):
    # x direction
    if x[i,0] >= params.r_bar:
        x[i,:] = hs.jump(x[i,:])
        stance, swing = swing, stance
        k += 1
    x_r[i,:] = hs.referenceWithTimer()
    eps[i,:] = x[i,:] - x_r[i,:]
    u_x[i] = hs.saturatedFb(eps[i,:])
    x[i+1,:] = hs.flow(x[i,:], u_x[i])
    x_ddot[i] = inter.computeCoMAcceleration(x[i,0], u_x[i])
    # y direction
    if i % params.ratio == 0:
        mpc.updateCurrentState(hs.tau, x[i,:], y_mpc[j,:])
        mpc.step()
        y_mpc[j+1,:] = mpc.y_next
        u_y[j] = mpc.U_y_current
        z_y[j] = mpc.Z_y
        j += 1
        count = 0
    y[i+1,:] = inter.integrateCoMLateralState(y_mpc[j-1], u_y[j-1], count)
    count +=1
    y_ddot[i] = inter.computeCoMAcceleration(y[i,0], u_y[j-1])
    # Feet trajectories
    inter.set_time(hs.t_HS, x[i,:], hs.tau)
    feet_traj[swing][i,:,:] = inter.footTrajectoryFromHS(foot_steps[k-1,:], foot_steps[k+1,:])
    feet_traj[stance][i,:,:] = inter.footTrajectoryFromHS(foot_steps[k,:], foot_steps[k,:], 0)
    # if np.linalg.norm(x[i,0] - x[i,0]) > params.r_bar:
    #     stance, swing = swing, stance
    #     k +=1


### PLOTS ###
t = np.arange(0, n_steps) * params.dt
t_mpc = np.arange(0, walk_time_des) * params.dt_mpc

# MPC
# fig1, ax = plt.subplots(2, 1)
# ax[0].plot(t, x_r[:,0], color='DarkOrange', label='$x_r$')
# ax[0].plot(t, x[:-1,0], color='blue', label='$x$')
# ax[0].set_ylabel('CoM x HS position (m)')
# ax[0].grid()
# ax[0].legend()
# ax[1].plot(t_mpc, y_mpc[:-1,0], color='blue', label='$y$')
# ax[1].plot(t_mpc, u_y, color='green', label='$u_y$')
# ax[1].plot(t_mpc, z_y - params.foot_width/2, color='coral', linestyle='dashed', label='$u_{min}$')
# ax[1].plot(t_mpc, z_y, color='red', linestyle='dashed', label='$u_{ref}$')
# ax[1].plot(t_mpc, z_y + params.foot_width/2, color='darkred', linestyle='dashed', label='$u_{max}$')
# ax[1].grid()
# ax[1].legend()
# ax[1].set_ylabel('CoM and CoP y positions (m)')
# ax[1].set_xlabel('Time (s)')

# INTER
fig, ax = plt.subplots(2, 1)
ax[0].plot(t_mpc, z_y, color='red', linestyle='dashed', label='CoP ref')
ax[0].plot(t, y[:-1,0], color='green', linewidth=2.5, label='$y_{wb}$')
ax[0].plot(t_mpc, y_mpc[:-1,0], color='blue', linestyle='dashed', label='$y_{mpc}$')
ax[0].grid()
ax[0].legend()
ax[0].set_ylabel('CoM and CoP y positions (m)')
ax[1].plot(t, y[:-1,1], color='green', linewidth=2.5, label='$\dot y_{wb}$')
ax[1].plot(t_mpc, y_mpc[:-1,1], color='blue', linestyle='dashed', label='$\dot y_{mpc}$')
ax[1].set_ylabel('CoM y velocity (m/s)')
ax[1].set_xlabel('Time (s)')
ax[1].grid()
ax[1].legend()

# CoM acc
fig2, ax = plt.subplots(2, 1)
ax[0].plot(t, x_ddot, color='blue', label='$\ddot x$')
ax[0].set_ylabel('CoM x acceleration (m/s^2)')
ax[0].legend()
ax[0].grid()
ax[1].plot(t, y_ddot, color='blue', label='$\ddot y$')
ax[1].legend()
ax[1].grid()
ax[1].set_ylabel('CoM y acceleration (m/s^2)')
ax[1].set_xlabel('Time (s)')

# FEET
fig3, ax = plt.subplots(3, 1)
ax[0].plot(t, feet_traj['left'][:,0,0], color='darkorange', label='$x_{LF}$')
ax[0].plot(t, feet_traj['right'][:,0,0], color='darkblue', label='$x_{RF}$')
ax[0].grid(), ax[0].legend(), ax[0].set_ylim([-0.4, 1.8]), ax[0].set_ylabel('x (m)')
ax[1].plot(t, feet_traj['left'][:,1,0], color='darkorange', label='$y_{LF}$')
ax[1].plot(t, feet_traj['right'][:,1,0], color='darkblue', label='$y_{RF}$')
ax[1].grid(), ax[1].legend(), ax[1].set_ylabel('y (m)')
ax[2].plot(t, feet_traj['left'][:,2,0], color='darkorange', label='$z_{LF}$')
ax[2].plot(t, feet_traj['right'][:,2,0], color='darkblue', label='$z_{RF}$')
ax[2].grid(), ax[2].legend(), ax[2].set_ylim([-0.01, 0.06]), ax[2].set_ylabel('z (m)')
ax[2].set_xlabel('Time (s)')


plt.show()
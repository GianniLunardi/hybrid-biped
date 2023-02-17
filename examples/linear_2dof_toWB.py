import numpy as np
import matplotlib.pyplot as plt
from hybrid_biped import BipedParams, LipmToWbc


params = BipedParams('../data')
data = np.load('../data/linear_mpc.npz')
X_wb = data['x_com']
U_x_wb = data['u_x']
Y_mpc = data['y_com']
U_y_mpc = data['u_y']
U_ref = data['u_ref']
foot_steps = np.vstack([[-0.3, 0.05] ,data['foot_steps'][:-1]])

wbc = LipmToWbc(params)
n_wb = len(X_wb) - 1
n_mpc = len(Y_mpc) - 1

# CoM x acceleration
X_ddot_wb = wbc.computeCoMAcceleration(X_wb[:-1,0], U_x_wb)

# CoM y acceleration
Y_wb = np.zeros((n_wb+1, 2))
Y_wb[0,:] = Y_mpc[0,:]
Y_ddot_wb = np.zeros(n_wb)
for i in range(n_mpc):
    for j in range(params.ratio_mpc):
        ii = i * params.ratio_mpc + j
        Y_wb[ii+1,:] = wbc.integrateCoMLateralState(Y_mpc[i,:], U_y_mpc[i], j)
        Y_ddot_wb[ii] = wbc.computeCoMAcceleration(Y_wb[ii,0], U_y_mpc[i])

# Feet trajectories (starting from RF)
feet_traj = dict()
feet_traj['LF'] = np.zeros((n_wb, 3, 3))
feet_traj['RF'] = np.zeros((n_wb, 3, 3))
j = 1
stance = 'RF'
swing = 'LF'
contact_phase = []
x0 = X_wb[0,:]
for i in range(n_wb):
    feet_traj[swing][i,:,:] = wbc.footTrajectoryFromHS(X_wb[i,:], x0, U_x_wb[i], foot_steps[j-1,:], foot_steps[j+1,:])
    feet_traj[stance][i,:,:] = wbc.footTrajectoryFromHS(X_wb[i,:], x0, U_x_wb[i], foot_steps[j,:], foot_steps[j,:], 0)
    contact_phase.append(stance)
    if np.linalg.norm(X_wb[i+1,0] - X_wb[i,0]) > params.r_bar:
        stance, swing = swing, stance
        j +=1
        x0 = X_wb[i+1,:]

#### PLOTS ####
t_mpc = np.arange(0, n_mpc) * params.dt_mpc
t_wb = np.arange(0, n_wb) * params.dt

fig, ax = plt.subplots(2, 1)
ax[0].plot(t_mpc, U_ref, color='red', linestyle='dashed', label='CoP ref')
ax[0].plot(t_wb, Y_wb[:-1,0], color='green', linewidth=2.5, label='$y_{wb}$')
ax[0].plot(t_mpc, Y_mpc[:-1,0], color='blue', linestyle='dashed', label='$y_{mpc}$')
ax[0].grid()
ax[0].legend()
ax[0].set_ylabel('CoM and CoP y positions (m)')
ax[1].plot(t_wb, Y_wb[:-1,1], color='green', linewidth=2.5, label='$\dot y_{wb}$')
ax[1].plot(t_mpc, Y_mpc[:-1,1], color='blue', linestyle='dashed', label='$\dot y_{mpc}$')
ax[1].set_ylabel('CoM y velocity (m/s)')
ax[1].set_xlabel('Time (s)')
ax[1].grid()
ax[1].legend()

fig2, ax = plt.subplots(2, 1)
ax[0].plot(t_wb, X_ddot_wb, color='blue', label='$\ddot x$')
ax[0].set_ylabel('CoM x acceleration (m/s^2)')
ax[0].legend()
ax[0].grid()
ax[1].plot(t_wb, Y_ddot_wb, color='blue', label='$\ddot y$')
ax[1].legend()
ax[1].grid()
ax[1].set_ylabel('CoM y acceleration (m/s^2)')
ax[1].set_xlabel('Time (s)')

fig3, ax = plt.subplots(3, 1)
ax[0].plot(t_wb, feet_traj['LF'][:,0,0], color='darkorange', label='$x_{LF}$')
ax[0].plot(t_wb, feet_traj['RF'][:,0,0], color='darkblue', label='$x_{RF}$')
ax[0].grid(), ax[0].legend(), ax[0].set_ylim([-0.4, 1.8]), ax[0].set_ylabel('x (m)')
ax[1].plot(t_wb, feet_traj['LF'][:,1,0], color='darkorange', label='$y_{LF}$')
ax[1].plot(t_wb, feet_traj['RF'][:,1,0], color='darkblue', label='$y_{RF}$')
ax[1].grid(), ax[1].legend(), ax[1].set_ylabel('y (m)')
ax[2].plot(t_wb, feet_traj['LF'][:,2,0], color='darkorange', label='$z_{LF}$')
ax[2].plot(t_wb, feet_traj['RF'][:,2,0], color='darkblue', label='$z_{RF}$')
ax[2].grid(), ax[2].legend(), ax[2].set_ylim([-0.01, 0.06]), ax[2].set_ylabel('z (m)')
ax[2].set_xlabel('Time (s)')


plt.show()

# Save the results
np.savez('../data/walking_ref.npz', ddcom_x = X_ddot_wb, ddcom_y = Y_ddot_wb,
         x_RF=feet_traj['RF'][:,:,0], dx_RF=feet_traj['RF'][:,:,1], ddx_RF=feet_traj['RF'][:,:,2],
         x_LF=feet_traj['LF'][:,:,0], dx_LF=feet_traj['LF'][:,:,1], ddx_LF=feet_traj['LF'][:,:,2],
         contact_phase=contact_phase
         )

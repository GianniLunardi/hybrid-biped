import numpy as np
import matplotlib.pyplot as plt
from hybrid_biped import BipedParams, LipmToWbc


params = BipedParams()
data = np.load('../data/linear_mpc.npz')
X_wb = data['x_com']
U_x_wb = data['u_x']
Y_mpc = data['y_com']
U_y_mpc = data['u_y']
U_ref = data['u_ref']
foot_steps = np.vstack([[-params.step_length, params.step_width/2] ,data['foot_steps'][:-1]])

wbc = LipmToWbc(params)
n_wb = len(X_wb) - 1
n_mpc = len(Y_mpc) - 1

# CoM x acceleration
X_ddot_wb = wbc.computeCoMAcceleration(X_wb[:-1,0], U_x_wb)

# Compute predicted period for steps
T_pred = []
count = 0
for i in range(n_wb):
    if np.linalg.norm(X_wb[i + 1, 0] - X_wb[i, 0]) < params.r_bar:
        count += 1
    else:
        T_pred.append(count * params.dt)
        count = 0
T_pred.append(T_pred[-1])

# CoM y acceleration
Y_wb = np.zeros((n_wb+1, 2))
Y_wb[0,:] = Y_mpc[0,:]
Y_ddot_wb = np.zeros(n_wb)
for i in range(n_mpc):
    for j in range(params.ratio):
        ii = i * params.ratio + j
        Y_wb[ii+1,:] = wbc.integrateCoMLateralState(Y_mpc[i,:], U_y_mpc[i], j)
        Y_ddot_wb[ii] = wbc.computeCoMAcceleration(Y_wb[ii,0], U_y_mpc[i])

# Feet trajectories (starting from RF)
feet_traj = dict()
feet_traj['LF'] = np.zeros((n_wb, 3, 3))
feet_traj['RF'] = np.zeros((n_wb, 3, 3))
j = 1
stance = 'RF'
swing = 'LF'
x0 = X_wb[0,:]

# Initial count
t_init = 1/params.omega * np.log(
    (-params.x_hat_0[0] - np.sqrt(params.v_bar** 2 / params.omega** 2 + params.x_hat_0[0]** 2 - params.r_bar**2)) /
    (params.r_bar - params.v_bar / params.omega))
count = np.round(t_init / params.dt)
T_pred[0] += count * params.dt

for i in range(n_wb):
    wbc.set_time_offline(count * params.dt, T_pred[0])
    count += 1
    feet_traj[swing][i,:,:] = wbc.footTrajectoryFromHS(foot_steps[j-1,:], foot_steps[j+1,:])
    feet_traj[stance][i,:,:] = wbc.footTrajectoryFromHS(foot_steps[j,:], foot_steps[j,:], 0)
    if np.linalg.norm(X_wb[i+1,0] - X_wb[i,0]) > params.r_bar:
        stance, swing = swing, stance
        j +=1
        x0 = X_wb[i+1,:]
        count = 1
        T_pred.pop(0)

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

fig4 = plt.figure()
plt.plot(feet_traj['LF'][:,0,0], feet_traj['LF'][:,2,0], color='darkorange', label='LF')
plt.plot(feet_traj['RF'][:,0,0], feet_traj['RF'][:,2,0], color='darkblue', label='RF')
plt.grid(), plt.legend()
plt.xlabel('x (m)')
plt.ylabel('z (m)')

# plt.figure()
# plt.plot(t_wb, wbc.t_arr)

#### TRY OF POLY ####

# t_try = np.linspace(0, params.T, n_wb)
# xy_try = np.zeros((n_wb, 2, 3))
# for i in range(n_wb):
#     xy_try[i,:,:] = wbc.fifthOrderPolynomial(t_try[i], np.array([2, 0]), np.array([3, 2]))
# z_try = wbc.sixthOrderPolynomial(t_try, 0.05)
# print(n_wb)
#
# plt.figure()
# plt.plot(t_try, z_try[:,0])
#
# plt.figure()
# plt.plot(t_wb, xy_try[:,0,0])
# plt.plot(t_wb, xy_try[:,1,0])

plt.show()

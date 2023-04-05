import matplotlib.pyplot as plt
from hybrid_biped import *


params = BipedParams('../data')
hs = HybridLipm(params.dt, params.tau_hat_0, params)
mpc = LipmMPC(params)
inter = LipmToWbc(params)
tsid = TsidBiped(params)

nb_steps_per_T = int(params.T/params.dt_mpc)
walk_time_first = rolloutLipmDynamics(params.x_hat_0, params.tau_hat_0, mpc.dt_mpc, params)
walk_time_des = (params.nb_steps - 1) * nb_steps_per_T + walk_time_first
n_steps = walk_time_des * params.ratio

N_pre = int(params.T_pre / params.dt)
N_post = int(params.T_post / params.dt)

# CoM state
x = np.zeros((n_steps+1, 2))
x_real = np.zeros((n_steps+1, 2))
x_ddot = np.zeros(n_steps)
x[0,:] = params.x_hat_0
x_real[0,:] = params.x_hat_0

y = np.zeros((n_steps+1,2))
y_ddot = np.zeros(n_steps)
y[0,:] = params.y_hat_0
y_mpc = np.zeros((walk_time_des+1, 2))
y_mpc[0,:] = params.y_hat_0

# CoM x reference
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
# For TSID
com_pos = np.zeros((n_steps+N_post, 3))
com_vel = np.zeros((n_steps+N_post, 3))
com_acc = np.zeros((n_steps+N_post, 3))
com_pos_ref = np.zeros((n_steps, 3))
com_vel_ref = np.zeros((n_steps, 3))
com_acc_ref = np.zeros((n_steps, 3))
x_LF = np.zeros_like(com_acc)
x_RF = np.zeros_like(com_acc)

# Start to walk
offset = x_real[0,0] - tsid.robot.com(tsid.formulation.data())[0]

### SIMUALTION ###
j, k = 0, 1
count = 1
stance, swing = 'right', 'left'

t = -params.T_pre
q, v = tsid.q, tsid.v
com_pos_ref[0,0] = x_real[0,0] - offset

for i in range(-N_pre, n_steps + N_post):
    if i == 0:
        tsid.remove_contact_LF()
    if i < 0:
        tsid.set_com_ref(com_pos_ref[0,:], com_vel_ref[0,:], com_acc_ref[0,:])
    elif i < n_steps:
        # x direction
        if x[i,0] >= params.r_bar:
            x[i,:] = hs.jump(x[i,:])
            stance, swing = swing, stance
            k += 1
            if stance == 'left':
                tsid.add_contact_LF()
                tsid.remove_contact_RF()
            else:
                tsid.add_contact_RF()
                tsid.remove_contact_LF()
        # Roll out the real CoM x position
        x_real[i,0] = x[i,0] + (k - 1) * params.step_length
        x_real[i,1] = x[i,1]

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
        # TSID problem
        com_pos_ref[i,:] = np.array([x_real[i,0] - offset, y[i,0], params.h_com])
        com_vel_ref[i,:] = np.array([x_real[i,1], y[i,1], 0])
        com_acc_ref[i,:] = np.array([x_ddot[i], y_ddot[i], 0])
        tsid.set_com_ref(com_pos_ref[i,:], com_vel_ref[i,:], com_acc_ref[i,:])
        tsid.set_LF_3d_ref(feet_traj['left'][i,:,0], feet_traj['left'][i,:,1], feet_traj['left'][i,:,2])
        tsid.set_RF_3d_ref(feet_traj['right'][i,:,0], feet_traj['right'][i,:,1], feet_traj['right'][i,:,2])

    HQPData = tsid.formulation.computeProblemData(t, q, v)
    sol = tsid.solver.solve(HQPData)
    if sol.status != 0:
        print("QP problem could not be solved! Error code:", sol.status)
        break
    if np.linalg.norm(v, 2) > 40.0:
        print('Time ', i * params.dt,' Velocities are too high, stop everything! ', np.linalg.norm(v))
        break
    dv = tsid.formulation.getAccelerations(sol)
    q, v = tsid.integrate_dv(q, v, dv, params.dt)
    if i >= 0:
        com_pos[i,:] = tsid.robot.com(tsid.formulation.data())
        com_vel[i,:] = tsid.robot.com_vel(tsid.formulation.data())
        com_acc[i,:] = tsid.comTask.getAcceleration(dv)
        x_LF[i,:], dx_LF, ddx_LF = tsid.get_LF_3d_pos_vel_acc(dv)
        x_RF[i,:], dx_RF, ddx_RF = tsid.get_RF_3d_pos_vel_acc(dv)

    # Display
    if i % params.DISPLAY_N == 0:
        tsid.display(q)

    t += params.dt

### PLOTS ###
t = np.arange(0, n_steps + N_post) * params.dt
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

# fig, ax = plt.subplots(2, 1)
# ax[0].plot(t, x_real[:-1,0], color='green', linewidth=2.5, label='$x_{wb}$')
# ax[0].grid()
# ax[0].legend()
# ax[0].set_ylabel('CoM real x position (m)')
# ax[1].plot(t_mpc, z_y, color='red', linestyle='dashed', label='CoP ref')
# ax[1].plot(t, y[:-1,0], color='green', linewidth=2.5, label='$y_{wb}$')
# ax[1].plot(t_mpc, y_mpc[:-1,0], color='blue', linestyle='dashed', label='$y_{mpc}$')
# ax[1].set_ylabel('CoM and CoP y positions (m)')
# ax[1].set_xlabel('Time (s)')
# ax[1].grid()
# ax[1].legend()

# CoM pos
fig, ax = plt.subplots(3, 1)
ax[0].plot(t, com_pos[:,0], color='blue', label='$x$')
ax[0].plot(t[:n_steps], com_pos_ref[:,0], color='red', linestyle='dashed', label='$x_{ref}$')
ax[0].set_ylabel('CoM x position (m)')
ax[0].legend()
ax[0].grid()
ax[1].plot(t, com_pos[:,1], color='blue', label='$y$')
ax[1].plot(t[:n_steps], com_pos_ref[:,1], color='red', linestyle='dashed', label='$y_{ref}$')
ax[1].legend()
ax[1].grid()
ax[1].set_ylabel('CoM y position (m)')
ax[2].plot(t, com_pos[:,2], color='blue', label='$z$')
ax[2].plot(t[:n_steps], com_pos_ref[:,2], color='red', linestyle='dashed', label='$z_{ref}$')
ax[2].legend()
ax[2].grid()
ax[2].set_ylabel('CoM z position (m)')
ax[2].set_xlabel('Time (s)')

# CoM vel
fig1, ax = plt.subplots(3, 1)
ax[0].plot(t, com_vel[:,0], color='blue', label='$\dot x$')
ax[0].plot(t[:n_steps], com_vel_ref[:,0], color='red', linestyle='dashed', label='$\dot x_{ref}$')
ax[0].set_ylabel('CoM x velocity (m/s)')
ax[0].legend()
ax[0].grid()
ax[1].plot(t, com_vel[:,1], color='blue', label='$\dot y$')
ax[1].plot(t[:n_steps], com_vel_ref[:,1], color='red', linestyle='dashed', label='$\dot y_{ref}$')
ax[1].legend()
ax[1].grid()
ax[1].set_ylabel('CoM y velocity (m/s)')
ax[2].plot(t, com_vel[:,2], color='blue', label='$\dot z$')
ax[2].plot(t[:n_steps], com_vel_ref[:,2], color='red', linestyle='dashed', label='$\dot z_{ref}$')
ax[2].legend()
ax[2].grid()
ax[2].set_ylabel('CoM z velocity (m/s)')
ax[2].set_xlabel('Time (s)')

# CoM acc
fig2, ax = plt.subplots(3, 1)
ax[0].plot(t, com_acc[:,0], color='blue', label='$\ddot x$')
ax[0].plot(t[:n_steps], com_acc_ref[:,0], color='red', linestyle='dashed', label='$\ddot x_{ref}$')
ax[0].set_ylabel('CoM x acceleration (m/s^2)')
ax[0].legend()
ax[0].grid()
ax[1].plot(t, com_acc[:,1], color='blue', label='$\ddot y$')
ax[1].plot(t[:n_steps], com_acc_ref[:,1], color='red', linestyle='dashed', label='$\ddot y_{ref}$')
ax[1].legend()
ax[1].grid()
ax[1].set_ylabel('CoM y acceleration (m/s^2)')
ax[2].plot(t, com_acc[:,2], color='blue', label='$\ddot z$')
ax[2].plot(t[:n_steps], com_acc_ref[:,2], color='red', linestyle='dashed', label='$\ddot z_{ref}$')
ax[2].legend()
ax[2].grid()
ax[2].set_ylabel('CoM z acceleration (m/s^2)')
ax[2].set_xlabel('Time (s)')

# FEET
# fig3, ax = plt.subplots(3, 1)
# ax[0].plot(t, x_LF[:,0], color='orange', label='$x_{LF}$')
# ax[0].plot(t, x_RF[:,0], color='blue', label='$x_{RF}$')
# ax[0].plot(t, feet_traj['left'][:,0,0], color='darkorange', linestyle='dashed', label='$x_{LF,ref}$')
# ax[0].plot(t, feet_traj['right'][:,0,0], color='darkblue', linestyle='dashed', label='$x_{RF,ref}$')
# ax[0].grid(), ax[0].legend(), ax[0].set_ylim([-0.4, 1.8]), ax[0].set_ylabel('x (m)')
# ax[1].plot(t, x_LF[:,1], color='orange', label='$y_{LF}$')
# ax[1].plot(t, x_RF[:,1], color='blue', label='$y_{RF}$')
# ax[1].plot(t, feet_traj['left'][:,1,0], color='darkorange', linestyle='dashed', label='$y_{LF,ref}$')
# ax[1].plot(t, feet_traj['right'][:,1,0], color='darkblue', linestyle='dashed', label='$y_{RF,ref}$')
# ax[1].grid(), ax[1].legend(), ax[1].set_ylabel('y (m)')
# ax[2].plot(t, x_LF[:,2], color='orange', label='$z_{LF}$')
# ax[2].plot(t, x_RF[:,2], color='blue', label='$z_{RF}$')
# ax[2].plot(t, feet_traj['left'][:,2,0], color='darkorange', linestyle='dashed', label='$z_{LF,ref}$')
# ax[2].plot(t, feet_traj['right'][:,2,0], color='darkblue', linestyle='dashed', label='$z_{RF,ref}$')
# ax[2].grid(), ax[2].legend(), ax[2].set_ylim([-0.01, 0.06]), ax[2].set_ylabel('z (m)')
# ax[2].set_xlabel('Time (s)')


plt.show()
import matplotlib.pyplot as plt
import numpy as np
from hybrid_biped import *
from numpy import nan

PLOT_COM = 1
PLOT_FEET = 1
PLOT_JOINT_POS = 0
PLOT_JOINT_VEL = 0

TSID_BOOL = 1

params = BipedParams()
hs = HybridLipm(params.dt, params.tau_hat_0, params)
mpc = LipmMPC(params)
inter = LipmToWbc(params)
tsid = TsidBiped(params)

nb_steps_per_T = int(params.T/params.dt_mpc)
walk_time_first = rolloutLipmDynamics(params.x_hat_0, params.tau_hat_0, mpc.dt_mpc, params)
walk_time_des = (params.nb_steps - 1) * nb_steps_per_T + walk_time_first
n_steps = walk_time_des * params.ratio

N_post = int(params.T_post / params.dt)

# CoM state
x = np.empty((n_steps+1, 2)) * nan
x_real = np.empty((n_steps+1, 2)) * nan
x_ddot = np.empty(n_steps) * nan
x[0,:] = params.x_hat_0
x_real[0,:] = params.x_hat_0

y = np.empty((n_steps+1,2)) * nan
y_ddot = np.empty(n_steps) * nan
y[0,:] = params.y_hat_0
y_mpc = np.empty((walk_time_des+1, 2)) * nan
y_mpc[0,:] = params.y_hat_0

# CoM x reference
x_r = np.empty((n_steps, 2)) * nan
# Error on x direction
eps = np.empty((n_steps, 2)) * nan
# CoP
u_x = np.empty(n_steps) * nan
u_y = np.empty(walk_time_des) * nan
z_y = np.empty(walk_time_des) * nan
# Foot steps
foot_steps = np.vstack([[-params.step_length, params.step_width/2],
                        np.copy(mpc.foot_steps_des)])
feet_traj = dict()
feet_traj['left'] = np.empty((n_steps, 3, 3)) * nan
feet_traj['right'] = np.empty((n_steps, 3, 3)) * nan
# For TSID
com_pos = np.empty((n_steps+N_post, 3)) * nan
com_vel = np.empty((n_steps+N_post, 3)) * nan
com_acc = np.empty((n_steps+N_post, 3)) * nan
com_acc_des = np.empty((n_steps+N_post, 3)) * nan
com_pos_ref = np.empty((n_steps, 3)) * nan
com_vel_ref = np.empty((n_steps, 3)) * nan
com_acc_ref = np.empty((n_steps, 3)) * nan

x_LF = np.empty_like(com_acc) * nan
x_RF = np.empty_like(com_acc) * nan
f_RF = np.zeros((n_steps+N_post, 6))
f_LF = np.zeros((n_steps+N_post, 6))
cop_RF = np.zeros((n_steps+N_post, 2))
cop_LF = np.zeros((n_steps+N_post, 2))

tau_log = np.zeros((n_steps+N_post, tsid.robot.na))
q_log  = np.zeros((n_steps+N_post, tsid.robot.nq))
v_log  = np.zeros((n_steps+N_post, tsid.robot.nv))

# Start to walk
offset = x_real[0,0] - tsid.robot.com(tsid.formulation.data())[0]

j, k = 0, 1
count = 1
stance, swing = 'right', 'left'

### INVERSE KINEMATICS ###
t_init = 1/params.omega * np.log(
    (-params.x_hat_0[0] - np.sqrt(params.v_bar** 2 / params.omega** 2 + params.x_hat_0[0]** 2 - params.r_bar**2)) /
    (params.r_bar - params.v_bar / params.omega))
inter.set_time(t_init)
x_LF_0 = inter.footTrajectoryFromHS(foot_steps[0,:], foot_steps[2,:], rescaling=False)
x_RF_0 = inter.footTrajectoryFromHS(foot_steps[1,:], foot_steps[1,:], foot_h=0, rescaling=False)
x_LF_0[2,0] += params.lz
x_RF_0[2,0] += params.lz
ik = InverseKinematics(params, x_LF=x_LF_0[:,0], x_RF=x_RF_0[:,0], logger=False)
ik.compute_inverse_geometry()
x_dot_com = np.array([params.ref_0[0], params.ref_0[1], 0.0]) #([params.x_hat_0[1], params.y_hat_0[1], 0.])

t = 0
q, v = ik.q, ik.compute_state_velocity(x_LF_0[:,1], x_RF_0[:,1], x_dot_com)
hs.t_HS = round(t_init / params.dt) * params.dt

### SIMUALTION ###
tsid.remove_contact_LF()
print("Starting to walk (remove contact left foot)")
for i in range(n_steps ):
    if i < n_steps:
        # Verify the jump condition along the longitudinal direction
        if x[i,0] >= params.r_bar:
            x[i,:] = hs.jump(x[i,:])
            print("Time %.3f Changing contact phase from %s to %s" % (t, stance, swing))
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
        # Compute the CoP along x
        x_r[i,:] = hs.referenceWithTimer()
        eps[i,:] = x[i,:] - x_r[i,:]
        u_x[i] = hs.saturatedFb(eps[i,:])
        x_ddot[i] = inter.computeCoMAcceleration(x[i,0], u_x[i])
        # Compute the MPC for the CoP along y
        if i % params.ratio == 0:
            mpc.updateCurrentState(hs.tau, x[i,:], y_mpc[j,:])
            mpc.step()
            y_mpc[j+1,:] = mpc.y_next
            u_y[j] = mpc.U_y_current
            z_y[j] = mpc.Z_y
            j += 1
            count = 0
        # Perform the integration (flow dynamics)
        x[i+1,:] = hs.flow(x[i,:], u_x[i])
        y[i+1,:] = inter.integrateCoMLateralState(y_mpc[j-1], u_y[j-1], count)
        count +=1
        y_ddot[i] = inter.computeCoMAcceleration(y[i,0], u_y[j-1])
        # Feet trajectories
        inter.set_time(hs.t_HS, x[i,:], hs.tau)
        feet_traj[swing][i,:,:] = inter.footTrajectoryFromHS(foot_steps[k-1,:], foot_steps[k+1,:])
        feet_traj[stance][i,:,:] = inter.footTrajectoryFromHS(foot_steps[k,:], foot_steps[k,:], 0)
        feet_traj['left'][i, 2, 0] += params.lz
        feet_traj['right'][i, 2, 0] += params.lz

        com_pos_ref[i, :] = np.array([x_real[i, 0], y[i, 0], params.h_com])
        com_vel_ref[i, :] = np.array([x_real[i, 1], y[i, 1], 0])
        com_acc_ref[i, :] = np.array([x_ddot[i], y_ddot[i], 0])

    if TSID_BOOL:
        if i < n_steps:
            # TSID problem
            tsid.set_com_ref(com_pos_ref[i, :], com_vel_ref[i, :], com_acc_ref[i, :])
            tsid.set_LF_3d_ref(feet_traj['left'][i, :, 0], feet_traj['left'][i, :, 1], feet_traj['left'][i, :, 2])
            tsid.set_RF_3d_ref(feet_traj['right'][i, :, 0], feet_traj['right'][i, :, 1], feet_traj['right'][i, :, 2])
        else:
            tsid.set_com_ref(com_pos_ref[-1, :], np.zeros(3), np.zeros(3))

        HQPData = tsid.formulation.computeProblemData(t, q, v)
        sol = tsid.solver.solve(HQPData)
        if sol.status != 0:
            print("QP problem could not be solved! Error code:", sol.status)
            break
        if np.linalg.norm(v, 2) > 40.0:
            print('Time ', i * params.dt,' Velocities are too high, stop everything! ', np.linalg.norm(v))
            break
        dv = tsid.formulation.getAccelerations(sol)

        com_pos[i,:] = tsid.robot.com(tsid.formulation.data())
        com_vel[i,:] = tsid.robot.com_vel(tsid.formulation.data())
        com_acc[i,:] = tsid.comTask.getAcceleration(dv)
        com_acc_des[i,:] = tsid.comTask.getDesiredAcceleration

        x_LF[i,:], dx_LF, ddx_LF = tsid.get_LF_3d_pos_vel_acc(dv)
        x_RF[i,:], dx_RF, ddx_RF = tsid.get_RF_3d_pos_vel_acc(dv)

        if tsid.formulation.checkContact(tsid.contactRF.name, sol):
            T_RF = tsid.contactRF.getForceGeneratorMatrix
            f_RF[i, :] = T_RF @ tsid.formulation.getContactForce(tsid.contactRF.name, sol)
            if f_RF[i, 2] > 1e-3:
                cop_RF[i, 0] = f_RF[i, 4] / f_RF[i, 2]
                cop_RF[i, 1] = -f_RF[i, 3] / f_RF[i, 2]

        if tsid.formulation.checkContact(tsid.contactLF.name, sol):
            T_LF = tsid.contactRF.getForceGeneratorMatrix
            f_LF[i, :] = T_LF @ tsid.formulation.getContactForce(tsid.contactLF.name, sol)
            if f_LF[i, 2] > 1e-3:
                cop_LF[i, 0] = f_LF[i, 4] / f_LF[i, 2]
                cop_LF[i, 1] = -f_LF[i, 3] / f_LF[i, 2]

        if i % params.PRINT_N == 0:
            print("Time %.3f" % t)
            if tsid.formulation.checkContact(tsid.contactRF.name, sol) and i >= 0:
                print("\tnormal force %s: %.1f" % (tsid.contactRF.name.ljust(20, '.'), f_RF[i, 2]))

            if tsid.formulation.checkContact(tsid.contactLF.name, sol) and i >= 0:
                print("\tnormal force %s: %.1f" % (tsid.contactLF.name.ljust(20, '.'), f_LF[i, 2]))

            print("\ttracking err %s: %.3f" % (tsid.comTask.name.ljust(20, '.'), norm(tsid.comTask.position_error, 2)))
            print("\t||v||: %.3f\t ||dv||: %.3f" % (norm(v, 2), norm(dv)))

        q, v = tsid.integrate_dv(q, v, dv, params.dt)
        t += params.dt

        # Display
        if i % params.DISPLAY_N == 0:
            tsid.display(q)

        q_log[i,:] = q
        v_log[i,:] = v

### PLOTS ###
t = np.arange(0, n_steps + N_post) * params.dt
t_mpc = np.arange(0, walk_time_des) * params.dt_mpc

if PLOT_COM:
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
    ax[0].plot(t, com_acc_des[:,0], color='green', linestyle='dashed', label='$\ddot x_{des}$')
    ax[0].plot(t[:n_steps], com_acc_ref[:,0], color='red', linestyle='dashed', label='$\ddot x_{ref}$')
    ax[0].set_ylabel('CoM x acceleration (m/s^2)')
    ax[0].legend()
    ax[0].grid()
    ax[1].plot(t, com_acc[:,1], color='blue', label='$\ddot y$')
    ax[1].plot(t, com_acc_des[:,1], color='green', linestyle='dashed', label='$\ddot y_{des}$')
    ax[1].plot(t[:n_steps], com_acc_ref[:,1], color='red', linestyle='dashed', label='$\ddot y_{ref}$')
    ax[1].legend()
    ax[1].grid()
    ax[1].set_ylabel('CoM y acceleration (m/s^2)')
    ax[2].plot(t, com_acc[:,2], color='blue', label='$\ddot z$')
    ax[2].plot(t, com_acc_des[:,2], color='green', linestyle='dashed', label='$\ddot z_{des}$')
    ax[2].plot(t[:n_steps], com_acc_ref[:,2], color='red', linestyle='dashed', label='$\ddot z_{ref}$')
    ax[2].legend()
    ax[2].grid()
    ax[2].set_ylabel('CoM z acceleration (m/s^2)')
    ax[2].set_xlabel('Time (s)')

# FEET
if PLOT_FEET:
    for i in range(3):
        plt.figure()
        plt.plot(t, x_RF[:, i], label='x RF ' + str(i))
        plt.plot(t[:n_steps], feet_traj['right'][:,i,0], ':', label='x RF ref ' + str(i))
        plt.plot(t, x_LF[:, i], label='x LF ' + str(i))
        plt.plot(t[:n_steps], feet_traj['left'][:,i,0], ':', label='x LF ref ' + str(i))
        plt.legend()

if PLOT_JOINT_POS:
    for i in range(tsid.robot.nq):
        plt.figure()
        plt.plot(t, q_log[:,i], color='blue', label='q ' + str(i))
        plt.legend()
        plt.grid()
        plt.xlabel('Time (s)')
        plt.ylabel('Joint position')

if PLOT_JOINT_VEL:
    for i in range(tsid.robot.nv):
        plt.figure()
        plt.plot(t, v_log[:,i], color='blue', label='v ' + str(i))
        plt.legend()
        plt.grid()
        plt.xlabel('Time (s)')
        plt.ylabel('Joint velocity')

plt.show()
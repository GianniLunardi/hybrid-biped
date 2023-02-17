import numpy as np
from numpy.linalg import norm
import time
import matplotlib.pyplot as plt
import pinocchio as pin
from hybrid_biped import BipedParams, TsidBiped


data = np.load('../data/walking_ref.npz')
contact_phase = data['contact_phase']

com_acc_ref = np.vstack([data['ddcom_x'], data['ddcom_y'], np.zeros(len(data['ddcom_x']))])
x_RF_ref    = data['x_RF']
dx_RF_ref   = data['dx_RF']
ddx_RF_ref  = data['ddx_RF']
x_LF_ref    = data['x_LF']
dx_LF_ref   = data['dx_LF']
ddx_LF_ref  = data['ddx_LF']

params = BipedParams('../data')
tsid = TsidBiped(params, pin.visualize.GepettoVisualizer)
N = len(data['ddcom_x'])
N_pre = int(params.T_pre/params.dt)
N_post = int(params.T_post/params.dt)

com_pos = np.zeros((3, N+N_post))
com_vel = np.zeros((3, N+N_post))
com_acc = np.zeros((3, N+N_post))
com_acc_des = np.zeros((3, N+N_post))
x_LF   = np.zeros((3, N+N_post))
dx_LF  = np.zeros((3, N+N_post))
ddx_LF = np.zeros((3, N+N_post))
ddx_LF_des = np.zeros((3, N+N_post))
x_RF   = np.zeros((3, N+N_post))
dx_RF  = np.zeros((3, N+N_post))
ddx_RF = np.zeros((3, N+N_post))
ddx_RF_des = np.zeros((3, N+N_post))
f_RF = np.zeros((6, N+N_post))
f_LF = np.zeros((6, N+N_post))
cop_RF = np.zeros((2, N+N_post))
cop_LF = np.zeros((2, N+N_post))
tau    = np.zeros((tsid.robot.na, N+N_post))
q_log  = np.zeros((tsid.robot.nq, N+N_post))
v_log  = np.zeros((tsid.robot.nv, N+N_post))

x_rf = tsid.get_placement_RF().translation
offset = x_rf - x_RF_ref[0, :]
com_pos_ref = np.array([0.03, 0., 0.6])
com_vel_ref = np.zeros(3)
for i in range(N):
    x_RF_ref[i, :] += offset
    x_LF_ref[i, :] += offset

t = -params.T_pre
q, v = tsid.q, tsid.v

for i in range(-N_pre, N + N_post):
    time_start = time.time()

    if i == 0:
        print("Starting to walk (remove contact left foot)")
        tsid.remove_contact_LF()
    elif 0 < i < N - 1:
        if contact_phase[i] != contact_phase[i - 1]:
            print("Time %.3f Changing contact phase from %s to %s" % (t, contact_phase[i - 1], contact_phase[i]))
            if contact_phase[i] == 'left':
                tsid.add_contact_LF()
                tsid.remove_contact_RF()
            else:
                tsid.add_contact_RF()
                tsid.remove_contact_LF()

    if i < 0:
        tsid.set_com_ref(com_pos_ref, 0 * com_vel_ref, 0 * com_acc_ref[:, 0])
    elif i < N:
        tsid.set_com_ref(com_pos_ref, com_vel_ref, com_acc_ref[:, i])
        tsid.set_LF_3d_ref(x_LF_ref[i, :], dx_LF_ref[i, :], ddx_LF_ref[i, :])
        tsid.set_RF_3d_ref(x_RF_ref[i, :], dx_RF_ref[i, :], ddx_RF_ref[i, :])

    HQPData = tsid.formulation.computeProblemData(t, q, v)

    sol = tsid.solver.solve(HQPData)
    if sol.status != 0:
        print("QP problem could not be solved! Error code:", sol.status)
        break
    if norm(v, 2) > 40.0:
        print("Time %.3f Velocities are too high, stop everything!" % t, norm(v))
        break

    if i > 0:
        q_log[:, i] = q
        v_log[:, i] = v
        tau[:, i] = tsid.formulation.getActuatorForces(sol)
    dv = tsid.formulation.getAccelerations(sol)

    if i >= 0:
        com_pos[:, i] = tsid.robot.com(tsid.formulation.data())
        com_vel[:, i] = tsid.robot.com_vel(tsid.formulation.data())
        com_acc[:, i] = tsid.comTask.getAcceleration(dv)
        com_acc_des[:, i] = tsid.comTask.getDesiredAcceleration
        x_LF[:, i], dx_LF[:, i], ddx_LF[:, i] = tsid.get_LF_3d_pos_vel_acc(dv)
        if not tsid.contact_LF_active:
            ddx_LF_des[:, i] = tsid.leftFootTask.getDesiredAcceleration[:3]
        x_RF[:, i], dx_RF[:, i], ddx_RF[:, i] = tsid.get_RF_3d_pos_vel_acc(dv)
        if not tsid.contact_RF_active:
            ddx_RF_des[:, i] = tsid.rightFootTask.getDesiredAcceleration[:3]

        if tsid.formulation.checkContact(tsid.contactRF.name, sol):
            T_RF = tsid.contactRF.getForceGeneratorMatrix
            f_RF[:, i] = T_RF @ tsid.formulation.getContactForce(tsid.contactRF.name, sol)
            if f_RF[2, i] > 1e-3:
                cop_RF[0, i] = f_RF[4, i] / f_RF[2, i]
                cop_RF[1, i] = -f_RF[3, i] / f_RF[2, i]
        if tsid.formulation.checkContact(tsid.contactLF.name, sol):
            T_LF = tsid.contactRF.getForceGeneratorMatrix
            f_LF[:, i] = T_LF @ tsid.formulation.getContactForce(tsid.contactLF.name, sol)
            if f_LF[2, i] > 1e-3:
                cop_LF[0, i] = f_LF[4, i] / f_LF[2, i]
                cop_LF[1, i] = -f_LF[3, i] / f_LF[2, i]

    if i % params.PRINT_N == 0:
        print("Time %.3f" % t)
        if tsid.formulation.checkContact(tsid.contactRF.name, sol) and i >= 0:
            print("\tnormal force %s: %.1f" % (tsid.contactRF.name.ljust(20, '.'), f_RF[2, i]))

        if tsid.formulation.checkContact(tsid.contactLF.name, sol) and i >= 0:
            print("\tnormal force %s: %.1f" % (tsid.contactLF.name.ljust(20, '.'), f_LF[2, i]))

        print("\ttracking err %s: %.3f" % (tsid.comTask.name.ljust(20, '.'), norm(tsid.comTask.position_error, 2)))
        print("\t||v||: %.3f\t ||dv||: %.3f" % (norm(v, 2), norm(dv)))

    q, v = tsid.integrate_dv(q, v, dv, params.dt)
    t += params.dt

    if i % params.DISPLAY_N == 0: tsid.display(q)

    time_spent = time.time() - time_start
    if time_spent < params.dt:
        time.sleep(params.dt - time_spent)

# PLOT STUFF
time = np.arange(0.0, (N + N_post) * params.dt, params.dt)

(f, ax) = plt.subplots(3, 1)
for i in range(3):
    ax[i].plot(time, com_pos[i, :], label='CoM ' + str(i))
    ax[i].plot(time[:N], com_pos_ref[i] * np.ones_like(time[:N]), 'r:', label='CoM Ref ' + str(i))
    ax[i].set_xlabel('Time [s]')
    ax[i].set_ylabel('CoM [m]')
    leg = ax[i].legend()
    leg.get_frame().set_alpha(0.5)

(f, ax) = plt.subplots(3, 1)
for i in range(3):
    ax[i].plot(time, com_vel[i, :], label='CoM Vel ' + str(i))
    ax[i].plot(time[:N], com_vel_ref[i] * np.ones_like(time[:N]), 'r:', label='CoM Vel Ref ' + str(i))
    ax[i].set_xlabel('Time [s]')
    ax[i].set_ylabel('CoM Vel [m/s]')
    leg = ax[i].legend()
    leg.get_frame().set_alpha(0.5)

(f, ax) = plt.subplots(3, 1)
for i in range(3):
    ax[i].plot(time, com_acc[i, :], label='CoM Acc ' + str(i))
    ax[i].plot(time[:N], com_acc_ref[i, :], 'r:', label='CoM Acc Ref ' + str(i))
    ax[i].plot(time, com_acc_des[i, :], 'g--', label='CoM Acc Des ' + str(i))
    ax[i].set_xlabel('Time [s]')
    ax[i].set_ylabel('CoM Acc [m/s^2]')
    leg = ax[i].legend()
    leg.get_frame().set_alpha(0.5)


plt.show()

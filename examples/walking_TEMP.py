import matplotlib.pyplot as plt
from hybrid_biped import *
from numpy import nan


params = BipedParams('../data')
hs = HybridLipm(params.dt, params.tau_hat_0, params)
mpc = LipmMPC(params)
inter = LipmToWbc(params)
tsid = TsidBiped(params)

nb_steps_per_T = int(params.T/params.dt_mpc)
walk_time_first = rolloutLipmDynamics(params.x_hat_0, params.tau_hat_0, mpc.dt_mpc, params)
walk_time_des = (params.nb_steps - 1) * nb_steps_per_T + walk_time_first

N = walk_time_des * params.ratio
N_post = int(params.T_post / params.dt)
N_tot = N + N_post

# Arrays
com_pos = np.empty((N_tot, 3))*nan
com_vel = np.empty((N_tot, 3))*nan
com_acc = np.empty((N_tot, 3))*nan
x_LF = np.empty((N_tot, 3))*nan
x_RF = np.empty((N_tot, 3))*nan
com_pos_ref = np.empty((N, 3))*nan
com_vel_ref = np.empty((N, 3))*nan
com_acc_ref = np.empty((N, 3))*nan

# Foot steps
foot_steps = np.vstack([[0., 0.08],
                        np.copy(mpc.foot_steps_des)])
feet_traj = dict()
feet_traj['left'] = np.empty((N, 3, 3))*nan
feet_traj['right'] = np.empty((N, 3, 3))*nan


x_lf   = tsid.get_placement_LF().translation
offset = x_lf - foot_steps[0,1]

t = 0

# Use inverse geometry !!!
import numpy as np
from quadprog import solve_qp
import lmpc_walking.second_order as lw


class LinearMPC:

    def __init__(self, params):
        self.params = params
        self.dt_mpc = params.dt_mpc
        self.omega = params.omega
        self.nb_mpc_per_T = int(round(params.T/params.dt_mpc))
        self.horizon = 2 * self.nb_mpc_per_T

        # Cost weights
        self.alpha = params.alpha
        self.beta = params.beta
        self.gamma = params.gamma

        # Estimated initial state
        self.updateCurrentState(params.tau_hat_0, params.x_hat_0, params.y_hat_0)

        # Recursive matrices for the forward dynamics
        self.P_ps, self.P_vs, self.P_pu, self.P_vu = \
            lw.compute_recursive_matrices(self.dt_mpc, self.omega, self.horizon)

        # Compute desired and planned walking time
        nb_planned_steps = params.nb_steps + 2
        self.walk_time_des = params.nb_steps * self.nb_mpc_per_T
        walk_time_planned = nb_planned_steps * self.nb_mpc_per_T

        # Compute the desired foot steps with manual foot placement
        self.foot_steps_des = lw.manual_foot_placement(params.foot_step_0,
                                                       params.step_length,
                                                       params.nb_steps)
        Z_ref_des = lw.create_CoP_trajectory(params.nb_steps, self.foot_steps_des,
                                             self.walk_time_des, self.nb_mpc_per_T)

        self.Z_ref_planned = np.zeros((walk_time_planned, 2))
        self.Z_ref_planned[0:self.walk_time_des,:] = Z_ref_des
        self.Z_ref_planned[self.walk_time_des:walk_time_planned,:] = Z_ref_des[self.walk_time_des-1,:]

        self.Z_ref_k = self.Z_ref_planned[0:self.horizon,:]

        self.j = 0
        self.horizon_data = []

    def updateCurrentState(self, tau_hat, x_hat, y_hat):
        """ Update the initial state of the MPC step from the current values of the HS and TSID """
        self.tau_hat_k = tau_hat
        self.x_hat_k = x_hat
        self.y_hat_k = y_hat

    def step(self):
        horizon_data = {}
        # Compute the objective terms
        Q_k, p_k = lw.compute_objective_terms(self.alpha, self.beta, self.gamma,
                                              self.params.T, self.nb_mpc_per_T, self.horizon,
                                              self.params.step_length, self.params.step_width,
                                              self.P_ps, self.P_pu, self.P_vs, self.P_vu,
                                              self.x_hat_k, self.y_hat_k, self.Z_ref_k)
        # Add ZMP constraints
        A_zmp, b_zmp = lw.add_ZMP_constraints(self.horizon, self.params.foot_length, self.params.foot_width,
                                              self.Z_ref_k, self.x_hat_k, self.y_hat_k)

        # Solve QP
        current_U = solve_qp(Q_k, -p_k, A_zmp.T, b_zmp)[0]

        # Recursive dynamics
        X_k, Y_k = lw.compute_recursive_dynamics(self.P_ps, self.P_vs, self.P_pu, self.P_vu,
                                                 self.horizon, self.x_hat_k, self.y_hat_k, current_U)

        # Save the CoM and CoP trajectories (only y-axis)
        self.y_next = Y_k[0,:]
        self.U_y_current = current_U[self.horizon]
        self.Z_y = self.Z_ref_k[0,1]

        horizon_data['zmp_reference'] = self.Z_ref_k
        horizon_data['X_k'] = X_k
        horizon_data['Y_k'] = Y_k
        horizon_data['Z_x_k'] = current_U[:self.horizon]
        horizon_data['Z_y_k'] = current_U[self.horizon:]

        # Update the Z_ref_k
        self.Z_ref_k = self.Z_ref_planned[self.j+1:self.j+1+self.horizon,:]
        self.j += 1

        self.horizon_data.append(horizon_data)
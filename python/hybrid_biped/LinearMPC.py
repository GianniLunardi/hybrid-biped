import numpy as np
from quadprog import solve_qp
import lmpc_walking.second_order as lw
from hybrid_biped.HybridLIPM import rolloutLipmDynamics


class LipmMPC:
    def __init__(self, params):
        self.params = params
        self.dt_mpc = params.dt_mpc
        self.omega = params.omega
        self.nb_steps_per_T = int(params.T/params.dt_mpc)
        self.horizon = 2 * self.nb_steps_per_T

        # Cost weights
        self.alpha = params.alpha
        self.beta = params.beta
        self.gamma = params.gamma

        # Estimated initial state
        self.updateCurrentState(params.tau_hat_0, params.x_hat_0, params.y_hat_0)

        # Recursive matrices for the forward dynamics
        self.P_ps, self.P_vs, self.P_pu, self.P_vu = \
            lw.compute_recursive_matrices(self.dt_mpc, self.omega, self.horizon)

        self.foot_steps_des = lw.manual_foot_placement(self.params.foot_step_0,
                                                       self.params.step_length,
                                                       self.params.nb_steps + 6)

    def updateCurrentState(self, tau_hat, x_hat, y_hat):
        """ Update the initial state of the MPC step from the current values of the HS and TSID """
        self.tau_hat_k = tau_hat
        self.x_hat_k = x_hat
        self.y_hat_k = y_hat

    def step(self):
        # Rollout the HS dynamics to compute the time needed for the first foot step
        nb_left = rolloutLipmDynamics(self.x_hat_k, self.tau_hat_k, self.dt_mpc, self.params)

        # Plan the CoP reference
        self.Z_ref_k, fsteps = lw.varying_CoP_trajectory(self.foot_steps_des, self.horizon,
                                                         self.nb_steps_per_T, nb_left)
        self.foot_steps_des = np.copy(fsteps)

        # NB: Probably this will work as it is
        Q_k, p_k = lw.compute_objective_terms(self.alpha, self.beta, self.gamma,
                                              self.params.T, self.nb_steps_per_T, self.horizon,
                                              self.params.step_length, self.params.step_width,
                                              self.P_ps, self.P_pu, self.P_vs, self.P_vu,
                                              self.x_hat_k, self.y_hat_k, self.Z_ref_k)

        A_zmp, b_zmp = lw.add_ZMP_constraints(self.horizon, self.params.foot_length, self.params.foot_width,
                                              self.Z_ref_k, self.x_hat_k, self.y_hat_k)

        current_U = solve_qp(Q_k, -p_k, A_zmp.T, b_zmp)[0]

        X_k, Y_k = lw.compute_recursive_dynamics(self.P_ps, self.P_vs, self.P_pu, self.P_vu,
                                                 self.horizon, self.x_hat_k, self.y_hat_k, current_U)

        # Save the CoM and CoP 
        self.y_next = Y_k[0,:]
        self.U_y_current = current_U[self.horizon]
        self.Z_y = self.Z_ref_k[0,1]


import numpy as np
from quadprog import solve_qp
import lmpc_walking.second_order as lw
from hybrid_biped.HybridLIPM import HybridLipm


class LipmMPC:

    def __init__(self, x_hat_0, y_hat_0, params, closed_loop = False):
        self.closed_loop = closed_loop

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
        self.x_hat_k = x_hat_0
        self.y_hat_k = y_hat_0
        self.tau_hat_k = params.tau0

        # Recursive matrices for the forward dynamics
        self.P_ps, self.P_vs, self.P_pu, self.P_vu = \
            lw.compute_recursive_matrices(self.dt_mpc, self.omega, self.horizon)

        walk_time_first = self.rolloutLipmDynamics(self.params.x0[0], self.params.tau0)
        self.walk_time_des = (self.params.nb_stpes_des - 1) * self.nb_steps_per_T + walk_time_first

        self.foot_steps_des = lw.manual_foot_placement(self.params.foot_step_0,
                                                       self.params.step_length,
                                                       self.params.nb_stpes_des + 3)

        self.Y_total = np.zeros((self.walk_time_des+1, 2))
        self.Z_y_ref = np.zeros(self.walk_time_des)
        self.Z_y_total = np.zeros(self.walk_time_des)

        # Save initial state
        self.Y_total[0,:] = y_hat_0

    def updateCurrentState(self, tau_hat, x_hat, y_hat):
        """ Update the initial state of the MPC step from the current values of the HS and TSID """
        self.tau_hat_k = tau_hat
        self.x_hat_k = x_hat
        if self.closed_loop:
            self.y_hat_k = y_hat

    def rolloutLipmDynamics(self, x0, tau0):
        """ Roll-out of the Hybrid LIPM dynamics, considering the same time step of the MPC """
        i = 0
        hs_lipm = HybridLipm(self.dt_mpc, tau0, self.params)
        x_hb = x0

        while hs_lipm.tau < self.horizon:
            if x_hb[0] >= self.params.r_bar:
                return i
            x_hb_ref = hs_lipm.referenceWithTimer()
            eps = x_hb - x_hb_ref
            u = hs_lipm.saturatedFb(eps)
            x_hb = hs_lipm.flow(x_hb, u)
            i += 1

    def step(self, i):
        # TODO remove the dependency from i
        # Rollout the HS dynamics to compute the time needed for the first foot step
        nb_HS_steps = self.rolloutLipmDynamics(self.x_hat_k, self.tau_hat_k)

        # Plan the CoP reference
        self.Z_ref_k, fsteps = lw.varying_CoP_trajectory(self.foot_steps_des, self.horizon,
                                                         self.nb_steps_per_T, nb_HS_steps)
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

        # Update current state for the next iteration for Open Loop pipeline
        if self.closed_loop is False:
            self.y_hat_k = Y_k[0,:]

        # Save the CoM and CoP (only y directions are interesting
        self.Y_total[i+1,:] = Y_k[0,:]
        self.Z_y_total[i] = current_U[self.horizon]
        self.Z_y_ref[i] = self.Z_ref_k[0,1]


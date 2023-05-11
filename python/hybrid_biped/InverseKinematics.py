import numpy as np
from numpy import nan
from numpy.linalg import norm
import os
import subprocess
import time
import pinocchio as pin
from orc.utils.robot_wrapper import RobotWrapper


class InverseKinematics:
    def __init__(self, params, logger = False, x_LF = None, x_RF = None):
        self.logger = logger
        self.robot = RobotWrapper.BuildFromURDF(params.urdf, [params.path], pin.JointModelFreeFlyer())
        self.model = self.robot.model
        self.id_lf = self.model.getFrameId(params.lf_frame_name)
        self.id_rf = self.model.getFrameId(params.rf_frame_name)
        pin.loadReferenceConfigurations(self.model, params.srdf, False)
        q = self.model.referenceConfigurations["half_sitting"]
        H_rf_ref = self.robot.framePlacement(q, self.id_rf)
        q[2] -= H_rf_ref.translation[2] - params.lz
        self.q0 = q

        launched = subprocess.getstatusoutput("ps aux |grep 'gepetto-gui'|grep -v 'grep'|wc -l")
        if int(launched[1]) == 0:
            os.system('gepetto-gui &')
        time.sleep(1)
        self.viz = pin.visualize.GepettoVisualizer(self.robot.model, self.robot.collision_model,
                                                   self.robot.visual_model)
        self.viz.initViewer(loadModel=True)
        self.viz.displayCollisions(False)
        self.viz.displayVisuals(True)
        self.viz.display(self.q0)

        self.gui = self.viz.viewer.gui
        self.gui.addFloor('world/floor')
        self.gui.setLightingMode('world/floor', 'OFF')

        # Desired E-Es positions
        self.com_pos_des = np.array([params.x_hat_0[0], params.y_hat_0[0], params.h_com])
        if x_LF is None or x_RF is None:
            x_LF = params.x_LF_trial
            x_RF = params.x_RF_trial
        self.x_LF = np.hstack([x_LF, np.array([0., 0., 0.])])
        self.x_RF = np.hstack([x_RF, np.array([0., 0., 0.])])
        self.x_des = np.hstack([self.x_LF, self.x_RF, self.com_pos_des])

        # Parameters
        self.beta = 0.1
        self.gamma = 0
        self.hessian_regu = 1e-1
        self.gradient_threshold = 2*1e-4
        self.line_search = 1
        self.PRINT_N = 1
        self.DISPLAY_N = 1
        self.max_iter = 100

    def compute_next_foot_position(self, q, frame_id):
        return self.robot.framePlacement(q, frame_id).translation

    def compute_next_foot_pose(self, q, framed_id):
        H = self.robot.framePlacement(q, framed_id)
        return np.hstack([H.translation,
                          pin.rpy.matrixToRpy(H.rotation)])

    def compute_next_com_position(self, q):
        return self.robot.com(q)

    def inverse_geometry_step(self, q, x, J, i):
        e = self.x_des - x
        cost = norm(e)

        # Newton method
        nv = J.shape[1]
        J_reg = J.T.dot(J) + self.hessian_regu * np.eye(nv)
        gradient = J.T.dot(e)
        delta_q = np.linalg.inv(J_reg).dot(gradient)

        # if gradient is null you are done
        grad_norm = norm(gradient)
        if grad_norm < self.gradient_threshold:
            print("Terminate because gradient is (almost) zero:", grad_norm)
            print("Problem solved after %d iterations with error %f" % (i, norm(e)))
            return None

        if self.line_search:
            # back-tracking line search
            alpha = 1.0
            iter_line_search = 0
            while True:
                q_next = pin.integrate(self.model, q, alpha * delta_q)
                self.robot.computeJointJacobians(q_next)
                self.robot.framesForwardKinematics(q_next)
                x_new = np.hstack([self.compute_next_foot_pose(q_next, self.id_lf),
                                   self.compute_next_foot_pose(q_next, self.id_rf),
                                   self.compute_next_com_position(q_next)])
                cost_new = norm(self.x_des - x_new)
                if cost_new < (1.0 - alpha * self.gamma) * cost:
                    if self.logger:
                        print("Backtracking line search converged with log(alpha)=%.1f" % np.log10(alpha))
                    self.J = J
                    self.q = q
                    break
                else:
                    alpha *= self.beta
                    iter_line_search += 1
                    if iter_line_search == self.max_iter:
                        print('Backtracking line search could not converge')
                        break
        else:
            q_next = pin.integrate(self.model, q, delta_q)
        if i % self.PRINT_N == 0 and self.logger:
            print("Iteration %d, ||x_des-x||=%f, norm(gradient)=%f" % (i, norm(e), grad_norm))

        return q_next, cost, grad_norm

    def compute_inverse_geometry(self):
        q = np.empty((self.max_iter+1, self.robot.nq))*nan  # joint angles
        x = np.empty((self.max_iter, 15))*nan               # frame position --> lf pose (6) + rf pose (6) + CoM position (3)
        cost = np.empty(self.max_iter)*nan
        grad_norm = np.empty(self.max_iter)*nan         # gradient norm
        q[0,:] = self.q0

        for i in range(self.max_iter):
            self.robot.computeJointJacobians(q[i,:])
            self.robot.framesForwardKinematics(q[i,:])
            H_lf = self.robot.framePlacement(q[i,:], self.id_lf)
            H_rf = self.robot.framePlacement(q[i,:], self.id_rf)
            # EE positions
            # LF pose
            x[i,:3] = H_lf.translation
            x[i,3:6] = pin.rpy.matrixToRpy(H_lf.rotation)
            # RF pose
            x[i,6:9] = H_rf.translation
            x[i,9:12] = pin.rpy.matrixToRpy(H_rf.rotation)
            # CoM position
            x[i,-3:] = self.robot.com(q[i,:])
            if self.logger:
                print('Iter : ', i, 'LF: ', x[i,:3], 'RF: ', x[i,6:9], 'CoM: ', x[i,-3:])
            # Jacobians
            J_lf = self.robot.frameJacobian(q[i,:], self.id_lf)
            J_rf = self.robot.frameJacobian(q[i,:], self.id_rf)
            J_com = self.robot.Jcom(q[i,:])
            J = np.vstack([J_lf, J_rf, J_com[:3,:]])
            result = self.inverse_geometry_step(q[i,:], x[i,:], J, i)

            if result is None:
                break
            else:
                q_next, c, g = result
                q[i+1,:] = q_next
                cost[i] = c
                grad_norm[i] = g

            if i % self.DISPLAY_N == 0:
                self.viz.display(q[i,:])
                time.sleep(0.1)

    def compute_state_velocity(self, x_dot_lf, x_dot_rf, x_dot_com):
        """ Compute the joint and base velocities from the frame linear velocities"""
        x_dot_lf = np.hstack([x_dot_lf, np.zeros(3)])
        x_dot_rf = np.hstack([x_dot_rf, np.zeros(3)])
        x_dot = np.hstack([x_dot_lf, x_dot_rf, x_dot_com])
        J_pseudo = np.linalg.pinv(self.J)
        return J_pseudo.dot(x_dot)




import numpy as np
from numpy.linalg import norm
import os
import subprocess
import time
import pinocchio as pin
import example_robot_data
from orc.utils.robot_wrapper import RobotWrapper


class InverseGeometry:
    def __init__(self, params, logger = False):
        self.logger = logger
        rb = example_robot_data.load('romeo')
        self.robot = RobotWrapper(rb.model, rb.collision_model, rb.visual_model)
        self.model = self.robot.model
        self.id_lf = self.model.getFrameId(params.lf_frame_name)
        self.id_rf = self.model.getFrameId(params.rf_frame_name)
        pin.loadReferenceConfigurations(self.model, params.srdf, False)
        q = self.model.referenceConfigurations["half_sitting"]
        H_rf_ref = self.robot.framePlacement(q, self.id_rf)
        q[2] -= H_rf_ref.translation[2] - params.lz
        self.q0 = q
        # print('LEFT tran: ', self.robot.data.oMf[self.id_lf].translation)
        # print('RIGHT tran: ', self.robot.data.oMf[self.id_rf].translation)
        # print('LEFT rot: ', pin.rpy.matrixToRpy(self.robot.data.oMf[self.id_lf].rotation))
        # print('RIGHT rot: ', pin.rpy.matrixToRpy(self.robot.data.oMf[self.id_rf].rotation))

        launched = subprocess.getstatusoutput("ps aux |grep 'gepetto-gui'|grep -v 'grep'|wc -l")
        if int(launched[1]) == 0:
            os.system('gepetto-gui &')
        time.sleep(1)
        self.viz = pin.visualize.GepettoVisualizer(rb.model, rb.collision_model,
                                                   rb.visual_model)
        self.viz.initViewer(loadModel=True)
        self.viz.displayCollisions(False)
        self.viz.displayVisuals(True)
        self.viz.display(self.q0)

        self.gui = self.viz.viewer.gui
        # self.gui.setCameraTransform(0, params.CAMERA_TRANSFORM)
        self.gui.addFloor('world/floor')
        self.gui.setLightingMode('world/floor', 'OFF')

        # Desired E-Es positions
        self.com_pos_des = np.array([params.x_hat_0[0], params.y_hat_0[0], params.h_com])
        self.x_LF = np.array([0., 0.096, 0.07, 0., 0., 0.])
        self.x_RF = np.array([-0.0, -0.096, 0.12, 0., 0., -0.2])
        self.x_des = np.hstack([self.x_LF, self.x_RF, self.com_pos_des])

        # Parameters
        self.beta = 0.1
        self.gamma = 0
        self.hessian_regu = 1e-1
        self.gradient_threshold = 5*1e-4
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
        H = J.T.dot(J) + self.hessian_regu * np.eye(nv)
        gradient = J.T.dot(e)
        delta_q = np.linalg.inv(H).dot(gradient)

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
                    print("Backtracking line search converged with log(alpha)=%.1f" % np.log10(alpha))
                    break
                else:
                    alpha *= self.beta
                    iter_line_search += 1
                    if iter_line_search == self.max_iter:
                        print('Backtracking line search could not converge')
                        break
        else:
            q_next = pin.integrate(self.model, q, delta_q)
        if i % self.PRINT_N == 0:
            print("Iteration %d, ||x_des-x||=%f, norm(gradient)=%f" % (i, norm(e), grad_norm))

        return q_next, cost, grad_norm




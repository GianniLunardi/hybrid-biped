import time
import numpy as np
import pinocchio as pin
from numpy import nan
from python.hybrid_biped import BipedParams, InverseKinematics


# class Config:
#     def __init__(self):
#         self.beta = 0.1
#         self.gamma = 0
#         self.gradient_threshold = 1e-6
#         self.line_search = 1
#         self.PRINT_N = 1
#         self.DISPLAY_N = 1
#
# def inverse_geometry_step(model, data, q, x, x_des, J, regu, i, N, robot, frame_id, conf):
#     e = x_des - x
#     cost = norm(e)
#
#     # gradient descent
#     #    q[:,i+1] = q[:,i] + alpha*J.T.dot(e)
#
#     # Newton method
#     nv = J.shape[1]
#     B = J.T.dot(J) + regu * np.eye(nv)  # approximate regularized Hessian
#     gradient = J.T.dot(e)  # gradient
#     delta_q = inv(B).dot(gradient)
#     q_next = pin.integrate(model, q , delta_q)
#
#     # if gradient is null you are done
#     grad_norm = norm(gradient)
#     if grad_norm < conf.gradient_threshold:
#         print("Terminate because gradient is (almost) zero:", grad_norm)
#         print("Problem solved after %d iterations with error %f" % (i, norm(e)))
#         return None
#
#     if not conf.line_search:
#         q_next = pin.integrate(model, q , delta_q)
#     else:
#         # back-tracking line search
#         alpha = 1.0
#         iter_line_search = 0
#         while True:
#             q_next = pin.integrate(model, q, alpha * delta_q)
#             pin.computeJointJacobians(model, data, q_next)
#             pin.framesForwardKinematics(model, data, q_next)
#             x_new = data.oMf[frame_id].translation
#             cost_new = norm(x_des - x_new)
#             if cost_new < (1.0 - alpha * conf.gamma) * cost:
#                 #            print("Backtracking line search converged with log(alpha)=%.1f"%np.log10(alpha))
#                 break
#             else:
#                 alpha *= conf.beta
#                 iter_line_search += 1
#                 if iter_line_search == N:
#                     print("Backtracking line search could not converge. log(alpha)=%.1f" % np.log10(alpha))
#                     break
#
#     if i % conf.PRINT_N == 0:
#         print("Iteration %d, ||x_des-x||=%f, norm(gradient)=%f" % (i, norm(e), grad_norm))
#
#     return q_next, cost, grad_norm
#
#
# # MAIN
# params = BipedParams('../data')
# conf = Config()
# tsid = TsidBiped(params)
#
# N = 100
# q = np.empty((N+1, tsid.robot.nq))*nan
# x = np.empty((N, 3))*nan                        # LF, RF foot in R^3, CoM in R^3
# cost = np.empty(N)*nan
# grad_norm = np.empty(N)*nan
# q[0,:] = tsid.q
#
# com_pos_des = np.array([0.1, 0.05, 0.6])
# x_LF = np.array([-0.05, 0.05, 0])
# x_RF = np.array([-0.05, -0.05, 0.2])
# x_des = x_RF #np.hstack([x_LF, x_RF, com_pos_des])
# regu = 1e-1
# iter_line_search = 0
# RF_id = tsid.model.getFrameId(params.rf_frame_name)
#
# robot = example_robot_data.load('romeo')
# rmodel = robot.model
# rdata = rmodel.createData()
#
# for i in range(N):
#     pin.computeJointJacobians(rmodel, rdata, q[i,:])
#     pin.framesForwardKinematics(rmodel, rdata, q[i,:])
#     # J_LF = tsid.robot.frameJacobian(q[i,:], tsid.model.getFrameId(params.lf_frame_name))
#
#     J_RF = pin.frameJacobian(rmodel, rdata, q[i,:], RF_id)
#     # H = pin.framePlacement(q[:,i], RF_id)
#
#     # J_com = robot.Jcom(q[i,:])
#     J = J_RF[:3,:]
#     x[i,:] = rdata.oMf[RF_id].translation
#
#     result = inverse_geometry_step(rmodel, rdata, q[i,:], x[i,:], x_des, J, regu, i, N, tsid.robot, RF_id, conf)
#
#     if result is None:
#         break
#     else:
#         q_next, c, g = result
#         q[i + 1,:] = q_next
#         cost[i] = c
#         grad_norm[i] = g
#
#         # display current configuration in viewer
#     if i % conf.DISPLAY_N == 0:
#         tsid.display(q[i,:])
#         time.sleep(0.1)
#
#     if iter_line_search == N:
#         break

params = BipedParams()
ig = InverseKinematics(params, logger=True)
ig.compute_inverse_geometry()
# robot = ig.robot
# N = ig.max_iter
# q = np.empty((N+1, robot.nq))*nan  # joint angles
# x = np.empty((N, 15))*nan          # frame position --> lf pose (6) + rf pose (6) + CoM position (3)
# cost = np.empty(N)*nan
# grad_norm = np.empty(N)*nan         # gradient norm
# q[0,:] = ig.q0
#
# for i in range(N):
#     robot.computeJointJacobians(q[i,:])
#     robot.framesForwardKinematics(q[i,:])
#     H_lf = robot.framePlacement(q[i,:], ig.id_lf)
#     H_rf = robot.framePlacement(q[i,:], ig.id_rf)
#     # EE positions
#     # LF pose
#     x[i,:3] = H_lf.translation
#     x[i,3:6] = pin.rpy.matrixToRpy(H_lf.rotation)
#     # RF pose
#     x[i, 6:9] = H_rf.translation
#     x[i, 9:12] = pin.rpy.matrixToRpy(H_rf.rotation)
#     # CoM position
#     x[i,-3:] = robot.com(q[i,:])
#     print('Iter : ', i, 'LF: ', x[i,:3], 'RF: ', x[i,6:9])
#     # Jacobians
#     J_lf = robot.frameJacobian(q[i,:], ig.id_lf)
#     J_rf = robot.frameJacobian(q[i,:], ig.id_rf)
#     J_com = robot.Jcom(q[i,:])
#     J = np.vstack([J_lf, J_rf, J_com[:3,:]])
#     result = ig.inverse_geometry_step(q[i,:], x[i,:], J, i)
#
#     if result is None:
#         break
#     else:
#         q_next, c, g = result
#         q[i+1,:] = q_next
#         cost[i] = c
#         grad_norm[i] = g
#
#     if i % ig.DISPLAY_N == 0:
#         ig.viz.display(q[i,:])
#         time.sleep(0.1)
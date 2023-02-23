import os
import subprocess
import time
import numpy as np
import pinocchio as pin
import tsid
from hybrid_biped import rolloutLipmDynamics


class Storage:
    def __init__(self, parent_class):
        self.parent_class = parent_class
        self.com_pos = np.zeros(3)
        self.com_vel = np.zeros(3)
        self.com_acc = np.zeros(3)
        self.com_acc_des = np.zeros(3)

        self.x_LF = np.zeros(3)
        self.dx_LF = np.zeros(3)
        self.ddx_LF = np.zeros(3)
        self.ddx_LF_des = np.zeros(3)

        self.x_RF = np.zeros(3)
        self.dx_RF = np.zeros(3)
        self.ddx_RF = np.zeros(3)
        self.ddx_RF_des = np.zeros(3)

    def save_com_references(self, com_pos_ref, com_vel_ref, com_acc_ref):
        self.parent_class.com_pos_ref = com_pos_ref.copy()
        self.parent_class.com_vel_ref = com_vel_ref.copy()
        self.parent_class.com_acc_ref = com_acc_ref.copy()

    def save_feet_references(self, x_LF_ref, x_RF_ref):
        self.parent_class.x_LF_ref = x_LF_ref[:,0].copy()
        self.parent_class.dx_LF_ref = x_LF_ref[:,1].copy()
        self.parent_class.ddx_LF_ref = x_LF_ref[:,2].copy()

        self.parent_class.x_RF_ref = x_RF_ref[:,0].copy()
        self.parent_class.dx_RF_ref = x_RF_ref[:,1].copy()
        self.parent_class.ddx_RF_ref = x_RF_ref[:,2].copy()

class TsidBiped:
    """ Standard TSID formulation for a biped robot standing on its rectangular feet.
        - Center of mass task
        - Postural task
        - 6d rigid contact constraint for both feet
        - Regularization task for contact forces
    """

    def __init__(self, params, viewer=pin.visualize.GepettoVisualizer, simu = True):
        self.params = params
        self.dt = params.dt
        self.ratio = params.ratio
        self.prev_phase = params.contact_phase_0
        self.storage = Storage(self)
        self.simu = simu
        self.count = 0 # time counter
        # Initialize the MPC and the interpolator class

        self.robot = tsid.RobotWrapper(params.urdf, [params.path], pin.JointModelFreeFlyer(), False)
        robot = self.robot
        self.model = robot.model()
        pin.loadReferenceConfigurations(self.model, params.srdf, False)
        self.q0 = q = self.model.referenceConfigurations["half_sitting"]
        v = np.zeros(robot.nv)

        assert self.model.existFrame(params.rf_frame_name)
        assert self.model.existFrame(params.lf_frame_name)

        formulation = tsid.InverseDynamicsFormulationAccForce("tsid", robot, False)
        formulation.computeProblemData(0.0, q, v)
        data = formulation.data()
        contact_Point = np.ones((3, 4)) * (-params.lz)
        contact_Point[0, :] = [-params.lxn, -params.lxn, params.lxp, params.lxp]
        contact_Point[1, :] = [-params.lyn, params.lyp, -params.lyn, params.lyp]

        contactRF = tsid.Contact6d("contact_rfoot", robot, params.rf_frame_name, contact_Point,
                                   params.contactNormal, params.mu, params.fMin, params.fMax)
        contactRF.setKp(params.kp_contact * np.ones(6))
        contactRF.setKd(2.0 * np.sqrt(params.kp_contact) * np.ones(6))
        self.RF = robot.model().getFrameId(params.rf_frame_name)
        H_rf_ref = robot.framePosition(data, self.RF)

        # modify initial robot configuration so that foot is on the ground (z=0)
        q[2] -= H_rf_ref.translation[2] - params.lz
        formulation.computeProblemData(0.0, q, v)
        data = formulation.data()
        H_rf_ref = robot.framePosition(data, self.RF)
        contactRF.setReference(H_rf_ref)
        if params.w_contact >= 0.0:
            formulation.addRigidContact(contactRF, params.w_forceRef, params.w_contact, 1)
        else:
            formulation.addRigidContact(contactRF, params.w_forceRef)

        contactLF = tsid.Contact6d("contact_lfoot", robot, params.lf_frame_name, contact_Point,
                                   params.contactNormal, params.mu, params.fMin, params.fMax)
        contactLF.setKp(params.kp_contact * np.ones(6))
        contactLF.setKd(2.0 * np.sqrt(params.kp_contact) * np.ones(6))
        self.LF = robot.model().getFrameId(params.lf_frame_name)
        H_lf_ref = robot.framePosition(data, self.LF)
        contactLF.setReference(H_lf_ref)
        if params.w_contact >= 0.0:
            formulation.addRigidContact(contactLF, params.w_forceRef, params.w_contact, 1)
        else:
            formulation.addRigidContact(contactLF, params.w_forceRef)

        comTask = tsid.TaskComEquality("task-com", robot)
        comTask.setKp(params.kp_com * np.ones(3))
        comTask.setKd(2.0 * np.sqrt(params.kp_com) * np.ones(3))
        formulation.addMotionTask(comTask, params.w_com, 1, 0.0)

        copTask = tsid.TaskCopEquality("task-cop", robot)
        formulation.addForceTask(copTask, params.w_cop, 1, 0.0)

        amTask = tsid.TaskAMEquality("task-am", robot)
        amTask.setKp(params.kp_am * np.array([1., 1., 0.]))
        amTask.setKd(2.0 * np.sqrt(params.kp_am * np.array([1., 1., 0.])))
        formulation.addMotionTask(amTask, params.w_am, 1, 0.)
        sampleAM = tsid.TrajectorySample(3)
        amTask.setReference(sampleAM)

        postureTask = tsid.TaskJointPosture("task-posture", robot)
        postureTask.setKp(params.kp_posture * params.gain_vector)
        postureTask.setKd(2.0 * np.sqrt(params.kp_posture * params.gain_vector))
        postureTask.setMask(params.masks_posture)
        formulation.addMotionTask(postureTask, params.w_posture, 1, 0.0)

        self.leftFootTask = tsid.TaskSE3Equality("task-left-foot", self.robot, self.params.lf_frame_name)
        self.leftFootTask.setKp(self.params.kp_foot * np.ones(6))
        self.leftFootTask.setKd(2.0 * np.sqrt(self.params.kp_foot) * np.ones(6))
        self.trajLF = tsid.TrajectorySE3Constant("traj-left-foot", H_lf_ref)
        formulation.addMotionTask(self.leftFootTask, self.params.w_foot, 1, 0.0)

        self.rightFootTask = tsid.TaskSE3Equality("task-right-foot", self.robot, self.params.rf_frame_name)
        self.rightFootTask.setKp(self.params.kp_foot * np.ones(6))
        self.rightFootTask.setKd(2.0 * np.sqrt(self.params.kp_foot) * np.ones(6))
        self.trajRF = tsid.TrajectorySE3Constant("traj-right-foot", H_rf_ref)
        formulation.addMotionTask(self.rightFootTask, self.params.w_foot, 1, 0.0)

        self.tau_max = params.tau_max_scaling * robot.model().effortLimit[-robot.na:]
        self.tau_min = -self.tau_max
        actuationBoundsTask = tsid.TaskActuationBounds("task-actuation-bounds", robot)
        actuationBoundsTask.setBounds(self.tau_min, self.tau_max)
        if params.w_torque_bounds > 0.0:
            formulation.addActuationTask(actuationBoundsTask, params.w_torque_bounds, 0, 0.0)

        jointBoundsTask = tsid.TaskJointBounds("task-joint-bounds", robot, params.dt)
        self.v_max = params.v_max_scaling * robot.model().velocityLimit[-robot.na:]
        self.v_min = -self.v_max
        jointBoundsTask.setVelocityBounds(self.v_min, self.v_max)
        if params.w_joint_bounds > 0.0:
            formulation.addMotionTask(jointBoundsTask, params.w_joint_bounds, 0, 0.0)

        com_ref = robot.com(data)
        self.trajCom = tsid.TrajectoryEuclidianConstant("traj_com", com_ref)
        self.sample_com = self.trajCom.computeNext()

        q_ref = q[7:]
        self.trajPosture = tsid.TrajectoryEuclidianConstant("traj_joint", q_ref)
        postureTask.setReference(self.trajPosture.computeNext())

        self.sampleLF = self.trajLF.computeNext()
        self.sample_LF_pos = self.sampleLF.pos()
        self.sample_LF_vel = self.sampleLF.vel()
        self.sample_LF_acc = self.sampleLF.acc()

        self.sampleRF = self.trajRF.computeNext()
        self.sample_RF_pos = self.sampleRF.pos()
        self.sample_RF_vel = self.sampleRF.vel()
        self.sample_RF_acc = self.sampleRF.acc()

        self.solver = tsid.SolverHQuadProgFast("qp solver")
        self.solver.resize(formulation.nVar, formulation.nEq, formulation.nIn)

        self.comTask = comTask
        self.copTask = copTask
        self.amTask = amTask
        self.postureTask = postureTask
        self.contactRF = contactRF
        self.contactLF = contactLF
        self.actuationBoundsTask = actuationBoundsTask
        self.jointBoundsTask = jointBoundsTask
        self.formulation = formulation
        self.q = q
        self.v = v

        self.contact_LF_active = True
        self.contact_RF_active = True

        if viewer:
            self.robot_display = pin.RobotWrapper.BuildFromURDF(params.urdf, [params.path], pin.JointModelFreeFlyer())
            if viewer == pin.visualize.GepettoVisualizer:
                import gepetto.corbaserver
                launched = subprocess.getstatusoutput("ps aux |grep 'gepetto-gui'|grep -v 'grep'|wc -l")
                if int(launched[1]) == 0:
                    os.system('gepetto-gui &')
                time.sleep(1)
                self.viz = viewer(self.robot_display.model, self.robot_display.collision_model,
                                  self.robot_display.visual_model)
                self.viz.initViewer(loadModel=True)
                self.viz.displayCollisions(False)
                self.viz.displayVisuals(True)
                self.viz.display(q)

                self.gui = self.viz.viewer.gui
                # self.gui.setCameraTransform(0, params.CAMERA_TRANSFORM)
                self.gui.addFloor('world/floor')
                self.gui.setLightingMode('world/floor', 'OFF')
            elif viewer == pin.visualize.MeshcatVisualizer:
                self.viz = viewer(self.robot_display.model, self.robot_display.collision_model,
                                  self.robot_display.visual_model)
                self.viz.initViewer(loadModel=True)
                self.viz.display(q)

    def display(self, q):
        if hasattr(self, 'viz'):
            self.viz.display(q)

    def integrate_dv(self, q, v, dv, dt):
        v_mean = v + 0.5 * dt * dv
        v += dt * dv
        q = pin.integrate(self.model, q, dt * v_mean)
        return q, v

    def get_placement_LF(self):
        return self.robot.framePosition(self.formulation.data(), self.LF)

    def get_placement_RF(self):
        return self.robot.framePosition(self.formulation.data(), self.RF)

    def set_com_ref(self, pos, vel, acc):
        self.sample_com.pos(pos)
        self.sample_com.vel(vel)
        self.sample_com.acc(acc)
        self.comTask.setReference(self.sample_com)

    def set_RF_3d_ref(self, pos, vel, acc):
        self.sample_RF_pos[:3] = pos
        self.sample_RF_vel[:3] = vel
        self.sample_RF_acc[:3] = acc
        self.sampleRF.pos(self.sample_RF_pos)
        self.sampleRF.vel(self.sample_RF_vel)
        self.sampleRF.acc(self.sample_RF_acc)
        self.rightFootTask.setReference(self.sampleRF)

    def set_LF_3d_ref(self, pos, vel, acc):
        self.sample_LF_pos[:3] = pos
        self.sample_LF_vel[:3] = vel
        self.sample_LF_acc[:3] = acc
        self.sampleLF.pos(self.sample_LF_pos)
        self.sampleLF.vel(self.sample_LF_vel)
        self.sampleLF.acc(self.sample_LF_acc)
        self.leftFootTask.setReference(self.sampleLF)

    def get_LF_3d_pos_vel_acc(self, dv):
        data = self.formulation.data()
        H = self.robot.framePosition(data, self.LF)
        v = self.robot.frameVelocity(data, self.LF)
        a = self.leftFootTask.getAcceleration(dv)
        return H.translation, v.linear, a[:3]

    def get_RF_3d_pos_vel_acc(self, dv):
        data = self.formulation.data()
        H = self.robot.framePosition(data, self.RF)
        v = self.robot.frameVelocity(data, self.RF)
        a = self.rightFootTask.getAcceleration(dv)
        return H.translation, v.linear, a[:3]

    def remove_contact_RF(self, transition_time=0.0):
        H_rf_ref = self.robot.framePosition(self.formulation.data(), self.RF)
        self.trajRF.setReference(H_rf_ref)
        self.rightFootTask.setReference(self.trajRF.computeNext())

        self.formulation.removeRigidContact(self.contactRF.name, transition_time)
        self.contact_RF_active = False

    def remove_contact_LF(self, transition_time=0.0):
        H_lf_ref = self.robot.framePosition(self.formulation.data(), self.LF)
        self.trajLF.setReference(H_lf_ref)
        self.leftFootTask.setReference(self.trajLF.computeNext())

        self.formulation.removeRigidContact(self.contactLF.name, transition_time)
        self.contact_LF_active = False

    def add_contact_RF(self, transition_time=0.0):
        H_rf_ref = self.robot.framePosition(self.formulation.data(), self.RF)
        self.contactRF.setReference(H_rf_ref)
        if self.params.w_contact >= 0.0:
            self.formulation.addRigidContact(self.contactRF, self.params.w_forceRef, self.params.w_contact, 1)
        else:
            self.formulation.addRigidContact(self.contactRF, self.params.w_forceRef)

        self.contact_RF_active = True

    def add_contact_LF(self, transition_time=0.0):
        H_lf_ref = self.robot.framePosition(self.formulation.data(), self.LF)
        self.contactLF.setReference(H_lf_ref)
        if self.params.w_contact >= 0.0:
            self.formulation.addRigidContact(self.contactLF, self.params.w_forceRef, self.params.w_contact, 1)
        else:
            self.formulation.addRigidContact(self.contactLF, self.params.w_forceRef)

        self.contact_LF_active = True

    def start(self, com_pos_ref, com_vel_ref, com_acc_ref):
        """ Start the TSID controller """
        if self.prev_phase == 'right':
            self.remove_contact_LF()
        else:
            self.remove_contact_RF()

        # TODO first solve of the MPC problem

        self.storage.save_com_references(com_pos_ref, com_vel_ref, com_acc_ref)

        # self.storage.save_feet_references()

    def compute(self, device, current_phase, tau):
        """ Run one iteration of the control-loop """
        # Update the contact phase
        if current_phase != self.prev_phase:
            if current_phase == 'left':
                self.add_contact_LF()
                self.remove_contact_RF()
            else:
                self.add_contact_RF()
                self.remove_contact_LF()

        self.set_com_ref(self.com_pos_ref, self.com_vel_ref, self.com_acc_ref)
        self.set_LF_3d_ref(self.x_LF_ref, self.dx_LF_ref, self.ddx_LF_ref)
        self.set_RF_3d_ref(self.x_RF_ref, self.dx_RF_ref, self.ddx_RF_ref)

        HQPData = self.formulation.computeProblemData(self.count * self.dt, self.q, self.v)
        sol = self.solver.solve(HQPData)

        # Check the if the solution is feasible or not
        tau_motor = self.formulation.getActuatorForces(sol)
        self.dv = self.formulation.getAccelerations(sol)

        # Update the state
        self.update_state(device, tau)

        # Integrate
        self.q, self.v = self.integrate_dv(self.q, self.v, self.dv, self.dt)
        self.count += 1

        # TODO compute the CoP

        if self.count % self.params.DISPLAY_N:
            self.display(self.q)

    def update_state(self, device, tau):
        if self.simu:
            self.storage.com_pos = self.robot.com(self.formulation.data())
            self.storage.com_vel = self.robot.com_vel(self.formulation.data())
            self.storage.com_acc = self.comTask.getAcceleration(self.dv)
            self.storage.x_LF, self.storage.dx_LF, self.storage.ddx_LF = self.get_LF_3d_pos_vel_acc(self.dv)
            if not self.contact_LF_active:
                self.storage.ddx_LF_des = self.leftFootTask.getDesiredAcceleration[:3]
            self.storage.x_RF, self.storage.dx_RF, self.storage.ddx_RF = self.get_RF_3d_pos_vel_acc(self.dv)
            if not self.contact_RF_active:
                self.storage.ddx_RF_des = self.rightFootTask.getDesiredAcceleration[:3]
        else:
            self.storage.com_pos = device.com_pos
            self.storage.com_vel = device.com_vel
            self.storage.com_acc = device.com_acc

        nb_left = rolloutLipmDynamics(self.storage.com_pos[:2], tau, self.dt, self.params)
        self.t_left = self.dt * nb_left

import numpy as np
import scipy.io

class BipedParams:

    def __init__(self, mat_path, filename = None):
        self.h_com = 0.66  #0.58
        self.g = 9.81
        self.omega = np.sqrt(self.g / self.h_com)

        self.nb_steps = 5
        self.T = 1.2
        self.dt = 1e-3
        self.dt_mpc = 3 * 1e-2
        self.ratio = int(self.dt_mpc / self.dt)

        self.r_bar = 0.15
        self.v_bar = -self.omega * self.r_bar * (1 + np.exp(self.omega * self.T)) / (1 - np.exp(self.omega * self.T))

        foot_scaling = 1.
        self.lxp = foot_scaling * 0.075  # foot length in positive x direction
        self.lxn = foot_scaling * 0.075  # foot length in negative x direction
        self.lyp = foot_scaling * 0.05  # foot length in positive y direction
        self.lyn = foot_scaling * 0.05  # foot length in negative y direction
        self.foot_length = self.lxp + self.lxn
        self.foot_width = self.lyp + self.lyn
        self.x_sat = self.foot_length / 2
        self.y_sat = self.foot_width / 2

        data = scipy.io.loadmat(mat_path + '/data.mat')
        self.K = data['K']

        # ICs
        self.foot_step_0 = np.array([0., -0.08])
        self.tau_hat_0 = data['x0'][-1].item()
        self.x_hat_0 = data['x0'][:-1].T[0]
        self.y_hat_0 = np.array([-0.04, 0.0])

        self.step_length = self.r_bar * 2
        self.step_width = 2 * np.absolute(self.y_hat_0[0])
        self.step_height = 0.05

        # Weights
        self.alpha = 1e2
        self.beta = 0
        self.gamma = 1e-1

        # Configuration for TSID
        self.path = '/opt/openrobots/share/example-robot-data/robots/romeo_description'
        self.urdf = self.path + '/urdf/romeo.urdf'
        self.srdf = self.path + '/srdf/romeo.srdf'
        self.nv = 37
        self.lz = 0.07  # foot sole height with respect to ankle joint
        self.mu = 0.3  # friction coefficient
        self.fMin = 0.0  # minimum normal force
        self.fMax = 1e6  # maximum normal force
        self.rf_frame_name = "RAnkleRoll"  # right foot frame name
        self.lf_frame_name = "LAnkleRoll"  # left foot frame name
        self.contactNormal = np.matrix([0., 0., 1.]).T  # direction of the normal to the contact surface
        self.contact_phase_0 = 'right'

        self.T_pre = 1.5  # simulation time before starting to walk
        self.T_post = 1.5  # simulation time after walking

        self.w_com = 1.0  # weight of center of mass task
        self.w_cop = 0.0  # weight of center of pressure task
        self.w_am = 1e-4  # weight of angular momentum task
        self.w_foot = 1e0  # weight of the foot motion task
        self.w_contact = 1e2  # weight of the foot in contact
        self.w_posture = 1e-4  # weight of joint posture task
        self.w_forceRef = 1e-5  # weight of force regularization task
        self.w_torque_bounds = 0.0  # weight of the torque bounds
        self.w_joint_bounds = 0.0

        self.tau_max_scaling = 1.45  # scaling factor of torque bounds
        self.v_max_scaling = 0.8

        self.kp_contact = 10.0  # proportional gain of contact constraint
        self.kp_foot = 10.0  # proportional gain of contact constraint
        self.kp_com = 3.0  # proportional gain of center of mass task
        self.kp_am = 10.0  # proportional gain of angular momentum task
        self.kp_posture = 1.0  # proportional gain of joint posture task
        self.gain_vector = self.kp_posture * np.ones(self.nv - 6)
        self.masks_posture = np.ones(self.nv - 6)

        # Configuration for viewer
        self.PRINT_N = 500  # print every PRINT_N time steps
        self.DISPLAY_N = 20  # update robot configuration in viwewer every DISPLAY_N time steps
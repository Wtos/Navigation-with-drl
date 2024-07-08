from ros_wrapper_pkg.ros_msg.imu import ImuData
from ros_wrapper_pkg.ros_msg.ros_dtype import ROSDtype
import gin
import os
from utilis.utilis import get_joint_id
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

ROS_IMU_TOPIC="imu"

class IMUSensor:
    def __init__(self, ros_wrapper, robot_id, joint_id) :
        self.ros_wrapper = ros_wrapper 
        self.robot_id = robot_id
        self.joint_id = joint_id
        self.frame = p.getJointInfo(robot_id, self.joint_id)[12].decode("utf-8") 
        self.ros_wrapper.add_publisher(ROS_IMU_TOPIC, ROSDtype.IMU, use_namespace=False)
        self.gravity = np.array([0.0, 0.0, 9.8])
        self.lin_vel = np.array([0.0, 0.0, 0.0])
        self.quat  = np.array([0.0, 0.0, 0.0, 1.0])
        self.ang_vel_r =  np.array([0.0, 0.0, 0.0])
        self.lin_vel_last = np.array([0.0, 0.0, 0.0])
        self.lin_acc_nog = np.array([0.0, 0.0, 0.0])
        self.acc_filter = 0.9
        self.env_dt = p.getPhysicsEngineParameters()['fixedTimeStep']
    
    def update_imu(self):
        link_state = p.getLinkState(self.robot_id, self.joint_id, computeLinkVelocity=True, computeForwardKinematics=False)
        self.quat = np.array(link_state[1])
        rot_mat = Rotation.from_quat(self.quat).as_matrix()
        self.lin_vel = np.array(link_state[6])
        self.ang_vel_r =  np.dot(rot_mat.transpose(), np.array(link_state[7]))
        self.lin_acc_nog = self.acc_filter * self.lin_acc_nog + (1.0 - self.acc_filter) * ((self.lin_vel - self.lin_vel_last) / self.env_dt)
        lin_acc_g = self.lin_acc_nog + self.gravity
        lin_acc_g_r = np.dot(rot_mat.transpose(), lin_acc_g)
        imu = ImuData(self.quat, self.ang_vel_r, lin_acc_g_r, self.frame)
        self.lin_vel_last = self.lin_vel
        self.ros_wrapper.publish_msg(ROS_IMU_TOPIC, imu)
        
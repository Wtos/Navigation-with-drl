import pybullet as p
import numpy as np
from lidar.lidar import Lidar
from imu.imu import IMUSensor
from scipy.spatial.transform import Rotation
from ros_wrapper_pkg.ros_msg.ros_dtype import ROSDtype
from ros_wrapper_pkg.ros_msg.odom import Odom
import gin
import os
from utilis.utilis import get_joint_id
import rospy

#发送nvigation模块输出的速度控制指令
ROS_CMD_VEL_TOPIC = "cmd_vel"
ROS_CMD_VEL_LEARNING_TOPIC = "cmd_vel_learning"
ROS_ODOM_TOPIC = "odom"

#odom 与 imu只需要发布


@gin.configurable
class Chassis:
    def __init__(self, ros_wrapper, robot_id, joint_id, lidar_joint, imu_joint):
        self.ros_wrapper = ros_wrapper
        self.robot_id = robot_id
        #print("root id:",robot_id)
        self.joint_id = joint_id # x y theta chassis
        self.lidar_joint = lidar_joint
        self.cmd_ratio = rospy.get_param("/cmd_ratio", 1.0)
        
        self.lidar_joint_id = get_joint_id(self.robot_id, self.lidar_joint)
        self.imu_joint = imu_joint
        self.imu_joint_id = get_joint_id(self.robot_id, self.imu_joint)
        self.frame = p.getJointInfo(robot_id, self.joint_id[3])[12].decode("utf-8") 
        self.twist = np.array([0.0, 0.0, 0.0]) # x y theta
        self.twist_learning = np.array([0.0, 0.0, 0.0]) # x y theta
        self.pos = np.array([0.0, 0.0, 0.0])
        self.chassis_state = p.getLinkState(self.robot_id, self.joint_id[3], computeLinkVelocity=True, computeForwardKinematics=False) # joint_id[3] means the last joint for chassis, the base
        self.pos_offset = np.array(self.chassis_state[0])
        self.lidar = Lidar(self.ros_wrapper, self.robot_id, self.lidar_joint_id)
        self.imu = IMUSensor(self.ros_wrapper, self.robot_id, self.imu_joint_id)
        self.init_ros_wrapper()
        self.update_twist()
    
    def init_ros_wrapper(self):
        self.ros_wrapper.add_subscriber(ROS_CMD_VEL_TOPIC, ROSDtype.TWIST, use_namespace=False, callback=self.cmdvel_callback)
        self.ros_wrapper.add_subscriber(ROS_CMD_VEL_LEARNING_TOPIC, ROSDtype.TWIST, use_namespace=False, callback=self.cmdvel_learning_callback)
        self.ros_wrapper.add_publisher(ROS_ODOM_TOPIC, ROSDtype.ODOM, use_namespace=False)
        
    
    def cmdvel_callback(self, msg):
        self.twist[0] = msg.linear.x
        self.twist[1] = msg.linear.y
        self.twist[2] = msg.angular.z
        self.update_twist()
    
    def cmdvel_learning_callback(self, msg):
        self.twist_learning[0] = msg.linear.x
        self.twist_learning[1] = msg.linear.y
        self.twist_learning[2] = msg.angular.z
        self.update_twist()
        
    #控制底盘运动   
    def update_twist(self):
        self.update_pos()
        twist = self.cmd_ratio * self.twist + (1 - self.cmd_ratio) * self.twist_learning
        theta = self.pos[2]
        twist_w = np.array([0.0, 0.0, 0.0])
        twist_w[0] = twist[0] * np.cos(theta) - twist[1] * np.sin(theta)
        twist_w[1] = twist[0] * np.sin(theta) + twist[1] * np.cos(theta)
        twist_w[2] = twist[2]
        for i in range(3):
            p.setJointMotorControl2(self.robot_id, self.joint_id[i] , p.VELOCITY_CONTROL, targetVelocity=twist_w[i],force=1000)
    
    def update_pos(self):
        self.chassis_state = p.getLinkState(self.robot_id, self.joint_id[3], computeLinkVelocity=True, computeForwardKinematics=False)
        self.pos[0] = self.chassis_state[0][0] -self.pos_offset[0]
        self.pos[1] = self.chassis_state[0][1] - self.pos_offset[1]
        self.pos[2] = Rotation.from_quat(self.chassis_state[1]).as_euler('xyz')[2]
    
    def publish_odom(self):
        self.update_pos()
        pos = np.array(self.chassis_state[0]) - self.pos_offset
        odom = Odom(pos, self.chassis_state[1], self.chassis_state[6], self.chassis_state[7],"odom")
        self.ros_wrapper.publish_msg(ROS_ODOM_TOPIC, odom)
        self.ros_wrapper.publish_tf("odom", self.frame, pos, self.chassis_state[1] )
    
    def publish_imu(self):
        self.imu.update_imu()
        
    
CONFIG_FILE = os.path.dirname(os.path.abspath(__file__)) + "/../config/chassis.gin"
gin.parse_config_file(CONFIG_FILE)
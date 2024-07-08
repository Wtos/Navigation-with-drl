from enum import Enum
from ros_wrapper_pkg.ros_msg.float_array import FloatArray
from ros_wrapper_pkg.ros_msg.imu import ImuData
from ros_wrapper_pkg.ros_msg.point_cloud import PointCloud
from ros_wrapper_pkg.ros_msg.robot_joint_state import RobotJointState
from ros_wrapper_pkg.ros_msg.ros_clock import ROSClock
from ros_wrapper_pkg.ros_msg.float import Float
from ros_wrapper_pkg.ros_msg.force import Force
from ros_wrapper_pkg.ros_msg.wrench import Wrench
from ros_wrapper_pkg.ros_msg.image import Image
from ros_wrapper_pkg.ros_msg.laser_scan import LaserScan
from ros_wrapper_pkg.ros_msg.twist import Twist
from ros_wrapper_pkg.ros_msg.odom import Odom

class ROSDtype(Enum):
    FLOAT = Float
    FLOAT_ARRAY = FloatArray
    JOINT_STATE = RobotJointState
    WRENCH = Wrench
    FORCE = Force
    IMU = ImuData
    CLOCK  = ROSClock
    IMAGE = Image
    POINT_CLOUD = PointCloud
    LASER_SCAN = LaserScan
    TWIST = Twist
    ODOM = Odom
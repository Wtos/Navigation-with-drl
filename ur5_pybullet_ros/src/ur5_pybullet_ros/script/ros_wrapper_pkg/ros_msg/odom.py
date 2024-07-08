from ros_wrapper_pkg.ros_msg.ros_msg_base import ROSMsgBase
from nav_msgs.msg import Odometry
import rospy

class Odom(ROSMsgBase):
    rosdtype = Odometry
    def __init__(self, pos, ori, lin_vel, ang_vel, frame_id):
        super().__init__()
        self.odom = Odometry()
        self.odom.header.frame_id = frame_id
        self.odom.pose.pose.position.x = pos[0]
        self.odom.pose.pose.position.y = pos[1]
        self.odom.pose.pose.position.z = pos[2]
        self.odom.pose.pose.orientation.x = ori[0]
        self.odom.pose.pose.orientation.y = ori[1]
        self.odom.pose.pose.orientation.z = ori[2]
        self.odom.pose.pose.orientation.w = ori[3]
        self.odom.twist.twist.linear.x = lin_vel[0]
        self.odom.twist.twist.linear.y = lin_vel[1]
        self.odom.twist.twist.linear.z = lin_vel[2]
        self.odom.twist.twist.angular.x = ang_vel[0]
        self.odom.twist.twist.angular.y = ang_vel[1]
        self.odom.twist.twist.angular.z = ang_vel[2]
        
    @classmethod
    def transform_rosmsg(cls, data, ros_time, frame_id=""):
        ros_msg = data.odom
        ros_msg.header.stamp = rospy.Time.from_sec(ros_time)
        return ros_msg
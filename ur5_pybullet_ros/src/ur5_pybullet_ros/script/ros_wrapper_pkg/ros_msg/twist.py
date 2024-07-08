from geometry_msgs.msg import Twist as TwistROS
from ros_wrapper_pkg.ros_msg.ros_msg_base import ROSMsgBase
import rospy


class Twist(ROSMsgBase):
    rosdtype = TwistROS
    def __init__(self, lin_vel, ang_vel):
        self.twist = TwistROS()
        self.twist.linear.x = lin_vel[0]
        self.twist.linear.y = lin_vel[1]
        self.twist.linear.z = lin_vel[2]
        self.twist.angular.x = ang_vel[0]
        self.twist.angular.y = ang_vel[1]
        self.twist.angular.z = ang_vel[2]

    @classmethod
    def transform_rosmsg(cls, data, ros_time, frame_id=""):
        ros_msg = data.twist
        return ros_msg
import rospy
from sensor_msgs.msg import JointState
from ros_wrapper_pkg.ros_msg.ros_msg_base import ROSMsgBase

class RobotJointState(ROSMsgBase):
    rosdtype = JointState
    def __init__(self, name, pos, vel, tor):
        super().__init__()
        self.joint_state = JointState()
        self.joint_state.name = name
        self.joint_state.position = pos
        self.joint_state.velocity = vel
        self.joint_state.effort = tor
    
    @classmethod
    def transform_rosmsg(cls, data, ros_time, frame_id=""):
        ros_msg = data.joint_state
        ros_msg.header.stamp = rospy.Time.from_sec(ros_time)
        return ros_msg
    
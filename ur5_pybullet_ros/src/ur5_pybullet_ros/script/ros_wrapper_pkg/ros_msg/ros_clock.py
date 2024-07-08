import rospy
import numpy as np
from rosgraph_msgs.msg import Clock
from ros_wrapper_pkg.ros_msg.ros_msg_base import ROSMsgBase


class ROSClock(ROSMsgBase):
    rosdtype = Clock
    def __init__(self, time):
        super().__init__()
        self.ros_clock = Clock()
        self.ros_clock.clock = rospy.Time.from_sec(time)
        # header=Header(stamp=rospy.Time.from_sec(time))
        # self = Clock()

    @classmethod
    def transform_rosmsg(cls, data, ros_time, frame_id=""):
        ros_msg = data.ros_clock
        return ros_msg
    
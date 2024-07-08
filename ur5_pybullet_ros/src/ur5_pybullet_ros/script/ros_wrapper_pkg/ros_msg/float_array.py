from std_msgs.msg import Float64MultiArray
from ros_wrapper_pkg.ros_msg.ros_msg_base import ROSMsgBase
import numpy as np

class FloatArray(ROSMsgBase):
    rosdtype = Float64MultiArray
    def __init__(self): # just us np.array or list
        super().__init__()
    
    @classmethod
    def transform_rosmsg(cls, data, ros_time, frame_id=""):
        data = np.array(data)
        ros_msg = Float64MultiArray()
        ros_msg.data = data
        return ros_msg
    
    
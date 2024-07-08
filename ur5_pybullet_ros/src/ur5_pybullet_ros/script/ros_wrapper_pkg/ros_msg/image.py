from ros_wrapper_pkg.ros_msg.ros_msg_base import ROSMsgBase
import numpy as np
from sensor_msgs.msg import Image as ImageROS
from cv_bridge import CvBridge

class Image(ROSMsgBase):
    rosdtype = ImageROS
    def __init__(self): # just us np.array or list
        super().__init__()
    
    @classmethod
    def transform_rosmsg(cls, data, ros_time, frame_id=""):
        bridge = CvBridge()
        ros_msg = bridge.cv2_to_imgmsg(data,  encoding="passthrough")
        return ros_msg
    
    
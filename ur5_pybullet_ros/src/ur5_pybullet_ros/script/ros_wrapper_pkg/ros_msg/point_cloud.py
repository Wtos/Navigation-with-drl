from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointField, PointCloud2
from ros_wrapper_pkg.ros_msg.ros_msg_base import ROSMsgBase
from std_msgs.msg import Header
import rospy
import numpy as np

# The data structure of each point in ros PointCloud2: 16 bits = x + y + z + rgb
FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]
FIELDS_XYZRGB = [PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('rgb', 12, PointField.FLOAT32, 1)]

# Bit operations
BIT_MOVE_16 = 2**16
BIT_MOVE_8 = 2**8

class PointCloud(ROSMsgBase): #just use 
    rosdtype = PointCloud2
    def __init__(self, open3d_cloud, use_color=False):
        super().__init__()
        self.open3d_cloud = open3d_cloud
        self.use_color = use_color
    
    @classmethod
    def transform_rosmsg(cls, data, ros_time, frame_id=""):
        header = Header()
        header.stamp = rospy.Time.from_sec(ros_time)
        header.frame_id = frame_id
        open3d_cloud = data.open3d_cloud
        points=np.asarray(open3d_cloud.points)
        if not data.use_color: # not open3d_cloud.colors: # XYZ only
            fields=FIELDS_XYZ
            cloud_data=points
        else: # XYZ + RGB
            fields=FIELDS_XYZRGB
            # -- Change rgb color from "three float" to "one 24-byte int"
            # 0x00FFFFFF is white, 0x00000000 is black.
            colors = np.floor(np.asarray(open3d_cloud.colors)*255) # nx3 matrix
            colors = colors[:,0] * BIT_MOVE_16 +colors[:,1] * BIT_MOVE_8 + colors[:,2]  
            cloud_data=np.c_[points, colors]
        ros_msg = point_cloud2.create_cloud(header, fields, cloud_data)
        return ros_msg

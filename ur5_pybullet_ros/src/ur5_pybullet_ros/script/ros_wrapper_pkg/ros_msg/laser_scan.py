from ros_wrapper_pkg.ros_msg.ros_msg_base import ROSMsgBase
from sensor_msgs.msg import LaserScan as ROSLaserScan
import rospy

class LaserScan(ROSMsgBase):
    rosdtype = ROSLaserScan
    def __init__(self, range_min, range_max, angle_min, angle_max, ray_data, frame_id):
        super().__init__()
        self.lisar_scan = ROSLaserScan()
        self.lisar_scan.range_min = range_min
        self.lisar_scan.range_max = range_max
        self.lisar_scan.angle_min = angle_min
        self.lisar_scan.angle_max = angle_max
        self.lisar_scan.angle_increment = (angle_max - angle_min) / len(ray_data)
        self.lisar_scan.header.frame_id = frame_id
        self.lisar_scan.time_increment = 0.0 # ?
        self.lisar_scan.scan_time = 0.0 # 10 hz
        self.rays_data = ray_data
    
    @classmethod
    def transform_rosmsg(cls, data, ros_time, frame_id=""):
        ros_msg = data.lisar_scan
        ros_msg.ranges = data.rays_data
        ros_msg.header.stamp = rospy.Time.from_sec(ros_time)
        return ros_msg
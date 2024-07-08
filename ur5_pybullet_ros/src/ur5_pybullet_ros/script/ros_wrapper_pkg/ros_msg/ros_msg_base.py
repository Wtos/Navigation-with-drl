
class ROSMsgBase:
    rosdtype = None
    def __init__(self):
        pass
    
    @classmethod
    def transform_rosmsg(cls, data, ros_time, frame_id=""):
        raise NotImplementedError("transform_rosmsg has not been implemented")
    
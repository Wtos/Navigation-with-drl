import cv2
import json
import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation
from robots.ur5 import UR5
from camera import Camera, CameraIntrinsic, Frame


class DebugAxes(object):
    """
    可视化某个局部坐标系, 红色x轴, 绿色y轴, 蓝色z轴
    """
    def __init__(self):
        self.uids = [-1, -1, -1]

    def update(self, pos, orn):
        """
        Arguments:
        - pos: len=3, position in world frame
        - orn: len=4, quaternion (x, y, z, w), world frame
        """
        pos = np.asarray(pos)
        rot3x3 = Rotation.from_quat(orn).as_matrix()
        axis_x, axis_y, axis_z = rot3x3.T
        self.uids[0] = p.addUserDebugLine(pos, pos + axis_x * 0.05, [1, 0, 0], replaceItemUniqueId=self.uids[0])
        self.uids[1] = p.addUserDebugLine(pos, pos + axis_y * 0.05, [0, 1, 0], replaceItemUniqueId=self.uids[1])
        self.uids[2] = p.addUserDebugLine(pos, pos + axis_z * 0.05, [0, 0, 1], replaceItemUniqueId=self.uids[2])


class BindCamera(Camera):
    def __init__(self, obj_id, intrinsic, near=0.01, far=4, rela_tform=None):
        """
        Arguments:
        - obj_id: int, object uid generate from p.loadURDF()
        - intrinsic: CameraIntrinsic object
        - rela_tform: 4x4 relative transform matrix to binding object
        """
        super(BindCamera, self).__init__(intrinsic, near, far)
        self.obj_id = obj_id
        self.rela_tform = rela_tform
    
    def object_pose(self):
        obj_pos, obj_orn = p.getBasePositionAndOrientation(self.obj_id)

        woT = np.eye(4)
        woT[:3, :3] = Rotation.from_quat(obj_orn).as_matrix()
        woT[:3, 3] = np.array(obj_pos)

        return woT
    
    def extrinsic(self):
        woT = self.object_pose()
        ocT = self.rela_tform if self.rela_tform is not None else np.eye(4)
        wcT = np.dot(woT, ocT)

        return wcT
    
    def render(self):
        wcT = self.extrinsic()
        cwT = np.linalg.inv(wcT)

        return super(BindCamera, self).render(cwT)


if __name__ == "__main__":
    p.connect(p.GUI)
    ur5_robot = UR5()
    camera_config = "./setup.json"
    with open(camera_config, "r") as j:
        config = json.load(j)
    camera_intrinsic = CameraIntrinsic.from_dict(config["intrinsic"])
    while True:
        pass
    

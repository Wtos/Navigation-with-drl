import numpy as np
import open3d as o3d
import pybullet as p
import json
import cv2
import time
import threading
from scipy.spatial.transform import Rotation as R
import gin
import os
from ros_wrapper_pkg.ros_msg.ros_dtype import ROSDtype
from ros_wrapper_pkg.ros_msg.point_cloud import PointCloud
import rospy
# some codes are copied from https://github.com/ethz-asl/vgn.git

ROS_IMAGE_TOPIC = "camera"
ROS_POINT_CLOUD_TOPIC = "point_cloud"

class CameraIntrinsic(object):
    """Intrinsic parameters of a pinhole camera model.

    Attributes:
        width (int): The width in pixels of the camera.
        height(int): The height in pixels of the camera.
        K: The intrinsic camera matrix.
    """

    def __init__(self, width, height, fx, fy, cx, cy):
        self.width = width
        self.height = height
        self.K = np.array(
            [[fx, 0.0, cx],
             [0.0, fy, cy],
             [0.0, 0.0, 1.0]]
        )

    @property
    def fx(self):
        return self.K[0, 0]

    @property
    def fy(self):
        return self.K[1, 1]

    @property
    def cx(self):
        return self.K[0, 2]

    @property
    def cy(self):
        return self.K[1, 2]

    def to_dict(self):
        """Serialize intrinsic parameters to a dict object."""
        data = {
            "width": self.width,
            "height": self.height,
            "K": self.K.flatten().tolist(),
        }
        return data

    @classmethod
    def from_dict(cls, data):
        """Deserialize intrinisic parameters from a dict object."""
        intrinsic = cls(
            width=data["width"],
            height=data["height"],
            fx=data["K"][0],
            fy=data["K"][4],
            cx=data["K"][2],
            cy=data["K"][5],
        )
        return intrinsic


@gin.configurable
class Camera(object):
    """Virtual RGB-D camera based on the PyBullet camera interface.

    Attributes:
        intrinsic: The camera intrinsic parameters.
    """

    def __init__(self, ros_wrapper, robot_id, ee_id, camera_config, near, far, relative_offset, downsample_resolution):
        camera_config = os.path.dirname(os.path.abspath(__file__)) + "/" + camera_config
        with open(camera_config, "r") as j:
            config = json.load(j)
        camera_intrinsic = CameraIntrinsic.from_dict(config["intrinsic"])
        self.ros_wrapper = ros_wrapper
        self.robot_id = robot_id
        self.ee_id = ee_id
        self.ee_frame = p.getJointInfo(robot_id,self.ee_id )[12].decode("utf-8") 
        self.camera_frame = "camera_link"
        self.intrinsic = camera_intrinsic
        self.near = near
        self.far = far
        self.proj_matrix = _build_projection_matrix(camera_intrinsic, near, far)
        self.gl_proj_matrix = self.proj_matrix.flatten(order="F")
        self.pose = [0, 0, 1]
        self.orien = [0, 0, 0, 1]
        self.rgb = None
        self.bgr = None
        self.point_cloud = None
        self.relative_offset = relative_offset # 相机原点相对于末端执行器局部坐标系的偏移量
        self.downsample_resolution = downsample_resolution
        self.init_ros_wrapper()
        # thread for updating camera image
        self.update_camera_image_thread = threading.Thread(
            target=self.update_camera_image)
        self.update_camera_image_thread.setDaemon(True)
        self.update_camera_image_thread.start()
        
        # self.update_camera_tf_thread = threading.Thread(
        #     target=self.publish_tf_thread)
        # self.update_camera_tf_thread.setDaemon(True)
        # self.update_camera_tf_thread.start()
        
        
    def init_ros_wrapper(self):
        self.ros_wrapper.add_publisher(ROS_IMAGE_TOPIC, ROSDtype.IMAGE)
        self.ros_wrapper.add_publisher(ROS_POINT_CLOUD_TOPIC, ROSDtype.POINT_CLOUD)

    def render(self, extrinsic):
        """Render synthetic RGB and depth images.

        Args:
            extrinsic: Extrinsic parameters, T_cam_ref.
        """
        # Construct OpenGL compatible view and projection matrices.
        gl_view_matrix = extrinsic.copy() if extrinsic is not None else np.eye(4)
        gl_view_matrix[2, :] *= -1  # flip the Z axis
        gl_view_matrix = gl_view_matrix.flatten(order="F")

        result = p.getCameraImage(
            width=self.intrinsic.width,
            height=self.intrinsic.height,
            viewMatrix=gl_view_matrix,
            projectionMatrix=self.gl_proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        rgb, z_buffer = np.ascontiguousarray(result[2][:, :, :3]), result[3]
        depth = (
            1.0 * self.far * self.near / (self.far - (self.far - self.near) * z_buffer)
        )
        return Frame(rgb, depth, self.intrinsic, extrinsic)
    
    def update_camera_image(self):
        # cv2.namedWindow("image")
        while True:
            self.update_camera_image_frame()
            # cv2.imshow("image", self.bgr)
            # key = cv2.waitKey(1)
            rospy.sleep(0.1)
    
    def update_camera_image_frame(self):
        self.update_pose()
        # self.publish_tf()
        wcT = self._bind_camera_to_end(self.pose, self.orien)
        cwT = np.linalg.inv(wcT)

        frame = self.render(cwT)
        assert isinstance(frame, Frame)

        self.rgb = frame.color_image()  # 这里以显示rgb图像为例, frame还包含了深度图, 也可以转化为点云
        self.bgr = np.ascontiguousarray(self.rgb[:, :, ::-1])  # flip the rgb channel
        self.point_cloud = PointCloud(frame.point_cloud(self.downsample_resolution), True)
        self.publish_data()

    def update_pose(self):
        end_state = p.getLinkState(self.robot_id, self.ee_id)
        self.pose = end_state[0]
        self.orien = end_state[1]
    
    def publish_tf(self):
        self.update_pose()
        translation, rotation = self.get_cam_offset()
        self.ros_wrapper.publish_tf(self.ee_frame, self.camera_frame, translation, rotation )
    
    def publish_tf_thread(self):
        last_time = 0
        while not rospy.is_shutdown():
            if last_time != self.ros_wrapper.ros_time:
                self.publish_tf()
            last_time = self.ros_wrapper.ros_time
            rospy.sleep(0.005)
            
    
    def publish_data(self):
        self.ros_wrapper.publish_msg(ROS_IMAGE_TOPIC, self.bgr)
        self.ros_wrapper.publish_msg(ROS_POINT_CLOUD_TOPIC, self.point_cloud, self.camera_frame)

    def _bind_camera_to_end(self, end_pos, end_orn):
        """设置相机坐标系与末端坐标系的相对位置
        
        Arguments:
        - end_pos: len=3, end effector position
        - end_orn: len=4, end effector orientation, quaternion (x, y, z, w)

        Returns:
        - wcT: shape=(4, 4), transform matrix, represents camera pose in world frame
        """
        
        end_orn = R.from_quat(end_orn).as_matrix()
        end_x_axis, end_y_axis, end_z_axis = end_orn.T
        wcT = np.eye(4)  # w: world, c: camera, ^w_c T
        wcT[:3, 0] = -end_y_axis  # camera x axis
        wcT[:3, 1] = -end_z_axis  # camera y axis
        wcT[:3, 2] = end_x_axis  # camera z axis
        wcT[:3, 3] = end_orn.dot(self.relative_offset) + end_pos  # eye position
        return wcT

    def get_cam_offset(self):
        translation = self.relative_offset
        rotation = [0.5,-0.5, 0.5, -0.5 ]
        return translation, rotation
        
            


class Frame(object):
    def __init__(self, rgb, depth, intrinsic, extrinsic=None):
        self.rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=o3d.geometry.Image(rgb),
            depth=o3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=2.0,
            convert_rgb_to_intensity=False
        )

        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=intrinsic.width,
            height=intrinsic.height,
            fx=intrinsic.fx,
            fy=intrinsic.fy,
            cx=intrinsic.cx,
            cy=intrinsic.cy,
        )

        self.extrinsic = extrinsic if extrinsic is not None \
            else np.eye(4)
    
    def color_image(self):
        return np.asarray(self.rgbd.color)
    
    def depth_image(self):
        return np.asarray(self.rgbd.depth)

    def point_cloud(self, downsample_voxel):
        pc = o3d.geometry.PointCloud.create_from_rgbd_image(
            image=self.rgbd,
            intrinsic=self.intrinsic,
            # extrinsic=self.extrinsic
        )
        if downsample_voxel > 0:
            voxel_size = downsample_voxel  # 体素大小，根据需要调整
            pc_downsampled = pc.voxel_down_sample(voxel_size=voxel_size)
            return pc_downsampled
        else:
            return pc

    
def _build_projection_matrix(intrinsic, near, far):
    perspective = np.array(
        [
            [intrinsic.fx, 0.0, -intrinsic.cx, 0.0],
            [0.0, intrinsic.fy, -intrinsic.cy, 0.0],
            [0.0, 0.0, near + far, near * far],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )
    ortho = _gl_ortho(0.0, intrinsic.width, intrinsic.height, 0.0, near, far)
    return np.matmul(ortho, perspective)


def _gl_ortho(left, right, bottom, top, near, far):
    ortho = np.diag(
        [2.0 / (right - left), 2.0 / (top - bottom), -2.0 / (far - near), 1.0]
    )
    ortho[0, 3] = -(right + left) / (right - left)
    ortho[1, 3] = -(top + bottom) / (top - bottom)
    ortho[2, 3] = -(far + near) / (far - near)
    return ortho

CONFIG_FILE = os.path.dirname(os.path.abspath(__file__)) + "/../config/camera.gin"
gin.parse_config_file(CONFIG_FILE)
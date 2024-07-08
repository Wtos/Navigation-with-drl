from controller.trajectory import Trajectory, get_interpolation_point
from enum import Enum, auto
import numpy as np


class FollowState(Enum):
    IDLE  = auto()
    RUNNNING  = auto()
    CANCELLED  = auto()
    SUCCESS  = auto()
    FAILED  = auto()
    
class TrajecyFollower:
    def __init__(self, joint_names):
        self.joint_names = joint_names
        self.target_pos = []
        self.trajectory = Trajectory()
        self.state = FollowState.IDLE
        self.current_velocity = []
        self.current_position = []
        self.goal_position = []
        self.goal_torlerance = 0.05

    def set_trajectory(self, trajectory:Trajectory):
        self.trajectory = trajectory
        self.goal_position = self.trajectory.points[-1].positions
        self.state = FollowState.RUNNNING
    
    def update_pos_vel(self, current_position, current_velocity):
        self.current_velocity = current_velocity
        self.current_position = current_position

    def get_control_point(self, time_now):
        return get_interpolation_point(self.trajectory, time_now)
    
    def loop(self, current_velocity, current_position):
        self.update_pos_vel(current_velocity, current_position)
        if self.state == FollowState.IDLE or self.state == FollowState.SUCCESS :
            return
        elif self.state == FollowState.RUNNNING:
            is_success = np.linalg.norm(np.array(self.current_position) - np.array(self.goal_position)) <  self.goal_torlerance
            if is_success:
               self.state = FollowState. SUCCESS
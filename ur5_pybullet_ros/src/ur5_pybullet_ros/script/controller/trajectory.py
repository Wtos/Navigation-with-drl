import numpy as np
from control_msgs.msg import FollowJointTrajectoryGoal

class Point:
    def __init__(self, positions=[], velocities=[], time_stamp = 0.0):
        self.positions = positions
        self.velocities = velocities
        self.time_stamp = time_stamp
        


class Trajectory:
    def __init__(self, joint_names=[], points=[]):
        self.joint_names = joint_names
        self.points = points
        self.joint_num = len(self.joint_names)
        self.positions = []
        self.velocities = []
        self.timesteps = []
        self.point_arrange()
        # print(self.positions)
        # print(self.velocities)
        # print(self.timesteps)
    
    def point_arrange(self):
        self.positions = [self.points[i].positions for i in range(len(self.points))]
        self.velocities = [self.points[i].velocities for i in range(len(self.points))]
        self.timesteps = [self.points[i].time_stamp for i in range(len(self.points))]
    
def get_interpolation_point(trajectory:Trajectory, query_time):
    interpolated_positions = np.zeros((trajectory.joint_num,))
    interpolated_velocities = np.zeros((trajectory.joint_num,))
    for i in range(trajectory.joint_num): 
        interpolated_positions[i] = np.interp(query_time, trajectory.timesteps, [trajectory.positions[k][i] for k in range(len(trajectory.positions))])
        interpolated_velocities[i] = np.interp(query_time, trajectory.timesteps, [trajectory.velocities[k][i] for k in range(len(trajectory.velocities))])
    return Point(interpolated_positions,  interpolated_velocities, query_time)

def get_trajectory_from_ros_msg(goal:FollowJointTrajectoryGoal, ros_time):
    joint_names = goal.trajectory.joint_names
    points = []
    for i in range(len(goal.trajectory.points)):
        point = Point(goal.trajectory.points[i].positions, goal.trajectory.points[i].velocities, goal.trajectory.points[i].time_from_start.to_sec()  +  ros_time)
        points.append(point)
    # positions = [goal.trajectory.points[i].positions for i in range(len(goal.trajectory.points))]
    # velocities = [goal.trajectory.points[i].velocities for i in range(len(goal.trajectory.points))]
    # timesteps = [goal.trajectory.points[i].time_from_start.to_sec() +  ros_time for i in range(len(goal.trajectory.points))]
    return Trajectory(joint_names, points)
        
if __name__ == "__main__":
    points = []
    for i in range(10):
        point = Point()
        point.positions = [i*0.1, i*0.2]
        point.velocities = np.array([0.1, 0.2]) / 0.05
        point.time_stamp = i * 0.05
        points.append(point)
    tra_follower = Trajectory(["joint1", "joint2"], points)
    result = get_interpolation_point(tra_follower, 0.13)
    print(result.positions)
    print(result.velocities)
#!/usr/bin/env python3

import sys
import rospy
import os
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from enum import Enum
from moveit_commander.conversions import pose_to_list
from scipy.spatial.transform import Rotation

try:
    from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt

class ActionResult(Enum):
    SUCCESS = 0
    PLAN_FAILED = 1
    EXECUTE_TIMEOUT = 2
    INACCURATE_EXECUTION = 3


#链接ur5-moveit 与 pybullet的
    
def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if the values in two lists are within a tolerance of each other.
    For Pose and PoseStamped inputs, the angle between the two quaternions is compared (the angle
    between the identical orientations q and -q is calculated correctly).
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        # Euclidean distance
        d = dist((x1, y1, z1), (x0, y0, z0))
        # phi = angle between orientations
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

    return True

class UR5MoveitInterface:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group_name = "arm"
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)
        self.display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )
        self.move_group.set_pose_reference_frame("base_link")
        self.planning_frame = self.move_group.get_planning_frame()
        
        print("============ Planning frame: %s" % self.planning_frame)
        self.eef_link = self.move_group.get_end_effector_link()
        print("============ End effector link: %s" % self.eef_link)
        
        self.group_names = self.robot.get_group_names()
        print("============ Available Planning Groups:", self.robot.get_group_names())
        self.move_group.set_max_velocity_scaling_factor(0.7)
        self.move_group.set_max_acceleration_scaling_factor(0.7)
        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        # print("============ Printing robot state")
        # print(self.robot.get_current_state())
        # print("")
        self.pose_goal = geometry_msgs.msg.PoseStamped()
        self.goal_pose_pub = rospy.Publisher('/goal_pose', geometry_msgs.msg.PoseStamped, queue_size=1)
    
    def go_to_joint_state(self, joint_state):
        move_group = self.move_group
        joint_goal = move_group.get_current_joint_values()
        assert len(joint_state) == len(joint_goal), "joint_state must have same dim with joint_goal!"
        joint_goal = joint_state
        print("set joint target ",  joint_goal)
        move_group.go(joint_goal, wait=True)
        move_group.stop()
        current_joints = move_group.get_current_joint_values()
        if all_close(joint_goal, current_joints, 0.05):
            print("joint state set success")
            return ActionResult.SUCCESS
        else:
            print("joint state set failed")

    def go_to_pose_goal(self, position, orientation, frame_id="base_link"): 
        """
        @param: position       position x y z 
        @param: orientation     x y z w
        @returns: bool
        """
        move_group = self.move_group
        self.pose_goal.header.frame_id = frame_id
        self.pose_goal.header.stamp = rospy.Time.now()
        self.pose_goal.pose.orientation.x = orientation[0]
        self.pose_goal.pose.orientation.y = orientation[1]
        self.pose_goal.pose.orientation.z = orientation[2]
        self.pose_goal.pose.orientation.w = orientation[3]
        
        self.pose_goal.pose.position.x = position[0]
        self.pose_goal.pose.position.y = position[1]
        self.pose_goal.pose.position.z = position[2]
        
        move_group.set_pose_target(self.pose_goal.pose)
        try:
            plan_success, plan_to_current, planning_time, error_code = move_group.plan()
            if not plan_success:
                rospy.logerr("Plan Error!")
                move_group.stop()
                move_group.clear_pose_targets()
                return ActionResult.PLAN_FAILED
        except rospy.exceptions.ROSInterruptException:
            pass
        # move_group.execute(plan_to_current, wait=True)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        current_pose = self.move_group.get_current_pose().pose

        output_file = "src/ur5_pybullet_ros/script/controller/success-pos2.txt"
        #if success:
        # 将成功的点写入文件
        #    with open(output_file, "a") as file:
        #        file.write(f"{position.tolist()},{orientation.tolist()}\n")
        return all_close(self.pose_goal.pose, current_pose, 0.01)
    
    def go_to_position_goal(self, position, frame_id="base_link"):  
        move_group = self.move_group
        self.pose_goal.header.frame_id = frame_id
        self.pose_goal.header.stamp = rospy.Time.now()
        self.pose_goal.pose.orientation.x = 0.0
        self.pose_goal.pose.orientation.y = 0.0
        self.pose_goal.pose.orientation.z = 0.0
        self.pose_goal.pose.orientation.w = 1.0
        
        self.pose_goal.pose.position.x = position[0]
        self.pose_goal.pose.position.y = position[1]
        self.pose_goal.pose.position.z = position[2]
        
        move_group.set_position_target(position)
        try:
            plan_success, plan_to_current, planning_time, error_code = move_group.plan()
            if not plan_success:
                rospy.logerr("Plan Error!")
                move_group.stop()
                move_group.clear_pose_targets()
                return ActionResult.PLAN_FAILED
        except rospy.exceptions.ROSInterruptException:
            pass
        move_group.execute(plan_to_current, wait=True)
        # success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        current_pose = self.move_group.get_current_pose().pose
        return all_close(self.pose_goal.pose, current_pose, 0.01)
    
    def publish_status(self):
        if self.pose_goal.header.frame_id:
            self.goal_pose_pub.publish(self.pose_goal)

if __name__ == "__main__":
    rospy.init_node("ur5_move_group_python_interface")
    ur5_moveit_interface = UR5MoveitInterface()
    
    # ur5_moveit_interface.go_to_joint_state([0,-0.5,0,0,0,0])
    ur5_moveit_interface.go_to_pose_goal([0.5,0.0,0.7], Rotation.from_euler('xyz', [0, 90, 0], degrees=True).as_quat())
    # ur5_moveit_interface.go_to_position_goal([0.5,0.0,0.6])
    while True:
        ur5_moveit_interface.publish_status()
        rospy.sleep(0.01)
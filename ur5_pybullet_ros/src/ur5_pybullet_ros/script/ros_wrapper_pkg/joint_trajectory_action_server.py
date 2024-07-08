import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, FollowJointTrajectoryActionFeedback, FollowJointTrajectoryResult,FollowJointTrajectoryFeedback
from trajectory_msgs.msg import JointTrajectoryPoint
from enum import Enum, auto
import numpy as np

#机械臂轨迹追踪：moveit，不需要
GOAL_TORLERANCE = 0.05

class ActionState(Enum):
    IDLE  = auto()
    RUNNNING  = auto()
    CANCELLED  = auto()
    SUCCESS  = auto()
    FAILED  = auto()

class JointTrajectoryActionServer:
    def __init__(self, joint_names, controller_name = "position_joint_trajectory_controller"):
        self.joint_names = joint_names
        self.action_state = ActionState.IDLE
        self.goal = FollowJointTrajectoryGoal()
        self.joint_pos = []
        self.joint_vel = []
        self.new_goal = False
        self.server = actionlib.SimpleActionServer("/%s/follow_joint_trajectory"%(controller_name),
            FollowJointTrajectoryAction,execute_cb=self.execute_trajectory,
            auto_start=False
        )
        rospy.loginfo("server init")
        self.server.start()
        
    
    def is_success(self):
        is_success = np.linalg.norm(np.array(self.joint_pos) - np.array(self.goal.trajectory.points[-1].positions)) < GOAL_TORLERANCE
        return is_success
            
    
    def execute_trajectory(self, goal):
        # goal = FollowJointTrajectoryGoal()
        rospy.loginfo("Received trajectory goal")
        # print(goal)
        self.goal = goal
        joint_names = self.goal.trajectory.joint_names
        index = [self.joint_names.index(joint_names[i]) for i in range(len(joint_names))] # the order in moveit control output is alphabet order, we have to correct it
        for i in range(len(self.goal.trajectory.points)):
            self.goal.trajectory.points[i].positions = [self.goal.trajectory.points[i].positions[k] for k in index]
        self.goal.trajectory.joint_names = joint_names
        self.action_state = ActionState.RUNNNING
        self.new_goal = True
        while self.action_state != ActionState.IDLE:
            if self.server.is_preempt_requested():
                rospy.loginfo("Trajectory execution preempted")
                self.action_state = ActionState.IDLE
                self.server.set_preempted()  
                return
            if not self.joint_pos:
                print("no pos received yet!")
                self.action_state = ActionState.FAILED
            if self.action_state == ActionState.RUNNNING:
                feedback = FollowJointTrajectoryFeedback()
                feedback.header.stamp = rospy.Time.now()
                feedback.joint_names = self.goal.trajectory.joint_names
                feedback.actual = JointTrajectoryPoint()
                feedback.actual.positions = self.joint_pos
                feedback.actual.velocities = self.joint_vel
                feedback.actual.time_from_start = rospy.Duration(0.5 )
                self.server.publish_feedback(feedback)
                if self.is_success():
                    self.action_state = ActionState.SUCCESS
            elif self.action_state == ActionState.SUCCESS:
                result = FollowJointTrajectoryResult()
                result.error_code = FollowJointTrajectoryResult.SUCCESSFUL
                # 发布结果
                self.server.set_succeeded(result)
                self.action_state == ActionState.IDLE
                return
            elif self.action_state == ActionState.FAILED:
                print("action failed!")
                result = FollowJointTrajectoryResult()
                result.error_code = FollowJointTrajectoryResult.INVALID_GOAL
                # 发布结果
                self.server.set_succeeded(result)
                self.action_state == ActionState.IDLE
                return
            rospy.sleep(0.02)
        
    def update_current_state(self, pos, vel):
        self.joint_pos = pos
        self.joint_vel = vel

if __name__ == '__main__':
    rospy.init_node('position_joint_trajectory_controller_server')
    server = JointTrajectoryActionServer([], "ur5_controller")
    rospy.spin()
    # while True:
    #     server.loop()
    #     time.sleep(0.01)

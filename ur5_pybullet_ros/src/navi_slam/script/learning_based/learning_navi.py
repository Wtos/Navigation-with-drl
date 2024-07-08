#!/usr/bin/env python3
import rospy
import os
import math
import fc
import torch
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from torchsummary import summary
from nav_msgs.msg import Path
from net import MLPPolicy, CNNPolicy
from ppo import generate_action_no_sampling
from collections import deque
import copy

SCAN_TOPIC = "/scan"
ODOM_TOPIC = "/odom"
PLAN_TOPIC = "/move_base/GlobalPlanner/plan"
NUM_INPUT = 362
NUM_ACTIONS = 5
SUB_GOAL_OFFSET = 30
XY_TOLERANCE = 0.2

def get_xy_pose(pose:PoseStamped):
    return np.array([pose.pose.position.x, pose.pose.position.y])

def get_zero_twist():
    twist = Twist()
    twist.linear.x = 0.0 
    twist.linear.y = 0.0 
    twist.angular.z = 0.0 
    return twist
        
class LearningNavigation:
    def __init__(self):
        self.scan = LaserScan()
        self.odom = Odometry()
        self.path = Path()
        self.sub_goal = PoseStamped()
        self.ready_flag = np.array([0, 0, 0])
        rospy.Subscriber(SCAN_TOPIC, LaserScan, self.scan_callback)
        rospy.Subscriber(ODOM_TOPIC, Odometry, self.odom_callback)
        rospy.Subscriber(PLAN_TOPIC, Path, self.path_callback)
        self.pub_subgoal = rospy.Publisher('/subgoal', PoseStamped, queue_size=1)
        self.pub_twist = rospy.Publisher('/cmd_vel_learning', Twist, queue_size=1)
        self.model = rospy.get_param("/RL_model", 0)
        
        if self.model == 0:
            self.net = fc.FC_DQN(NUM_INPUT, NUM_ACTIONS)
            self.net.train(False)
            model_path = os.path.dirname(os.path.abspath(__file__)) + "/dqn_agent_best_fc_l2.dat"
            self.net.load_state_dict(torch.load( model_path, map_location=torch.device('cpu')))
            summary(self.net, (1, NUM_INPUT))
        elif self.model == 1:
            self.action_bound = [[-1, -1], [1, 1]]  # the limitation of velocity
            LASER_HIST = 3
            trained_model_file = os.path.dirname(os.path.abspath(__file__)) + "/stage2.pth"
            self.policy = CNNPolicy(frames=LASER_HIST, action_space=2)
            self.policy.cpu()  # policy.cuda() for gpu
            state_dict = torch.load(
                trained_model_file, map_location=torch.device("cpu")
            )  # torch.load(trained_model_file) for gpu
            self.policy.load_state_dict(state_dict)
                    
        
        
        
    
    def scan_callback(self, msg:LaserScan):
        self.scan = msg
        self.ready_flag[0] = 1
    
    def odom_callback(self, msg:Odometry):
        self.odom = msg
        self.ready_flag[1] = 1
    
    def path_callback(self, msg:Path):
        self.path = msg
        self.ready_flag[2] = 1
    
    def get_subgoal(self):
        pos_now = get_xy_pose(self.odom.pose)
        dist_array = np.zeros(len(self.path.poses))
        for i, pose in enumerate(self.path.poses):
            pos = np.array([pose.pose.position.x, pose.pose.position.y])
            dist_array[i] = np.linalg.norm(pos_now - pos)
        min_index = np.argmin(dist_array)
        self.sub_goal = self.path.poses[min(min_index + SUB_GOAL_OFFSET, len(self.path.poses) - 1)]
        self.pub_subgoal.publish(self.sub_goal)
    
    def has_arrived(self):
        dist = np.linalg.norm(get_xy_pose(self.odom.pose) - get_xy_pose(self.sub_goal))
        return  dist < XY_TOLERANCE
    
    def get_laser_observation(self):
        scan = copy.deepcopy(np.array(self.scan.ranges))
        sub_array = np.hsplit(
            scan, 4
        )  # adapt scan info when min and max angel equal to [-1.57,4.69] (rlca is [-3.14,3.14])
        scan = np.concatenate(
            (sub_array[3], sub_array[0], sub_array[1], sub_array[2])
        )  # adapt scan info when min and max angel equal to [-1.57,4.69] (rlca is [-3.14,3.14])
        # max_range = rospy.get_param('laser_range')
        raw_beam_num = len(scan)
        sparse_beam_num = 512
        step = float(raw_beam_num) / sparse_beam_num
        sparse_scan_left = []
        index = 0.0
        for x in range(int(sparse_beam_num / 2)):
            sparse_scan_left.append(scan[int(index)])
            index += step
        sparse_scan_right = []
        index = raw_beam_num - 1.0
        for x in range(int(sparse_beam_num / 2)):
            sparse_scan_right.append(scan[int(index)])
            index -= step
        scan_sparse = np.concatenate(
            (sparse_scan_left, sparse_scan_right[::-1]), axis=0
        )
        return scan_sparse / 6.0 - 0.5
    
    def get_local_goal(self):
        q = self.odom.pose.pose.orientation
        psi = np.arctan2(2.0*(q.w*q.z + q.x*q.y),
                                1-2*(q.y*q.y+q.z*q.z))  # bounded by [-pi, pi]
        x = self.odom.pose.pose.position.x
        y = self.odom.pose.pose.position.y
        theta = psi
        [goal_x, goal_y] = [self.sub_goal.pose.position.x, self.sub_goal.pose.position.y]  # sub goal based on map
        local_x = (goal_x - x) * np.cos(theta) + (goal_y - y) * np.sin(theta)
        local_y = -(goal_x - x) * np.sin(theta) + (goal_y - y) * np.cos(theta)
        return [local_x, local_y]  # return subgoal position based on robot
    
    def get_twist(self):
        if self.model == 0:
            q = self.odom.pose.pose.orientation
            psi = np.arctan2(2.0*(q.w*q.z + q.x*q.y),
                                1-2*(q.y*q.y+q.z*q.z))  # bounded by [-pi, pi]
            v_p = self.odom.pose.pose.position
            v_g = self.sub_goal.pose.position
            v_pg = np.array([v_g.x-v_p.x, v_g.y-v_p.y])
            v_pose = np.array([math.cos(psi), math.sin(psi)])
            angle = np.math.atan2(np.linalg.det(
                [v_pose, v_pg]), np.dot(v_pose, v_pg))
            distance = np.linalg.norm(v_pg)
            angle = math.degrees(angle)
            sample = np.asanyarray(self.scan.ranges).tolist()
            
            observation = [distance]+[angle]+sample
            state_v = torch.FloatTensor([observation]).to('cpu')
            q_vals_v = self.net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())
            # action_space = np.array([[0.2, 0], [0.15, 0.75], [0.15, -0.75], [0.0, 1.5], [0.0, -1.5]]) * 1.0
            action_space = np.array([[0.2, 0.0, 0], [0.15, 0.05, 0.75], [0.15, -0.05, -0.75], [0.05, 0.1,  1.5], [0.05, -0.1, -1.5]]) * 1.0 # x y omega
            # action_space = {0: [0.2,0], 1: [0.15,0.35], 2: [0.15,-0.35], 3: [0.0,0.75], 4: [0.0,-0.75]}
            twist = Twist()
            twist.linear.x = action_space[action][0]
            twist.linear.y = action_space[action][1]
            twist.angular.z = action_space[action][2]
            return twist
        elif self.model == 1:
            obs = self.get_laser_observation()
            obs_stack = deque([obs, obs, obs])
            q = self.odom.pose.pose.orientation
            psi = np.arctan2(2.0*(q.w*q.z + q.x*q.y),
                                1-2*(q.y*q.y+q.z*q.z))  # bounded by [-pi, pi]
            goal = np.asarray(self.get_local_goal())
            speed = np.asarray([self.odom.twist.twist.linear.x, self.odom.twist.twist.angular.z], dtype="float64")
            obs_state_list = [[obs_stack, goal, speed]]
            _, scaled_action = generate_action_no_sampling(
            obs_state_list, self.policy, self.action_bound
            )
            action = scaled_action[0]
            action[0] = 0.3 * action[0]  # the maximum speed of cmd_vel 0.3
            action[1] = 0.5 * action[1]
            twist = Twist()
            twist.linear.x = action[0]
            twist.linear.y = 0.0
            twist.angular.z = action[1]
            return twist
    
    def step(self):
        if np.min(self.ready_flag) > 0 and len(self.path.poses) > 0:
            self.get_subgoal()
            if not self.has_arrived():
                twist = self.get_twist()
            else:
                twist = get_zero_twist()
            self.pub_twist.publish(twist)
    
    

if __name__ == '__main__':
    rospy.init_node('learning_navigation')
    learning_navigation = LearningNavigation()
    while(not rospy.is_shutdown()):
        learning_navigation.step()
        rospy.Rate(20).sleep()
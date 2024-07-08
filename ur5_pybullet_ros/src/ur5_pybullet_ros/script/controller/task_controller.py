#!/usr/bin/env python3
from controller.ur5_moveit_interface import UR5MoveitInterface
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation
import time
from enum import Enum
from std_msgs.msg import Float64, Float64MultiArray
import threading
import pybullet as p
import controller.config_li as config_li
import os

#xyz
WORK1_POS = [np.array([0.65, -0.82, 0.0]), -0.5 * np.pi]
PRE_WORK1_POS = [np.array([0.65, -0.0, 0.0]), -0.5 * np.pi]
WORK2_POS = [np.array([-6.6, 0.82, 0.0]), 0.5 * np.pi]
PRE_WORK2_POS = [np.array([-6.6, 0.0, 0.0]), 0.5 * np.pi]
WORK3_POS = [np.array([-6.35, 0.82, 0.0]), 0.5 * np.pi]
WORK4_POS = [np.array([1.0, -0.82, 0.0]), -0.5 * np.pi]
PRE_WORK4_POS = [np.array([1.0, -0.0, 0.0]), -0.5 * np.pi]

class JointConfiguration(Enum):
    UPRIGHT = [0.0,-0.5 * np.pi,0,0,0,0]
    #改了
    DOWN_VIEW = [0.0, -0.5 *np.pi, 0.2 * np.pi, -0.5 * np.pi, -0.5 * np.pi, 0]

class TaskController:
    def __init__(self):
        #move base是navigation
        self.moveit_interface = UR5MoveitInterface()
        self.movebase_goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
        self.grip_ratio_pub = rospy.Publisher("/ur5_pybullet/gripper_open_ratio", Float64, queue_size=1)
        
        self.navi_target = [np.array([0, 0, 0]), 0.0]
        self.robot_pos = [np.array([0, 0, 0]), 0.0]
        self.target_pos = np.array([[0, 0, 0], [0, 0, 0]]).astype(Float64) #抓取对象
        self.gripper_ratio = 1.0
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        #rospy.Subscriber("/ur5_pybullet/target_position", Float64MultiArray, self.target_position_callback)
        #多线程？
        self.send_command_thread = threading.Thread(target=self.send_command_thread_fun)
        self.send_command_thread.setDaemon(True)
        self.send_command_thread.start()
    
    #def target_position_callback(self, msg:Float64MultiArray):
        #self.target_pos[0] = msg.data[0:3]
        #self.target_pos[1] = msg.data[3:6]
    
    def odom_callback(self, msg:Odometry):
        #订阅odom后直接处理其数据
        self.robot_pos[0] = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y ,msg.pose.pose.position.z])
        orientation = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        self.robot_pos[1] = Rotation.from_quat(orientation).as_euler('xyz')[2]
        
    def send_navigation_target(self, pos):
        #发布navigation目标位置
        position = pos[0]
        yaw = pos[1]
        msg = PoseStamped()
        msg.header.frame_id = "world"
        msg.header.stamp = rospy.Time.now()
        msg.pose.position.x = position[0]
        msg.pose.position.y = position[1]
        msg.pose.position.z = position[2]
        
        orientation = Rotation.from_euler('xyz', [0, 0, yaw], degrees=False).as_quat()
        msg.pose.orientation.x = orientation[0]
        msg.pose.orientation.y = orientation[1]
        msg.pose.orientation.z = orientation[2]
        msg.pose.orientation.w = orientation[3]
        self.movebase_goal_pub.publish(msg)
    
    def send_command_thread_fun(self):
        #为什么多线程？
        while not rospy.is_shutdown():
            #Navigation模块如果不发送目标，它就不运行，不动
            #只要ros运行，它就一直循环
            #发布目标位置，需要一直发布，因为这需要时间实现，并且如果不使用多线程，程序会一直卡在这里，但是navigation模块确实要一直不停的接收到目标位置
            if self.navi_target:
                self.send_navigation_target(self.navi_target)
            self.moveit_interface.publish_status()
            self.set_gripper_ratio(self.gripper_ratio)
            time.sleep(0.2)
    
    #moveit 略过
    def set_arm_joint_configuration(self, config:JointConfiguration):
        self.moveit_interface.go_to_joint_state(config.value)
    
    #pybullet接口执行机械臂运动
    #def grab_target(self):
        #p.connect(p.GUI)
        #robot_id = config_li.get_shared_variable()

    #    input_file = "src/ur5_pybullet_ros/script/controller/end-pos.txt"
    #    num_lines_to_read = 40
    #    with open(input_file, 'r') as file:
    #        lines = file.readlines()

    #    total_lines = len(lines)
    #    interval = total_lines // num_lines_to_read

    #    selected_lines = [lines[i] for i in range(0, total_lines, interval)][:num_lines_to_read]
    #    for line in selected_lines:
    #            # 移除行尾的换行符和空格
    #            line = line.strip()
    #            # 去掉方括号并分割字符串
    #            elements = line[1:-1].split(',')
    #            # 将字符串元素转换为float并创建numpy数组
    #            array = np.array([float(element) for element in elements], dtype=np.float64)
    #            target_pose = array
    #            base_orientation =  Rotation.from_euler('xyz', [0, 0, self.robot_pos[1]], degrees=False).as_quat()
    #            base_position = self.robot_pos[0] + np.array([0, 0, 0.06])
    #            target_pose_r = np.dot(np.linalg.inv(Rotation.from_quat(base_orientation).as_matrix()), (target_pose - base_position))
    #            self.moveit_interface.go_to_pose_goal(target_pose_r, Rotation.from_euler('xyz', [0, 0.5 * np.pi, 0], degrees=False).as_quat(), "base_link")
    #            time.sleep(0.01)

    def grab_target(self):
        #p.connect(p.GUI)
        #robot_id = config_li.get_shared_variable()

        #input_file = "src/ur5_pybullet_ros/script/controller/success-pos.txt"
        #input_file = "src/ur5_pybullet_ros/script/controller/success-pos3.txt"
        #input_file = "src/ur5_pybullet_ros/script/controller/success-pos4.txt"
        input_file = "src/ur5_pybullet_ros/script/controller/success-pos5.txt"
        with open(input_file, 'r') as file:
            for line in file:
                # 移除行尾的换行符和空格
                line = line.strip()
                # 去掉方括号并分割字符串
                position_str, orientation_str = line.split('],[')
                position_str = position_str.replace('[', '').replace(']', '')
                orientation_str = orientation_str.replace('[', '').replace(']', '')
                # 将字符串元素转换为float并创建numpy数组
                position = np.array([float(element) for element in position_str.split(',')], dtype=np.float64)
                orientation = np.array([float(element) for element in orientation_str.split(',')], dtype=np.float64)
                #array = np.array([float(element) for element in elements], dtype=np.float64)
                #target_pose = array
                #base_orientation =  Rotation.from_euler('xyz', [0, 0, self.robot_pos[1]], degrees=False).as_quat()
                #base_position = self.robot_pos[0] + np.array([0, 0, 0.06])
                #target_pose_r = np.dot(np.linalg.inv(Rotation.from_quat(base_orientation).as_matrix()), (target_pose - base_position))
                self.moveit_interface.go_to_pose_goal(position, orientation, "base_link")
                time.sleep(0.01)
        self.gripper_ratio = 0.5

        #pybullet接口执行机械臂运动
    #def grab_target_2(self):
        #p.connect(p.GUI)
        #robot_id = config_li.get_shared_variable()

        #input_file = "src/ur5_pybullet_ros/script/controller/success-pos2.txt"
    #    input_file = "src/ur5_pybullet_ros/script/controller/success-pos2-1.txt"
    #    with open(input_file, 'r') as file:
    #        for line in file:
    #            # 移除行尾的换行符和空格
    #            line = line.strip()
    #            # 去掉方括号并分割字符串
    #            position_str, orientation_str = line.split('],[')
    #            position_str = position_str.replace('[', '').replace(']', '')
    #            orientation_str = orientation_str.replace('[', '').replace(']', '')
    #            # 将字符串元素转换为float并创建numpy数组
    #            position = np.array([float(element) for element in position_str.split(',')], dtype=np.float64)
    #            orientation = np.array([float(element) for element in orientation_str.split(',')], dtype=np.float64)
                #array = np.array([float(element) for element in elements], dtype=np.float64)
                #target_pose = array
                #base_orientation =  Rotation.from_euler('xyz', [0, 0, self.robot_pos[1]], degrees=False).as_quat()
                #base_position = self.robot_pos[0] + np.array([0, 0, 0.06])
                #target_pose_r = np.dot(np.linalg.inv(Rotation.from_quat(base_orientation).as_matrix()), (target_pose - base_position))
    #            self.moveit_interface.go_to_pose_goal(position, orientation, "base_link")
    #            time.sleep(0.01)
    #    self.gripper_ratio = 1.0
    
    def grab_target_2(self):
        #p.connect(p.GUI)
        #robot_id = config_li.get_shared_variable()

        #input_file = "src/ur5_pybullet_ros/script/controller/success-pos2-1.txt"
        #output_file = "src/ur5_pybullet_ros/script/controller/success-pos2-1-1.txt"
        #input_file = "src/ur5_pybullet_ros/script/controller/success-pos2-1-1.txt"
        #input_file = "src/ur5_pybullet_ros/script/controller/success-pos2-1-2.txt"
        #input_file = "src/ur5_pybullet_ros/script/controller/1.txt"
        #2成功
        #input_file = "src/ur5_pybullet_ros/script/controller/2.txt"
        #
        #input_file = "src/ur5_pybullet_ros/script/controller/3.txt"
        #input_file = "src/ur5_pybullet_ros/script/controller/4.txt"
        input_file = "src/ur5_pybullet_ros/script/controller/5.txt"
        num_lines_to_read = 100
        with open(input_file, 'r') as file:
            lines = file.readlines()

        #total_lines = len(lines)
        #interval = total_lines // num_lines_to_read

        #selected_lines = [lines[i] for i in range(0, total_lines, interval)][:num_lines_to_read]
        selected_lines = lines
        #with open(output_file, 'w') as file:
        #    file.writelines(selected_lines)
        for line in selected_lines:
                # 移除行尾的换行符和空格
                line = line.strip()
                # 去掉方括号并分割字符串
                position_str, orientation_str = line.split('],[')
                position_str = position_str.replace('[', '').replace(']', '')
                orientation_str = orientation_str.replace('[', '').replace(']', '')
                # 将字符串元素转换为float并创建numpy数组
                position = np.array([float(element) for element in position_str.split(',')], dtype=np.float64)
                orientation = np.array([float(element) for element in orientation_str.split(',')], dtype=np.float64)
                self.moveit_interface.go_to_pose_goal(position, orientation, "base_link")
                time.sleep(0.01)
        self.gripper_ratio = 1.0


    def set_gripper_ratio(self, ratio):
        msg = Float64()
        msg.data = ratio
        self.grip_ratio_pub.publish(msg)
    
    #navigation是否到达目标地点
    def has_reach_target(self, dist_thres=0.30, ang_thres=10.0):
        #小车位置
        dist_err = np.linalg.norm(self.robot_pos[0] - self.navi_target[0])
        #小车底盘朝向
        ang_err = np.linalg.norm(self.robot_pos[1] - self.navi_target[1])
        return dist_err < dist_thres and ang_err < ang_thres
    
    def navigate_and_wait(self, target, timeout=50.0):
        self.navi_target = target
        wait = 0.0
        #多线程的原因
        #主程序判断是否到达，另一个线程一直发送navigate的目标，进行navigate，如果到达，那么就直接停止发送目标，并执行主程序下一个任务
        while not self.has_reach_target():
            time.sleep(0.1)
            wait += 0.1
            if wait > timeout:
                print("navigation timeout!")
                self.navi_target = None
                return
        self.navi_target = None
        time.sleep(4.0)
        
    
    def work(self):
        self.set_arm_joint_configuration(JointConfiguration.UPRIGHT)
        #如果没有到达指定位置怎么办？
        self.navigate_and_wait(PRE_WORK1_POS)
        self.set_arm_joint_configuration(JointConfiguration.DOWN_VIEW)
        #for i in range(14):
        #    joint_pos = p.getJointState(self.id,i)[0]
        self.navigate_and_wait(WORK1_POS)
        time.sleep(1)
        #方法1
        #得到env文件里机械臂各种末端执行器id：
        #我需要运行时每次都要step-simulation与否
        self.grab_target()
        time.sleep(0.5)
        self.set_arm_joint_configuration(JointConfiguration.UPRIGHT)
        #self.navigate_and_wait(PRE_WORK1_POS)
        self.navigate_and_wait(PRE_WORK2_POS)
        self.set_arm_joint_configuration(JointConfiguration.DOWN_VIEW)
        self.navigate_and_wait(WORK2_POS)
        #self.gripper_ratio = 1.0
        time.sleep(1)
        self.navigate_and_wait(WORK3_POS)
        self.grab_target_2()
        self.gripper_ratio = 1.0
        time.sleep(0.5)
        self.set_arm_joint_configuration(JointConfiguration.UPRIGHT)
        #self.navigate_and_wait(PRE_WORK2_POS)
        self.navigate_and_wait(PRE_WORK4_POS)
        self.set_arm_joint_configuration(JointConfiguration.DOWN_VIEW)
        self.navigate_and_wait(WORK4_POS)
        self.gripper_ratio = 1.0
        time.sleep(0.5)
        self.navigate_and_wait(PRE_WORK4_POS)
        self.set_arm_joint_configuration(JointConfiguration.UPRIGHT)
        
        
        
        
        
    
    
        


if __name__ == "__main__":
    rospy.init_node("task_controller")
    task_controller = TaskController()
    start_sec = time.time()
    task_controller.work()
    end = time.time()
    print("time_usage:" , end - start_sec) # 92 150 155
    # while not rospy.is_shutdown():
    #     task_controller.step()
    #     time.sleep(0.1)
    
    
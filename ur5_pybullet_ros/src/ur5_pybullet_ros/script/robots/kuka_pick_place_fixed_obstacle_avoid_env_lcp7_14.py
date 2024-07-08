import pybullet as p
import pybullet_data
import os
import sys
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from math import sqrt
import random
import time
from numpy import arange
import logging
import math
from termcolor import colored

#### 一些变量 ######
LOGGING_LEVEL = logging.INFO



class KukaReachEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    max_steps_one_episode = 1500

    def __init__(self, is_render=False, is_good_view=False):

        self.is_render = is_render
        self.is_good_view = is_good_view

        if self.is_render:
            
            p.connect(p.GUI)

            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            
            p.connect(p.DIRECT)

        self.gripper_length = 0.257
        self.x_low_obs = 0.15
        self.x_high_obs = 0.85
        self.y_low_obs = -0.6
        self.y_high_obs = 0.7
        self.z_low_obs = 0
        self.z_high_obs = 1.2

        self.x_low_action = -0.4
        self.x_high_action = 0.4
        self.y_low_action = -0.4
        self.y_high_action = 0.4
        self.z_low_action = -0.6
        self.z_high_action = 0.4
        self.a_low_action = -1
        self.a_high_action = 1
        self.timeStep = 1. / 240.
        self.maxVelocity = 1.35
        self.maxForce = 600.
        self.fingerAForce = 2
        self.fingerBForce = 2.5
        self.fingerTipForce = 2
        self._actionRepeat = 200
        self.current_a = 0
        self.EndEffectorPos = [0.537, 0.0, 0.5]
        self.graspSuccess = 0
        self.attempted_grasp = False
        self.robot_state = None
        self.distance_before = 0
        self.distance_before_obj_target = 0
        self.distance_ori_before_obj_target = 0
        self.distance_ori_before = 0
        self.termination = False
        self.terminated = False
        self.object_hold = False
        self.action_difference = 0
        #正视图
        #p.resetDebugVisualizerCamera(cameraDistance=1.5,
        #                             cameraYaw=0,
        #                             cameraPitch=-40,
        #                             cameraTargetPosition=[0.55, -0.35, 0.2])
        
        #右视图
        p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                     cameraYaw= 100,
                                     cameraPitch=-40,
                                     cameraTargetPosition=[0.55, -0.35, 0.2])

        self.action_space = spaces.Box(low=np.array([
                                        self.x_low_action, 
                                        self.y_low_action, 
                                        self.z_low_action
                                        #self.a_low_action
                                        ]),
                                       high=np.array([
                                           self.x_high_action,
                                           self.y_high_action,
                                           self.z_high_action
                                           #self.a_high_action
                                       ]),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([self.x_low_obs, 2*self.y_low_obs, self.z_low_obs, self.x_low_obs, 2*self.y_low_obs, self.z_low_obs, self.x_low_obs, self.y_low_obs, self.z_low_obs, self.x_low_obs, self.y_low_obs, self.z_low_obs, -3, -3, -3, -3, -3, -3, 0.25]),
            high=np.array([self.x_high_obs, 2*self.y_high_obs, self.z_high_obs, self.x_high_obs, 2*self.y_high_obs, self.z_high_obs, self.x_high_obs, self.y_high_obs, self.z_high_obs, self.x_high_obs, self.y_high_obs, self.z_high_obs, 3, 3, 3, 3, 3, 3, 1]),
            dtype=np.float32)
        self.step_counter = 0

        #机械臂末端执行器
        self.end_effector_index = 6
        #gripper
        self.gripper_index = 7


        self.urdf_root_path = pybullet_data.getDataPath()
        # lower limits for null space
        self.lower_limits = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        # upper limits for null space
        self.upper_limits = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        # joint ranges for null space
        self.joint_ranges = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        # restposes for null space
        self.rest_poses = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        # joint damping coefficents
        self.joint_damping = [
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001
        ]

        self.init_joint_positions = [
            0.006418, 0.413184, -0.011401, -2.789317, 0.005379, 2.337684,
            -0.006539, 0.000048, -0.299912, 0.000000, -0.000043, 0.299960,
            0.000000, -0.000200
        ]

        self.orientation = p.getQuaternionFromEuler(
            [0., -math.pi, math.pi / 2.])

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def random_pos(self):
        # 使用 self.np_random 生成一个随机动作
        # 假设动作是在 [0, 1, 2] 中随机选择一个
        action = self.np_random.choice([0, 1, 2])
        return action
    def random_obs(self):
        # 使用 self.np_random 生成一个随机动作
        # 假设动作是在 [0, 1, 2,3, 4] 中随机选择一个
        #action = self.np_random.choice([0, 1, 2, 3])
        #action = self.np_random.choice([2, 3, 4, 5, 6])
        action = self.np_random.choice([2, 6])
        return action
    
    def reset(self):
        #p.connect(p.GUI)
        self.decay_factor = 1
        self.step_counter = 0
        self.attempted_grasp = False
        self.object_hold = False
        self.distance_before = 0
        self.distance_before_obj_target = 0
        self.distance_ori_before_obj_target = 0
        self.distance_hand_obj_ori_before = 0
        self.action_difference = 0
        self.id_obstacle_pos = self.random_obs()
        #self.id_obstacle_pos = 2
        #velocity_all = [[0.005, 0, 0], [0.005, 0.003, 0.002], [-0.005, 0.003, 0.002]]
        #self.velocity = velocity_all[self.id_obstacle_pos]
        self.velocity = [0.002, 0, 0]
        p.resetSimulation()
        #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.setGravity(0, 0, -10)

        #box_postion = 
        object_postion_all = [[0.5, -0.35, 0], [0.35, -0.35, 0], [0.6, -0.35, 0]]
        target_postion_all = [[0.5, 0.45, 0.08],[0.35, 0.45, 0.08],[0.6, 0.45, 0.08]]
        #                                        leftup            right up up            目标上方                                                  对象附近
        obstacle_postion_all = [[0.4, 0.3, 0.13], [0.2, -0.25, 0.1] , [0.1, 0.6, 0.25], [0.35, 0.5, 0.1], [0.13, 0.5, 0.1], [0.55, 0.5, 0.1], [0.4, -0.5, 0.1]]
        #self.box_postion =
        object_postion = object_postion_all[self.random_pos()]
        target_postion = target_postion_all[self.random_pos()]
        #self.id_obstacle_pos = self.random_pos()
        #self.id_obstacle_pos = 2
        obstacle_postion = obstacle_postion_all[self.id_obstacle_pos]
        #print("id_object: ", self.random_pos())
        #print("id_target: ", self.random_pos())
        #添加目标圆点
        
        # 创建目标点的位置
        #self.target_position =[
        #    random.uniform(self.x_low_obs + 0.15  ,
        #    self.x_high_obs - 0.1),
        #    random.uniform(self.y_low_obs + 0.1,
        #    self.y_high_obs - 0.1), 0.01
        #]
        self.target_position = target_postion

        #min_distance = 0.3
        #max_distance = 0.4

        #while True:
        #    object_position = [
        #        random.uniform(self.x_low_obs + 0.15, self.x_high_obs - 0.1),
        #        random.uniform(self.y_low_obs + 0.1, self.y_high_obs - 0.1),
        #        0.01
        #    ]
        #    distance = np.linalg.norm(np.array(self.target_position) - np.array(object_position))
        #    if min_distance <= distance <= max_distance:
        #        break

        # 定义圆形的半径和颜色
        circle_radius = 0.04
        circle_color = [0, 1, 0]  # 绿色

        # 生成圆形上的点
        num_points = 100
        circle_points = [
            [self.target_position[0] + circle_radius * np.cos(2 * np.pi * i / num_points),
            self.target_position[1] + circle_radius * np.sin(2 * np.pi * i / num_points),
            self.target_position[2]] for i in range(num_points)]
        
        # 添加圆形上的线段，以形成圆
        for i in range(num_points):
            p.addUserDebugLine(lineFromXYZ=circle_points[i], lineToXYZ=circle_points[(i + 1) % num_points], lineColorRGB=circle_color, lineWidth=2)




        #建立障碍物圆球
        radius = 0.13
        colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        visSphereId = p.createVisualShape(p.GEOM_SPHERE, radius=radius)
        self.box_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colSphereId, 
                             baseVisualShapeIndex=visSphereId, 
                             basePosition= obstacle_postion)
        

        #建立矩形障碍物
        # 定义盒子的尺寸（矩形尺寸）
        #box_size = [0.18, 0.001, 0.15]  # 盒子的长、宽、高

        # 创建盒子的碰撞形状和可视化形状
        #collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=box_size)
        #visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=box_size)

        #创建多体
        #self.box_id = p.createMultiBody(baseMass=0, 
        #                           baseCollisionShapeIndex=collision_shape_id, 
        #                           baseVisualShapeIndex=visual_shape_id, 
        #                           basePosition=[0.4, 0.12, 0])

        
        #这些是周围那些白线，用来观察是否超过了obs的边界
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_low_obs, 0],
            lineToXYZ=[self.x_low_obs, self.y_low_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_high_obs, 0],
            lineToXYZ=[self.x_low_obs, self.y_high_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs, self.y_low_obs, 0],
            lineToXYZ=[self.x_high_obs, self.y_low_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs, self.y_high_obs, 0],
            lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_high_obs])

        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_low_obs, self.z_high_obs],
            lineToXYZ=[self.x_high_obs, self.y_low_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_high_obs, self.z_high_obs],
            lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_low_obs, self.z_high_obs],
            lineToXYZ=[self.x_low_obs, self.y_high_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs, self.y_low_obs, self.z_high_obs],
            lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_high_obs])
        
        #p.addUserDebugLine([self.x_low_obs, 0.003, self.z_low_obs], [self.x_high_obs, 0.003, self.z_low_obs], [1, 0, 0])

        p.loadURDF(os.path.join(self.urdf_root_path, "plane.urdf"),
                   basePosition=[0, 0, -0.65])
        
        #the return of loadSDF is a tuple(0, )
        self.kuka_id = p.loadSDF(os.path.join(self.urdf_root_path,
                                               "kuka_iiwa/kuka_with_gripper2.sdf"))[0]
        p.loadURDF(os.path.join(self.urdf_root_path, "table/table.urdf"),
                   basePosition=[0.5, 0.1, -0.65])

    

        #抓取对象随机出现，并且竖直摆放
        self.object_id = p.loadURDF(os.path.join(self.urdf_root_path,
                                                "random_urdfs/000/000.urdf"),
                                    basePosition= object_postion,
                                    baseOrientation=p.getQuaternionFromEuler([0 , 0 , 0.5 * math.pi]))

        #just for test grasp,同一个物体会一直出现在爪子正下方,障碍物竖直摆放
        #self.object_id = p.loadURDF(os.path.join(self.urdf_root_path,
        #                                         "random_urdfs/000/000.urdf"),
        #                            basePosition=[
        #                                0.51,
        #                                0.003, 
        #                                0.01
        #                            ],
        #                            baseOrientation=p.getQuaternionFromEuler([0 , 0, 0.5 * math.pi]))



        self.num_joints = p.getNumJoints(self.kuka_id)

        #初始化各关节的位置
        for i in range(self.num_joints):
            p.resetJointState(
                bodyUniqueId=self.kuka_id,
                jointIndex=i,
                targetValue=self.init_joint_positions[i],
            )
            #joint_info = p.getJointInfo(self.kuka_id, i)
            #print(joint_info)
            #print("kuka_id:", i)
            #print(joint_info)
            #print("init_joint_positions:", self.init_joint_positions[i])
        #time.sleep(10)
        
        p.resetJointState(self.kuka_id, 8, 0)
        p.resetJointState(self.kuka_id, 10, 0)
        p.resetJointState(self.kuka_id, 11, 0)
        p.resetJointState(self.kuka_id, 13, 0)

        p.stepSimulation()
        obs = self._low_dim_full_state()
        return obs
    
    def _low_dim_full_state(self):
        pos_obj, ori_obj = p.getBasePositionAndOrientation(self.object_id)
        pos_obs = p.getBasePositionAndOrientation(self.box_id)[0]
        pos_gripper = p.getLinkState(self.kuka_id, self.gripper_index)[4]
        pos_obj = np.array(pos_obj, dtype=np.float32)
        pos_gripper = np.array(pos_gripper, dtype=np.float32)
        pos_target = np.array(self.target_position, dtype= np.float32)
        pos_obs = np.array(pos_obs)
        pos_relative_obj_gripper = pos_gripper - pos_obj
        pos_relative_target_obj = pos_obj - pos_target 
        pos_relative_obs_gripper = pos_gripper - pos_obs
        pos_relative_obs_obj = pos_obj - pos_obs
        #obstacle
        closest_points_kuka_box = p.getClosestPoints(self.kuka_id, self.box_id, 0.15)
        closest_points_obj_box = p.getClosestPoints(self.object_id, self.box_id, 0.15)
        if closest_points_kuka_box:
            closest_point_info = closest_points_kuka_box[0]
            robot_closest_point_position = np.array(closest_point_info[5])  # Point on Robot
            obstacle_closest_point_position = np.array(closest_point_info[6]) # point on obstacle
            pos_relative_robot_obs_point = np.array(obstacle_closest_point_position - robot_closest_point_position)
        else:
            pos_relative_robot_obs_point = np.array([2, 2, 2])

        if closest_points_obj_box:
            closest_point_info = closest_points_obj_box[0]
            object_closest_point_position = np.array(closest_point_info[5])  # Point on object
            obstacle_closest_point_position = np.array(closest_point_info[6]) # point on obstacle
            pos_relative_obj_obs_point = np.array(obstacle_closest_point_position - object_closest_point_position)
        else:
            pos_relative_obj_obs_point = np.array([2, 2, 2])


        full_state = np.concatenate((pos_relative_obj_gripper, pos_relative_target_obj, pos_relative_obs_gripper, pos_relative_obs_obj, pos_relative_robot_obs_point, pos_relative_obj_obs_point))
        #full_state = np.append(full_state, self.action_difference)
        full_state = np.append(full_state, self.decay_factor)
        return full_state

    def step(self, action):       
        dv = 0.006
        #dv = 0.1
        dx = action[0] * dv
        dy = action[1] * dv
        #dz = -0.025
        dz = action[2] * dv
        #grasp沿x轴水平
        #da = 0
        #grasp沿y轴水平
        da = 0.5 * math.pi
        grasp_angle = 0.0
        action = [dx, dy, dz, da, grasp_angle]

        #Perform commanded action
        self.step_counter += 1
        state = np.array(p.getLinkState(self.kuka_id, self.end_effector_index)[4]).astype(np.float32)
        current_EndEffectorPos = state

        self.EndEffectorPos[0] = current_EndEffectorPos[0] + action[0]    
        self.EndEffectorPos[1] = current_EndEffectorPos[1] + action[1]
        self.EndEffectorPos[2] = current_EndEffectorPos[2] + action[2] 
        self.current_a += action[3]  # angel
        #state = np.array(p.getLinkState(self.kuka_id, self.end_effector_index)[4]).astype(np.float32)
        #end_effector_pos = state


        for i in range(100):
            initial_velocity_all = [[0.002, 0, 0.00], [0.0025, 0.0025, 0.0005], [0.0025, -0.0025, -0.0005],[0.0015, 0, 0.000], [0.0015, 0, 0.000], [0.0015, 0, 0.000], [0.002, 0, 0.000]]  # 沿Y轴移动
            back_velocity_all = [[-0.002, 0, -0.00], [-0.0025, -0.0025, -0.0005], [-0.0025, 0.0025, 0.0005],[-0.0015, 0, 0.000], [-0.0015, 0, 0.000], [-0.0015, 0, 0.000],[-0.002, 0, 0.000]]
            initial_velocity = initial_velocity_all[self.id_obstacle_pos]
            back_velocity = back_velocity_all[self.id_obstacle_pos]
            pose_obstacle = p.getBasePositionAndOrientation(self.box_id)[0]

            #要用位置判断，而不是频率，否则小球只会往出现次数最多的那个速度方向，而不会即时转向
            if pose_obstacle[0]  <= 0.2:
                self.velocity = initial_velocity
            elif pose_obstacle[0] >= 1:
                self.velocity = back_velocity

            p.resetBaseVelocity(self.box_id, linearVelocity = self.velocity)

            self.applyAction(self.EndEffectorPos, self.current_a, action[4])
            p.stepSimulation()
        
            state = np.array(p.getLinkState(self.kuka_id, self.end_effector_index)[4]).astype(np.float32)
            end_effector_pos = state
            #print("完成步进模拟,此时repeat-action：",i)
            if self.render:
                #time.sleep(0.001)
                pass
            if self.step_counter > self.max_steps_one_episode:
                break
            self.object_state = np.array(
            p.getBasePositionAndOrientation(self.object_id)[0]).astype(
                np.float32)

            if self.distance_hand_obj_ori_before <= 0.03 and end_effector_pos[2] <= 0.318 and self.object_state[2] <= 0.01 and self.object_hold == False:
                finger_angle = 0.0
                #print("松爪")
                #time.sleep(2)
                for _ in range(100):
                    #grasp_action
                    self.applyAction(end_effector_pos, self.current_a, fingerAngle=finger_angle)
                    p.stepSimulation()
                    if self.render:
                        time.sleep(0.02)
                    finger_angle += 0.8 / 100.
                    if finger_angle > 0.3:
                        finger_angle = 0.3 
                    end_effector_pos[2] = end_effector_pos[2] - 0.0008
                    self.robot_state = np.array(p.getLinkState(self.kuka_id, self.end_effector_index)[4]).astype(np.float32)
                    if self.robot_state[2] <= 0.259:
                        break
                # If we are close to the bin, attempt grasp.
                state = np.array(p.getLinkState(self.kuka_id, self.end_effector_index)[4]).astype(np.float32)
                end_effector_pos = state

                #maxDist = 0.005
                #closestPoints = p.getClosestPoints(self.object_id, self.kuka_id, maxDist)
                #0.258：grasp
                #if end_effector_pos[2] <= 0.259:
                #if (len(closestPoints)):
                
                for _ in range(100):
                        #grasp_action
                    self.applyAction(end_effector_pos, self.current_a, fingerAngle=finger_angle)
                    p.stepSimulation()
                    if self.render:
                        #pass
                        time.sleep(0.02)
                    finger_angle -= 0.3 / 100.
                    if finger_angle < 0:
                        finger_angle = 0
                    # up the gripple a little and grasp
                    #print("grasp angle1", finger_angle)
                    #end_effector_pos = np.array(end_effector_pos)
                for i in range(200):
                    end_effector_pos[2] = end_effector_pos[2] + 0.002
                        #print("正在grasp：",i)
                        #time.sleep(0.2)
                    self.applyAction(end_effector_pos, da=self.current_a, fingerAngle=finger_angle)
                    p.stepSimulation() #开启步进仿真
                    if self.render:
                        time.sleep(0.02)
                    finger_angle -= 0.03 / 100.
                    if finger_angle < 0:
                        finger_angle = 0
                        #print("grasp angle2", finger_angle)
                    object_pos = np.array(p.getBasePositionAndOrientation(self.object_id)[0])
                    if object_pos[2] >= 0.2:
                           break
                        #end_effector_pos = np.array(p.getLinkState(self.kuka_id, self.end_effector_index)[4])#机械臂运动是实时的，在运动是不能查看
                        #if end_effector_pos[2] >= 0.5:
                        #    break
                #self.EndEffectorPos = end_effector_pos
                self.EndEffectorPos = state = np.array(p.getLinkState(self.kuka_id, self.end_effector_index)[4]).astype(np.float32)
                self.attempted_grasp = True
                #print("对象开始动了")

                
        return self._reward()
        
    def applyAction(self, pos, da, fingerAngle):
        #gripper绕z轴转角
        self.current_endEffectorAngle = da
        self.current_endEffectorAngle = np.clip(self.current_endEffectorAngle, a_min=np.radians(-90),
                                                    a_max=np.radians(90))  # constrain in [-pi/2, pi/2]
        #直接将末端执行器的方向设置成水平向下：绕y转-pi，至于绕z转不是很重要
        #orn = p.getQuaternionFromEuler([0., -math.pi, math.pi / 2.])
        orn = p.getQuaternionFromEuler([0, -math.pi, 0])  # -math.pi,yaw])
        jointPoses = p.calculateInverseKinematics(self.kuka_id, self.end_effector_index, pos, orn,
                                                              jointDamping=self.joint_damping)

        #useSimulation
        for i in range(self.gripper_index):
            p.setJointMotorControl2(bodyUniqueId=self.kuka_id, jointIndex=i, controlMode=p.POSITION_CONTROL,
                                    maxVelocity=self.maxVelocity, targetVelocity=0,
                                    targetPosition=jointPoses[i], force=self.maxForce, positionGain=0.03,
                                    velocityGain=1)
        

        #for i in range(self.gripper_index):  # self.numJoints
        #    p.resetJointState(self.kuka_id, i, jointPoses[i])

        # fingers
        p.setJointMotorControl2(self.kuka_id, 7, p.POSITION_CONTROL, targetPosition=self.current_endEffectorAngle,
                                    force=self.maxForce)
        p.setJointMotorControl2(self.kuka_id, 8, p.POSITION_CONTROL, targetPosition=-fingerAngle,
                                    force=self.fingerAForce)
        p.setJointMotorControl2(self.kuka_id, 11, p.POSITION_CONTROL, targetPosition=fingerAngle,
                                    force=self.fingerBForce)

        p.setJointMotorControl2(self.kuka_id, 10, p.POSITION_CONTROL, targetPosition=0, force=self.fingerTipForce)
        p.setJointMotorControl2(self.kuka_id, 13, p.POSITION_CONTROL, targetPosition=0, force=self.fingerTipForce)
        

    def _reward(self):

        #一定注意是取第4个值，请参考pybullet手册的这个函数返回值的说明；末端执行器位于gripper的底座，Z轴向上为正
        self.robot_state = np.array(p.getLinkState(self.kuka_id, self.end_effector_index)[4]).astype(np.float32)
        #self.robot_state[2] -= self.gripper_length

        self.object_state = np.array(
            p.getBasePositionAndOrientation(self.object_id)[0]).astype(
                np.float32)
        #print("object_state", self.object_state)
        square_dx = np.float32((self.robot_state[0] - self.object_state[0])**2)
        square_dy = np.float32((self.robot_state[1] - self.object_state[1])**2)
        square_dz = np.float32((self.robot_state[2] - self.object_state[2] - self.gripper_length)**2)

        #hand move to object and close to object
        #用机械臂末端和物体的距离作为奖励函数的依据
        distance_hand_obj = np.float32(sqrt(square_dx + square_dy + square_dz))
        #只计算机械臂末端与物体在x-y平面的距离，为方向距离
        distance_hand_obj_ori = np.float32(sqrt(square_dx + square_dy))

        #object moving to target
        reward_ori_target = 0
        #self.object_vel = np.array(p.getBaseVelocity(self.object_id)[0]).astype(np.float32)
        #避免振动影响
        #if all(abs(vel) < 0.00001 for vel in self.object_vel):
        #    self.object_vel = np.array([0,0,0])
        #self.target_position = np.array(self.target_position, dtype=np.float32)
        #pos_target_object = self.target_position - self.object_state
        #distance_obj_target = np.linalg.norm(pos_target_object)
        #direction_obj_target = pos_target_object / (1e-6 + distance_obj_target)
        #reward_ori_target = 100 * np.dot(self.object_vel, direction_obj_target)


        #计算机械臂robot与障碍物最近点的距离
        closest_points_kuka_box = p.getClosestPoints(self.kuka_id, self.box_id, 0.3)

        # 假设最近距离初始为无穷大
        min_distance_kuka_box = float('inf')
        #closest_pair = None

        for point in closest_points_kuka_box:
            if point[8] < min_distance_kuka_box:
                min_distance_kuka_box = point[8]

        #计算抓取对象与障碍物最近点的距离
        closest_points_obj_box = p.getClosestPoints(self.object_id, self.box_id, 0.3)

        # 假设最近距离初始为无穷大
        min_distance_obj_box = float('inf')
        #closest_pair = None

        for point in closest_points_obj_box:
            if point[8] < min_distance_obj_box:
                min_distance_obj_box = point[8]

        #print("min_distance: ", min_distance)
        contact_points_kuka_box = p.getContactPoints(bodyA=self.kuka_id, bodyB=self.box_id)
        contact_points_obj_box = p.getContactPoints(bodyA=self.object_id, bodyB=self.box_id)
        contact_points_kuka_obj = p.getContactPoints(bodyA=self.object_id, bodyB=self.kuka_id)
        #if not contact_points:
        #    print("not collision")
        #else:
        #    print("collision")
        #print("contact_points: ", contact_points )
        

        #time.sleep(0.01)
        #print("dis_obj_target: ", distance_obj_target)
        #print("object_vel: ", self.object_vel)
        #print("distance_hand_obj", distance_hand_obj)
        #计算物体到target的距离
        square_dx_obj_target = np.float32((self.object_state[0] - self.target_position[0])**2) 
        square_dy_obj_target = np.float32((self.object_state[1] - self.target_position[1])**2)
        square_dz_obj_target = np.float32((self.object_state[2] - self.target_position[2])**2)
        distance_obj_target = np.float32(sqrt(square_dx_obj_target + square_dy_obj_target + square_dz_obj_target))
        distance_obj_target_ori = np.float32(sqrt(square_dx_obj_target + square_dy_obj_target))
        #print(self.distance)
        x = np.float32(self.robot_state[0])
        y = np.float32(self.robot_state[1])
        z = np.float32(self.robot_state[2] - self.gripper_length)

    
        #如果机械比末端超过了obs的空间，也视为done，而且会给予一定的惩罚
        out_of_obs_robot = bool(x < self.x_low_obs or x > self.x_high_obs
                          or y < self.y_low_obs or y > self.y_high_obs
                          or z < self.z_low_obs or z > self.z_high_obs) 

        out_of_obs = out_of_obs_robot 
        reward_done = 0
        reward_ori_obj = 0
        reward_clos_obj = 0
        reward_clos_target = 0
        reward_target_hit = 0
        reward_collision = 0
        reward_collision_obj_box = 0
        reward_collision_kuka_box = 0
        reward_action = 0
        #print("step_counter", self.step_counter)
        if out_of_obs:
            reward_done = -35
            self.terminated = True

        elif self.step_counter > self.max_steps_one_episode:
            reward_done = -35
            self.terminated = True

        elif contact_points_kuka_box or contact_points_obj_box:
            reward_done = -25
            self.terminated = True

        elif self.object_state[2] > 0.1 and self.attempted_grasp == True:
            self.graspSuccess += 1
            reward_done = 15
            self.attempted_grasp = False
            self.terminated = False
            self.object_hold = True

        elif self.attempted_grasp == True:
            reward_done = -0.002
            self.attempted_grasp = False
            self.terminated = False

        else:
            reward_done = 0
            self.terminated = False

        if contact_points_kuka_obj:
            self.object_hold = True
        else:
            self.object_hold = False

        
        if min_distance_kuka_box <= 0.04:
            reward_collision_kuka_box = np.log(0.1 + min_distance_kuka_box) * 0.01
        if min_distance_obj_box <= 0.04 and self.object_hold :
            reward_collision_obj_box =  np.log(0.1 + min_distance_obj_box) * 0.01
        if min_distance_obj_box <= 0.05 and self.object_hold :
            #reward_action = -(abs(distance_obj_target - self.distance_before_obj_target) ** 6) * 10
            reward_action = 0 
            self.action_difference = abs(distance_obj_target - self.distance_before_obj_target) ** 6

        if (distance_obj_target <= 0.055):
            reward_target_hit = 80
            self.terminated = True
            fingerAngle = 0
            for _ in range(50):
                self.applyAction(self.robot_state, self.current_a, fingerAngle)
                p.stepSimulation()
                fingerAngle += 0.01
                if fingerAngle >= 0.3:
                    fingerAngle = 0.3
                time.sleep(0.1)

        #if self.object_state[2] >= 0.005 and (self.distance_ori_before_obj_target > distance_obj_target_ori or distance_obj_target_ori <= 0.01):
        #    reward_ori_target = (1/(1 + distance_obj_target_ori)) * 0.008
        #elif self.object_state[2] >= 0.005 and self.distance_ori_before_obj_target <= distance_obj_target_ori:
        #    reward_ori_target = -distance_obj_target_ori * 0.003


        if self.distance_hand_obj_ori_before > distance_hand_obj_ori and self.object_hold == False:
            reward_ori_obj = (0.5/(0.5 + distance_hand_obj_ori)) * 0.012
        elif self.distance_hand_obj_ori_before == distance_hand_obj_ori and self.object_hold == False:
            reward_ori_obj = 0
        elif self.distance_hand_obj_ori_before < distance_hand_obj_ori and self.object_hold == False:
            reward_ori_obj = - distance_hand_obj_ori * 0.024

        if self.distance_before > distance_hand_obj and  self.object_hold == False:
            reward_clos_obj = (0.5 / (0.5 + distance_hand_obj)) * 0.06
        elif self.distance_before == distance_hand_obj and  self.object_hold == False:
            reward_clos_obj = 0
        elif self.distance_before < distance_hand_obj and  self.object_hold == False:
            reward_clos_obj = -distance_hand_obj * 0.02


        if self.distance_before_obj_target > distance_obj_target  and self.object_state[2] >= 0.005:
            reward_clos_target = (0.1/(0.1 + distance_obj_target)) * 0.036 * 5
        elif self.distance_before_obj_target < distance_obj_target and self.object_state[2] >= 0.005:
            reward_clos_target = -distance_obj_target * 0.024 * 0.5 * 5
        
        if self.distance_before_obj_target == distance_obj_target  and self.object_state[2] >= 0.005:
            reward_clos_target = -distance_obj_target * 0.024 * 0.04 

        #加一个时间衰减函数
        min_decay_rate = 0.25
        self.decay_factor =  max(min_decay_rate, 1 - (self.step_counter / self.max_steps_one_episode))

        reward_collision = reward_collision_kuka_box + reward_collision_obj_box
        reward = reward_ori_obj + reward_clos_obj + reward_ori_target + reward_clos_target + reward_collision 
        reward = reward * self.decay_factor
        reward += reward_done + reward_target_hit

        self.distance_before = distance_hand_obj
        self.distance_hand_obj_ori_before = distance_hand_obj_ori
        self.distance_before_obj_target = distance_obj_target
        self.distance_ori_before_obj_target = distance_obj_target_ori
        obs = self._low_dim_full_state()
        #done = self.termination
        done = self.terminated
        debug = {
            'is_success': self.graspSuccess,
            'distance ' : distance_hand_obj
        }
        #reward = reward_close + reward_grasp + reward_terminated
        return obs, reward, done, debug
      

    def close(self):
        p.disconnect()

if __name__ == '__main__':
    # 这一部分是做baseline，即让机械臂随机选择动作，看看能够得到的分数
    env = KukaReachEnv(is_render=True,is_good_view=True)
    #print(env)
    #print(env.observation_space.shape)
    # print(env.observation_space.sample())
    # print(env.action_space.sample())
    #print(env.action_space.shape)
    #obs = env.reset()
    #print(obs)
    sum_reward=0
    for i in range(100):
        env.reset()
        for i in range(1000):
            time.sleep(0.01)
            #print("准备产生action")
            action=env.action_space.sample()
            #action=np.array([0,0,0.47-i/1000])
            #print("得到sample-action",i)
            #print("action:",action)
            #只有当step()函数执行完，程序才会进行下一步循环
            obs,reward,done,info=env.step(action)
          #  print("i={},\naction={},\nobs={},\ndone={},\n".format(i,action,obs,done,))
            #print(colored("reward={},info={}, i={}, action={}".format(reward,info,i,action),"cyan"))
           # print(colored("info={}".format(info),"cyan"))
            sum_reward+=reward
            if done:
                break
            #time.sleep(1)
    print()
    print(sum_reward)
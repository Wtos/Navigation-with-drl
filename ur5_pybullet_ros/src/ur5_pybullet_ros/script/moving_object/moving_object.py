#!/usr/bin/env python3
import pybullet as p
import numpy as np
import time
import os
import pybullet_data

class MovingObject():
    def __init__(self, id, start_pos, end_pos, period):
        self.id = id
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.period = period
    
    def update_position(self, t):
        ratio = 1.0 - np.fabs(((t % (2 * self.period) - self.period) / self.period))
        pos =  ratio * np.array(self.end_pos) + (1.0 - ratio) * np.array(self.start_pos)
        p.resetBasePositionAndOrientation(self.id, pos, [0, 0, 0, 1])
    

if __name__ == "__main__":
    p.connect(p.GUI) # p.connect(p.DIRECT)
    p.setRealTimeSimulation(1)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF('plane.urdf', [0, 0, 0], [0, 0, 0, 1])
    udrf_path = os.path.dirname(os.path.abspath(__file__)) + "/../urdf/objects/"
    id = p.loadURDF(udrf_path + 'cylinder.urdf', [-1.0, 1.4, 0.5], [0, 0, 0, 1])
    moving_object = MovingObject(id, [0, 0, 0], [1, 1, 0], 1)
    t = 0
    while True:
        p.stepSimulation()
        time.sleep(0.001)
        t += 0.001
        moving_object.update_position(t)
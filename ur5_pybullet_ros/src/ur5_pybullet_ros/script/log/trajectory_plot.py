import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import os

resolution = 0.04 
origin = (-50.0, -50.0) 


path = os.path.dirname(os.path.abspath(__file__)) + "/env_log/" + "20240408072305.csv"
data = pd.read_csv(path)
map_img = mpimg.imread('/root/catkin_ws/src/navi_slam/map/room.pgm')
chassis_x = data['chassis_x'] 
chassis_y = data['chassis_y']

plt.figure()
plt.imshow(map_img, cmap='gray', extent=[origin[0], origin[0] + resolution * map_img.shape[1],
                                          origin[1], origin[1] + resolution * map_img.shape[0]])


robot_trajectory = list(zip(chassis_x, chassis_y)) 

total_distance = 0.0
for i in range(1, len(robot_trajectory)):

    distance = np.sqrt((robot_trajectory[i][0] - robot_trajectory[i-1][0])**2 +
                       (robot_trajectory[i][1] - robot_trajectory[i-1][1])**2)

    total_distance += distance

print("total_distance:", total_distance)
x_traj, y_traj = zip(*robot_trajectory)


plt.plot(x_traj, y_traj, 'r-', label='Robot Trajectory')


plt.title('Robot Trajectory on Map')
plt.legend()

plt.xlim(-8.6, 2.5)  
plt.ylim(-3.2, 3.2) 
plt.show()

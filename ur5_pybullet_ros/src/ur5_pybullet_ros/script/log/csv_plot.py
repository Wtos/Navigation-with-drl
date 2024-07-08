import pandas as pd
import matplotlib.pyplot as plt
import os

path = os.path.dirname(os.path.abspath(__file__)) + "/env_log/" + "20240408072305.csv"
data = pd.read_csv(path)
time = data['time']
ee_x = data['ee_x']
ee_y = data['ee_y']
ee_z = data['ee_z']
chassis_x = data['chassis_x']
chassis_y = data['chassis_y']


# 创建第一个绘图窗口
plt.figure(1)
plt.subplot(1, 1, 1)
plt.plot(time, ee_x, label='EE_x')
plt.plot(time, ee_y, label='EE_y')
plt.plot(time, ee_z, label='EE_z')
plt.xlabel('Time')
plt.ylabel('EE Values')
plt.title('EE Position over Time')
plt.legend()  # 添加图例
plt.grid(True)

# 创建第二个绘图窗口
plt.figure(2)
plt.subplot(1, 1, 1)
plt.plot(time, chassis_x, label='Chassis_x')
plt.plot(time, chassis_y, label='Chassis_y')
plt.xlabel('Time')
plt.ylabel('Chassis Values')
plt.title('Chassis Position over Time')
plt.legend()  # 添加图例
plt.grid(True)

plt.show()
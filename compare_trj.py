import json
import matplotlib.pyplot as plt

with open('/home/user/ws/src/ros-MonoGS/results/ros_depth/2024-04-18-10-56-28/plot/trj_2238.json', 'r') as f:
  monogs = json.load(f)

with open('/home/user/ws/src/ros-MonoGS/results/ros_depth/2024-04-18-10-56-28/chibi_trj.json', 'r') as f:
  chibi = json.load(f)

monogs_pos = [[pose[i][3] for i in range(3)] for pose in monogs['trj_est']]
monogs_x = [pos[0] for pos in monogs_pos]
monogs_z = [pos[2] for pos in monogs_pos]

chibi_x = chibi['x']
chibi_z = chibi['z']
chibi_x = chibi_x[:1300]
chibi_z = chibi_z[:1300]

print(len(monogs_x), len(chibi_x))
print(len(monogs_z), len(chibi_z))

plt.figure()
plt.plot(monogs_x, monogs_z, label='MonoGS')
plt.plot(chibi_x, chibi_z, label='GT')
plt.xlabel('X')
plt.ylabel('Z')
plt.title('XZ Trajectory')
plt.grid(True)
plt.legend()

plt.show()


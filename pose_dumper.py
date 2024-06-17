import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
import matplotlib.pyplot as plt
import json

pose_num = 3500 # for bag2_trimmed2

class PoseDumper(Node):
    def __init__(self):
        super().__init__('pose_dumper')
        self.create_subscription(
            PoseStamped,
            '/estimated_pose',
            self.pose_callback,
            1
        )
        self.is_first = True
        self.initial_pose = dict()
        self.positions = dict()
        self.positions['x'] = []
        self.positions['y'] = []
        self.positions['z'] = []

    def pose_callback(self, msg):
        if self.is_first:
            self.is_first = False
            self.initial_pose['z'] = msg.pose.position.x # MonoGSではz軸のため
            self.initial_pose['x'] = - msg.pose.position.y # MonoGSではx軸かつ反転のため
            self.initial_pose['y'] = msg.pose.position.z # MonoGSではy軸のため
            self.positions['z'].append(msg.pose.position.x)
            self.positions['x'].append(- msg.pose.position.y)
            self.positions['y'].append(msg.pose.position.z)
            return
        self.positions['z'].append(msg.pose.position.x - self.initial_pose['z'])
        self.positions['x'].append(- msg.pose.position.y - self.initial_pose['x'])
        self.positions['y'].append(msg.pose.position.z - self.initial_pose['y'])
        print(f'count: {len(self.positions["x"])}')

    def dump_pose(self):
        with open('pose.json', 'w') as f:
            json.dump(self.positions, f)

    def plot_pose(self):
        plt.figure()
        plt.plot(self.positions['x'], self.positions['z'])
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.title('XZ Trajectory')
        plt.grid(True)
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    pose_dumper = PoseDumper()
    while len(pose_dumper.positions['x']) < pose_num:
        rclpy.spin_once(pose_dumper)
    pose_dumper.destroy_node()
    rclpy.shutdown()
    pose_dumper.dump_pose()
    pose_dumper.plot_pose()

if __name__ == '__main__':
    main()

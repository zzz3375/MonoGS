import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from image_geometry import PinholeCameraModel

class CameraDataSubscriber(Node):
    def __init__(self, config):
        super().__init__('camera_data_subscriber')
        self.image_subscriber = self.create_subscription(
            Image,
            config["Dataset"]["image_topic"],
            self.image_callback,
            1
        )
        self.camera_info_subscriber = self.create_subscription(
            CameraInfo,
            config["Dataset"]["camera_info_topic"],
            self.camera_info_callback,
            1
        )
        try:
            self.depth_subscriber = self.create_subscription(
                Image,
                config["Dataset"]["depth_topic"],
                self.depth_callback,
                1
            )
        except KeyError:
            self.depth_subscriber = None
        self.is_set_camera_info = False
        self.camera_model = PinholeCameraModel()

    def camera_info_callback(self, msg):
        if self.is_set_camera_info:
            return
        self.camera_model.fromCameraInfo(msg)
        self.is_set_camera_info = True

    def image_callback(self, msg):
        self.image_msg = msg

    def depth_callback(self, msg):
        self.depth_msg = msg

    def get_camera_model(self):
        return self.camera_model

    def get_image_msg(self):
        return self.image_msg

    def get_depth_msg(self):
        return self.depth_msg

def start_node(node):
    while rclpy.ok():
        rclpy.spin_once(node)

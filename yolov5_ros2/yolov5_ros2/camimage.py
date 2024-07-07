#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
import rclpy # 导入 ROS2 的 Python 库
from rclpy.node import Node # 导入 Node 类，可以用来创建一个 ROS2 节点
from sensor_msgs.msg import Image # 导入 Image 消息类型，可以用来发布和订阅图像消息
from cv_bridge import CvBridge # 导入 CvBridge 库，该库提供了一个接口，可以将 OpenCV 的图像格式转换为 ROS 的图像消息，反之亦然
import cv2 # 导入 OpenCV 库，可以用来处理图像
import numpy as np # 导入 Numpy 库，用于进行一些数学和矩阵运算


class CameraImagePublisher(Node):   # 定义一个名为 CameraImagePublisher 的类，继承自 Node
    def __init__(self):             # 调用父类构造函数，创建一个名为 'camera_image_publisher' 的节点
        super().__init__('camera_image_publisher')
        self.publisher_ = self.create_publisher(Image, '/image_raw', 10)
        # 在 '/image_raw' 主题上创建一个发布者，消息类型为 Image，队列长度为 10
        self.timer_period = 0.016667
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        # 创建一个定时器，每隔 self.timer_period 秒执行一次 self.timer_callback 函数
        self.cap = cv2.VideoCapture(0)
        # 创建一个 VideoCapture 对象，参数 0 表示从默认的摄像头设备捕获视频
        if not self.cap.isOpened():             # 检查设备是否成功打开
            self.get_logger().error('Could not open video device')
            # 如果无法打开设备，打印错误日志
            exit(1)

        self.bridge = CvBridge()
        self.jpeg_quality = 60  # 初始JPEG质量值

    def timer_callback(self):                   # 定时器回调函数
        ret, frame = self.cap.read()            # 从设备读取一帧图像
        if ret:



            # 压缩图像为JPEG格式
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            result, encimg = cv2.imencode('.jpg', frame, encode_param)
            if not result:
                raise RuntimeError('Could not encode image!')

            # 将压缩的图像二进制数据转换为1D Numpy数组
            msg_data = np.array(encimg).tostring()

            # 使用cv_bridge构建Image消息的一部分
            msg_header = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8').header

            # 创建Image消息
            msg = Image(header=msg_header, height=frame.shape[0], width=frame.shape[1],
                        encoding='jpeg', is_bigendian=0, step=len(msg_data), data=msg_data)

            msg.encoding = '8UC1'  # 或使用 'mono8'，这样订阅者节点就不会因为无法识别而出错
            msg.height = 1
            msg.width = len(msg_data)
            msg.step = len(msg_data)
            msg.data = np.array(encimg).tobytes()

            # 发布消息
            self.publisher_.publish(msg)
            self.get_logger().info('Publishing compressed camera image')
        else:
            self.get_logger().error('Frame capture failed')

    # 注意: 在这里未实现动态调整JPEG质量的逻辑
    # 您需要提供一种机制（如ROS参数服务器）来在运行时改变self.jpeg_quality的值

    def __del__(self):
        self.cap.release()


def main(args=None):
    rclpy.init(args=args)
    camera_image_publisher = CameraImagePublisher()
    # 创建 CameraImagePublisher 对象
    try:
        rclpy.spin(camera_image_publisher)
        # 循环调用 rclpy.spin 函数，处理所有 ROS2 的事件
    except KeyboardInterrupt:
        pass
    finally:
        camera_image_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
    """



"""

import rclpy  # 引入ROS 2的Python客户端库
from rclpy.node import Node  # 从rclpy中引入Node类，用于创建ROS 2节点
from sensor_msgs.msg import Image  # 引入用于图像传输的消息类型
from cv_bridge import CvBridge  # 引入CvBridge，用于在ROS 2消息和OpenCV图像之间进行转换
import cv2  # 引入OpenCV库


# 创建一个ROS 2 Node类CustomCam2Image
class CustomCam2Image(Node):
    def __init__(self):
        # 调用父类构造函数，节点名为'custom_cam2image'
        super().__init__('custom_cam2image')
        # 获取节点自身的logger，用于打印日志

        # 声明并获取节点参数，可以通过ROS参数服务器动态设置
        self.declare_parameters(
            namespace='',
            parameters=[
                ('width', 640),
                ('height', 480),
                ('frequency', 30.0),
                ('device_id', -1)
            ]
        )

        # 获取参数值
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.frequency = self.get_parameter('frequency').value
        self.device_id = self.get_parameter('device_id').value

        # 初始化摄像头捕获
        self.cap = cv2.VideoCapture(self.device_id)
        # 设置摄像头的图像尺寸和帧率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.frequency)

        # 初始化cv_bridge
        self.bridge = CvBridge()
        # 创建图像数据的发布者
        self.publisher_ = self.create_publisher(Image, 'image_raw', 10)

        # 创建定时器用于获取图像并发布
        timer_period = 1 / self.frequency  # 设定定时器的周期（秒）
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()  # 从摄像头读取一帧图像
        if ret:
            # 将OpenCV的图像格式转换为ROS的Image消息格式
            img_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
            # 发布ROS图像消息
            self.publisher_.publish(img_msg)
        else:
            self.get_logger().error('Failed to capture image frame')


# 当模块被直接运行时会调用main函数
def main(args=None):
    rclpy.init(args=args)  # 初始化ROS 2
    cam2image_node = CustomCam2Image()  # 创建CustomCam2Image节点对象
    rclpy.spin(cam2image_node)  # 开启节点运行，等待回调函数触发
    cam2image_node.destroy_node()  # 关闭节点
    rclpy.shutdown()  # 关闭ROS 2


if __name__ == '__main__':
    main()
"""
"""
# 导入ROS 2 Python库
import rclpy
from rclpy.node import Node

# 导入sensor_msgs下的Image消息类型，用于发布图像数据
from sensor_msgs.msg import Image

# 导入OpenCV库
import cv2
# 导入cv_bridge，这是ROS和OpenCV之间的桥梁，用于转换图像数据格式
from cv_bridge import CvBridge


# 定义CustomCam2Image类，继承自Node
class CustomCam2Image(Node):
    def __init__(self):
        # 调用父类的构造函数，节点名字为'custom_cam2image'
        super().__init__('custom_cam2image')

        # 声明并获取节点参数，包括width、height、frequency、device_id
        # 这些参数用于配置摄像头的图像尺寸、帧率和设备ID
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('frequency', 30.0)
        self.declare_parameter('device_id', -1)

        self.width = self.get_parameter('width').get_parameter_value().integer_value
        self.height = self.get_parameter('height').get_parameter_value().integer_value
        self.frequency = self.get_parameter('frequency').get_parameter_value().double_value
        self.device_id = self.get_parameter('device_id').get_parameter_value().integer_value

        # 初始化摄像头捕获
        self.cap = cv2.VideoCapture(self.device_id)
        # 设置摄像头的图像尺寸和帧率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.frequency)

        # 初始化cv_bridge，用于ROS和OpenCV图像格式之间的转换
        self.bridge = CvBridge()
        # 创建图像数据的发布者，主题名为'image_raw'
        self.publisher_ = self.create_publisher(Image, 'image_raw', 10)

        # 根据指定帧率创建定时器，定时调用timer_callback函数
        timer_period = 1 / self.frequency  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        # 从摄像头读取一帧图像
        ret, frame = self.cap.read()
        if ret:
            # 如果成功读取到图像，将OpenCV的图像格式转换为ROS的Image消息格式
            msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
            # 发布图像消息
            self.publisher_.publish(msg)
        else:
            # 如果没有成功读取图像，记录警告信息
            self.get_logger().warn('No frames')


# main函数，程序入口
def main(args=None):
    # 初始化ROS 2
    rclpy.init(args=args)
    # 创建CustomCam2Image节点实例
    node = CustomCam2Image()
    try:
        # 循环运行节点，直到收到终止信号
        rclpy.spin(node)
    except KeyboardInterrupt:
        # 收到Ctrl+C时，输出信息并优雅关闭节点
        node.get_logger().info('Keyboard interrupt, shutting down.')
    finally:
        # 释放摄像头资源，销毁节点和关闭ROS 2
        node.cap.release()
        node.destroy_node()
        rclpy.shutdown()


# 如果该文件被当作主模块执行，运行main函数
if __name__ == '__main__':
    main()
"""

# 导入ROS 2 Python库
import rclpy                            # ROS2 Python接口库
from rclpy.node import Node

# 导入sensor_msgs下的Image消息类型，用于发布图像数据
from sensor_msgs.msg import Image

# 导入OpenCV库
import cv2
# 导入cv_bridge，这是ROS和OpenCV之间的桥梁，用于转换图像数据格式
from cv_bridge import CvBridge
# 定义CustomCam2Image类，继承自Node
import numpy as np # 导入 Numpy 库，用于进行一些数学和矩阵运算
class CustomCam2Image(Node):
    def __init__(self):
        # 调用父类的构造函数，节点名字为'custom_cam2image'
        super().__init__('custom_cam2image')

        # 声明并获取节点参数，包括width、height、frequency、device_id
        # 这些参数用于配置摄像头的图像尺寸、帧率和设备ID
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 640)
        self.declare_parameter('frequency', 60.0)
        self.declare_parameter('device_id', -1)
        self.jpeg_quality = 50  # 初始JPEG质量值

        self.width = self.get_parameter('width').get_parameter_value().integer_value
        self.height = self.get_parameter('height').get_parameter_value().integer_value
        self.frequency = self.get_parameter('frequency').get_parameter_value().double_value
        self.device_id = self.get_parameter('device_id').get_parameter_value().integer_value

        # 初始化摄像头捕获
        self.cap = cv2.VideoCapture(self.device_id)

        if not self.cap.isOpened():  # 检查设备是否成功打开
            self.get_logger().error('Could not open video device')
            # 如果无法打开设备，打印错误日志
            exit(1)

        # 设置摄像头的图像尺寸和帧率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.frequency)

        # 初始化cv_bridge，用于ROS和OpenCV图像格式之间的转换
        self.bridge = CvBridge()
        # 创建图像数据的发布者，主题名为'image_raw'
        self.publisher_ = self.create_publisher(Image, 'image_raw', 10)

        # 根据指定帧率创建定时器，定时调用timer_callback函数
        timer_period = 1 / self.frequency  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        # 从摄像头读取一帧图像
        ret, frame = self.cap.read()
        if ret:
            """
            # 如果成功读取到图像，将OpenCV的图像格式转换为ROS的Image消息格式
            msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
            # 发布图像消息
            self.publisher_.publish(msg)
            """
            """

            # 压缩图像为PNG格式，无损压缩
            encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 0]
            # IMWRITE_PNG_COMPRESSION 参数范围从 0 到 9（从不压缩的最高速度到最高压缩比的最低速度，默认值为 3）
            result, encimg = cv2.imencode('.png', frame, encode_param)
            if not result:
                raise RuntimeError('Could not encode image!')

            # 使用cv_bridge构建Image消息的一部分
            msg_header = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8').header

            # 创建Image消息
            # 注意：与JPEG不同，PNG压缩后的图像可以直接作为无损压缩图像发送，无需修改图像尺寸或格式
            msg_data = np.array(encimg).tostring()
            msg = Image(header=msg_header, height=frame.shape[0], width=frame.shape[1],
                        encoding='8UC1', is_bigendian=0, step=len(msg_data), data=msg_data)

            # 发布消息
            self.publisher_.publish(msg)
            self.get_logger().info('Publishing compressed camera image')
            """
            # 压缩图像为JPEG格式
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            result, encimg = cv2.imencode('.jpg', frame, encode_param)
            if not result:
                raise RuntimeError('Could not encode image!')

            # 将压缩的图像二进制数据转换为1D Numpy数组
            msg_data = np.array(encimg).tostring()

            # 使用cv_bridge构建Image消息的一部分
            msg_header = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8').header

            # 创建Image消息
            msg = Image(header=msg_header, height=frame.shape[0], width=frame.shape[1],
                        encoding='jpeg', is_bigendian=0, step=len(msg_data), data=msg_data)

            msg.encoding = '8UC1'  # 或使用 'mono8'，这样订阅者节点就不会因为无法识别而出错
            msg.height = 1
            msg.width = len(msg_data)
            msg.step = len(msg_data)
            msg.data = np.array(encimg).tobytes()

            # 发布消息
            self.publisher_.publish(msg)
            self.get_logger().info('Publishing compressed camera image')


        else:
            # 如果没有成功读取图像，记录警告信息
            self.get_logger().warn('No frames')
        # 输出日志信息，提示已经完成图像话题发布
        self.get_logger().info('Publishing video frame')  # 输出日志信息，提示已经完成图像话题发布


# main函数，程序入口
def main(args=None):
    # 初始化ROS 2
    rclpy.init(args=args)
    # 创建CustomCam2Image节点实例
    node = CustomCam2Image()
    try:
        # 循环运行节点，直到收到终止信号
        rclpy.spin(node)
    except KeyboardInterrupt:
        # 收到Ctrl+C时，输出信息并优雅关闭节点
        node.get_logger().info('Keyboard interrupt, shutting down.')
    finally:
        # 释放摄像头资源，销毁节点和关闭ROS 2
        node.cap.release()
        node.destroy_node()
        rclpy.shutdown()


# 如果该文件被当作主模块执行，运行main函数
if __name__ == '__main__':
    main()
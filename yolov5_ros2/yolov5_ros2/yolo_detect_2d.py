from math import frexp  # 引入数学库中的frexp函数，通常用于分解浮点数
from traceback import print_tb  # 引入堆栈跟踪中的print_tb函~~~~数，用于打印异常的跟踪信息
#from torch import imag  # 这行看上去可能是个错误，因为PyTorch模块torch中没有imag函数；如果需要，应该是import torch
import torch
from yolov5 import YOLOv5  # 从yolov5库中引入YOLOv5类，用于对象检测
import rclpy  # 引入ROS 2的Python客户端库
from rclpy.node import Node  # 从rclpy中引入Node类，用于创建ROS 2节点
from ament_index_python.packages import get_package_share_directory  # 用于获取ROS 2包的共享目录路径
from rcl_interfaces.msg import ParameterDescriptor  # 引入ROS 2参数描述消息类型
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose, Detection2D  # 引入用于对象检测的视觉消息类型
from sensor_msgs.msg import Image, CameraInfo  # 引入用于图像传输和相机信息的消息类型
from cv_bridge import CvBridge  # 引入CvBridge，用于在ROS 2消息和OpenCV图像之间进行转换
import cv2  # 引入OpenCV库
import yaml  # 引入YAML库，用于解析YAML文件
from yolov5_ros2.cv_tool import px2xy  # 从包内导入px2xy工具函数，可能用于将像素坐标转换为实际坐标
import os  # 引入os模块，用于访问操作系统相关功能
import numpy as np

# Get the ROS distribution version and set the shared directory for YoloV5 configuration files.
# 获取ROS发行版版本，并设置YOLOv5配置文件的共享目录路径。
ros_distribution = os.environ.get("ROS_DISTRO")
package_share_directory = get_package_share_directory('yolov5_ros2')



"""
这部分定义了一个名为YoloV5Ros2的类，该类是从ROS 2的Node类派生的。
在类的构造函数中，使用super().__init__调用基类的构造函数，
创建了一个名为yolov5_ros2的节点，并输出了当前ROS 2发行版信息。
"""
# Create a ROS 2 Node class YoloV5Ros2.
class YoloV5Ros2(Node):
    def __init__(self):
        super().__init__('yolov5_ros2')
        # 健壮性检查：确保环境变量正确设置
        ros_distribution = os.environ.get("ROS_DISTRO")
        if not ros_distribution:
            raise RuntimeError("缺少 ROS_DISTRO 环境变量。")

        self.get_logger().info(f"Current ROS 2 distribution: {ros_distribution}")

        # Declare ROS parameters.
        self.declare_parameter("device", "cuda", ParameterDescriptor(
            name="device", description="Compute device selection, default: cpu, options: cuda:0"))
        # 这行代码声明了一个名为device的ROS参数，它用于决定YOLOv5模型执行推理时应该使用哪种计算设备，
        # 默认值设置为"cuda"。ParameterDescriptor提供了参数的名称和描述。

        self.declare_parameter("model", "yolov5s", ParameterDescriptor(
            name="model", description="Default model selection: yolov5s"))
        # 声明模型选择

        self.declare_parameter("image_topic", "/image_raw", ParameterDescriptor(
            name="image_topic", description="Image topic, default: /image_raw"))
        # 图像话题

        self.declare_parameter("camera_info_topic", "/camera/camera_info", ParameterDescriptor(
            name="camera_info_topic", description="Camera information topic, default: /camera/camera_info"))
        # 相机信息话题

        # Read parameters from the camera_info topic if available, otherwise, use the file-defined parameters.
        self.declare_parameter("camera_info_file", f"{package_share_directory}/config/camera_info.yaml",ParameterDescriptor(
            name="camera_info",description=f"Camera information file path, default: {package_share_directory}/config/camera_info.yaml"))
        # 声明一个指向相机信息配置文件的参数



        # Default to displaying detection results.
        self.declare_parameter("show_result", False, ParameterDescriptor(
            name="show_result", description="Whether to display detection results, default: False"))
        # 声明是否显示检测结果的参数

        # Default to publishing detection result images.
        self.declare_parameter("pub_result_img", False, ParameterDescriptor(
            name="pub_result_img", description="Whether to publish detection result images, default: False"))
        # 声明是否发布检测结果图像的参数
        """
        # 1. Load the model.
        model_path = package_share_directory + "/config/" + self.get_parameter('model').value + ".pt"
        device = self.get_parameter('device').value
        self.yolov5 = YOLOv5(model_path=model_path, device=device)
        """


        """
        model_path = self.get_parameter('model').value

        device = self.get_parameter('device').value
        self.yolov5 = YOLOv5(model_path=model_path, device=device)
        # 加载YOLOv5模型。模型的路径是基于配置的共享目录和参数中的模型名称拼接而成的，并且指定了计算设备
        """
        try:
            # 1. Load the model.
            model_path = self.get_parameter('model').value

            # 检查模型文件是否存在
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")

            device = self.get_parameter('device').value
            self.yolov5 = YOLOv5(model_path=model_path, device=device)
            self.get_logger().info("模型加载成功。")

        except FileNotFoundError as e:
            self.get_logger().error(str(e))
            # 发生文件不存在异常时，可选择结束程序或者处理异常
            # sys.exit(1)  # 如果要结束程序，需要导入sys模块

        except Exception as e:
            self.get_logger().error("114514:模型加载失败。")
            print_tb(e.__traceback__)
            # 当加载模型失败时，同样可以选择退出或合适的异常处理
            # sys.exit(1)  # 如果要结束程序，需要导入sys模块













        # 2. Create publishers.
        self.yolo_result_pub = self.create_publisher(
            Detection2DArray, "yolo_result", 10)
        self.result_msg = Detection2DArray()

        self.result_img_pub = self.create_publisher(Image, "result_img", 10)
        # 创建了两个发布者（publisher），分别用于发布检测结果（Detection2DArray消息类型）和结果图像（Image消息类型）。


        # 3. Create an image subscriber (subscribe to depth information for 3D cameras, load camera info for 2D cameras).
        # 3. 创建图像订阅者（对于3D摄像机订阅深度信息，对于2D摄像机加载相机信息）。
        image_topic = self.get_parameter('image_topic').value
        self.image_sub = self.create_subscription(
            Image, image_topic, self.image_callback, 10)

        camera_info_topic = self.get_parameter('camera_info_topic').value
        self.camera_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self.camera_info_callback, 1)
        # 创建了一个图像订阅者（subscriber），它订阅由参数指定的图像话题，并指定对图像消息执行的回调函数self.image_callback。
        # 同样，相机信息订阅者订阅相机信息话题，并指定对相机信息消息执行的回调函数self.camera_info_callback。


        # Get camera information.
        with open(self.get_parameter('camera_info_file').value) as f:
            self.camera_info = yaml.full_load(f.read())
            self.get_logger().info(f"default_camera_info: {self.camera_info['k']} \n {self.camera_info['d']}")
        # 从指定的YAML配置文件中读取相机信息，并打印出关键的相机参数值。


        # 4. Image format conversion (using cv_bridge).
        # 创建一个CvBridge实例，以便在后续可以将ROS图像消息转换为OpenCV图像格式。
        self.bridge = CvBridge()

        self.show_result = self.get_parameter('show_result').value
        self.pub_result_img = self.get_parameter('pub_result_img').value
        # 设置两个成员变量以存储之前声明的参数值，控制是否应在OpenCV窗口中显示检测结果，以及是否发布结果图像。

        #self.frame_paths = []  # 保存每帧图像的路径

        self.confidence_threshold = 0.6  # 设置置信度阈值为0.6


    def camera_info_callback(self, msg: CameraInfo):
        # 定义了一个名为camera_info_callback的方法，它将在节点收到CameraInfo类型消息时被调用。
        # msg参数是收到的CameraInfo消息。
        """
        Get camera parameters through a callback function.
        通过回调函数获取相机参数
        """
        self.camera_info['k'] = msg.k
        self.camera_info['p'] = msg.p
        self.camera_info['d'] = msg.d
        self.camera_info['r'] = msg.r
        self.camera_info['roi'] = msg.roi
        # 将收到的兴趣区域（Region of Interest, roi）赋值给该节点实例的camera_info字典的'roi'键。

        self.camera_info_sub.destroy()    # 在更新相机参数之后，销毁相机信息订阅者。

    # 这意味着，一旦我们从CameraInfo话题成功获取了一次数据后，就不再需要订阅这个话题，因此通过调用destroy()方法来关闭订阅者，防止未来不必要的调用。
    def image_callback(self, msg: Image): # 一个回调函数，这个函数的工作原理是当有图像消息时，就会调用


        # 5. Detect and publish results.
        #image = self.bridge.imgmsg_to_cv2(msg)

        """

        # 解码压缩图像消息 (PNG格式)
        np_arr = np.frombuffer(msg.data, dtype=np.uint8)
        # 解码PNG图像数据到OpenCV格式
        image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

        # 确保图像解码成功
        if image is None:
            self.get_logger().error('Failed to decode compressed image')
            return
        # 调整图像大小匹配YOLOv5模型的输入尺寸（例如：640x640）
        image_resized = cv2.resize(image, (640, 640))
        """


        # 解码压缩图像消息 (JPEG格式)
        # 将压缩后的图像数据转换为1D数组
        np_arr = np.frombuffer(msg.data, dtype=np.uint8)
        # 解码JPEG图像数据到OpenCV格式
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # 确保图像解码成功
        if image is None:
            self.get_logger().error('Failed to decode compressed image')
            return
        # 调整图像大小匹配YOLOv5模型的输入尺寸（例如：640x640）
        #image_resized = cv2.resize(image, (1280, 1280))


        """


        # 保存当前帧图像至本地
        save_path = "/home/ak47k98/dev_ws/src/yolov5_ros2/yolov5_ros2/frames"
        frame_path = os.path.join(save_path, f'frame_{self.result_msg.header.stamp}.jpg')  # 使用绝对路径命名图像
        cv2.imwrite(frame_path, image)  # 保存图像
        # 将当前处理的图像路径保存到列表中
        self.frame_paths.append(frame_path)
        """



        detect_result = self.yolov5.predict(image)
        # predict(image) 调用预测函数对图像进行预测
        self.get_logger().info(str(detect_result))
        # 记录检测的结果。
        self.result_msg.detections.clear()
        # 清之前的检测结果，以防止结果的累积。
        self.result_msg.header.frame_id = "camera"
        self.result_msg.header.stamp = self.get_clock().now().to_msg()
        # 给结果设定帧ID和时间戳
        # Parse the results.
        predictions = detect_result.pred[0]
        # 对预测结果进行解码
        boxes = predictions[:, :4]  # x1, y1, x2, y2
        scores = predictions[:, 4]
        categories = predictions[:, 5]
        # 从解码结果中分离出目标的位置信息、置信度得分和目标的类别等信息。

        for index in range(len(categories)):    # 对每一个识别出的目标进行处理。

            score = float(scores[index])
            if score < self.confidence_threshold:
                continue  # 如果置信度小于阈值，则跳过当前目标的处理
            else :
                name = detect_result.names[int(categories[index])]
                # 将目标的类别信息转化为具体的名称。
                detection2d = Detection2D()
                # 创建一个2D检测对象
                detection2d.id = name
                x1, y1, x2, y2 = boxes[index]
                # 从目标的位置信息中提取出目标所在的边框的位置 :
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                # 计算出目标的中心点位置 :
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0

                if ros_distribution == 'galactic':
                    detection2d.bbox.center.x = center_x
                    detection2d.bbox.center.y = center_y
                else:
                    detection2d.bbox.center.position.x = center_x
                    detection2d.bbox.center.position.y = center_y
                # 根据目标框的顶点位置信息，计算出目标框的高度和宽度  :

                detection2d.bbox.size_x = float(x2 - x1)
                detection2d.bbox.size_y = float(y2 - y1)

                obj_pose = ObjectHypothesisWithPose()  # 创建一个带有物体位置的假设对象。
                obj_pose.hypothesis.class_id = name  # 获得物体类型信息。
                obj_pose.hypothesis.score = float(scores[index])  # 获得该目标的置信度得分。

                # px2xy
                world_x, world_y = px2xy(
                    [center_x, center_y], self.camera_info["k"], self.camera_info["d"], 1)
                # 利用内参数和畸变系数将像素坐标转换为世界坐标。

                obj_pose.pose.pose.position.x = world_x
                obj_pose.pose.pose.position.y = world_y
                detection2d.results.append(obj_pose)
                self.result_msg.detections.append(detection2d)

                """
                # Draw results.
                if self.show_result or self.pub_result_img:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image,f"{name}({world_x:.2f},{world_y:.2f})", (x1, y1),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    # 在识别出的目标上写上标签名和该目标所在的世界坐标信息。
                    cv2.waitKey(1)

                """

                score = "{:.2f}".format(obj_pose.hypothesis.score)  # 将置信度得分格式化为字符串
                label = f"{name} {score} ({world_x:.2f},{world_y:.2f})"  # 创建要显示的标签字符串

                # Draw results.
                if self.show_result or self.pub_result_img:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 画出边界框
                    cv2.putText(image, label, (x1, y1 - 10),  # 注意：将文本上移一点，避免与边界框重叠
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)






        # Display results if needed.
        if self.show_result:
            cv2.imshow('result', image)
            cv2.waitKey(1)

        # Publish result images if needed.
        if self.pub_result_img:
            result_img_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            # 将图像从OpenCV图像格式转回为ROS图像消息格式
            result_img_msg.header = msg.header
            # 过图片发布器发布处理后的图像信息。
            self.result_img_pub.publish(result_img_msg)

        if len(categories) > 0:
            self.yolo_result_pub.publish(self.result_msg)
            # 如果识别出了目标，就通过YOLO结果发布器发布目标检测的结果。

def main():
    rclpy.init()
    yolo_node = YoloV5Ros2()
    try:

        rclpy.spin(yolo_node)


    except KeyboardInterrupt:
        """
        # 捕获Ctrl+C键盘中断，开始制作视频
        if yolo_node.frame_paths:
            # 读取第一张图片以获取图像大小
            first_frame = cv2.imread(yolo_node.frame_paths[0])
            height, width, layers = first_frame.shape

            # 指定视频输出路径，注意修改为所需路径，并确保输出为.mp4格式
            video_path = "/home/ak47k98/dev_ws/src/yolov5_ros2/yolov5_ros2/output_video.mp4"
            # 指定视频编解码器，这里更改为'mp4v'
            video_codec = cv2.VideoWriter_fourcc(*'mp4v')

            # 创建VideoWriter对象，注意确保文件扩展名为.mp4
            video_writer = cv2.VideoWriter(video_path, video_codec, 20.0, (width, height))

            # 遍历所有的图像路径，将它们添加到视频中
            for frame_path in yolo_node.frame_paths:
                frame = cv2.imread(frame_path)
                video_writer.write(frame)

            # 释放VideoWriter对象
            video_writer.release()
            """



    finally:
        # 正常关闭节点和清理资源
        yolo_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
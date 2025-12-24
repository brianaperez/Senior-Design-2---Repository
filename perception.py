import rclpy # Import the ROS 2 client library for Python
from rclpy.node import Node # Import the Node class for creating ROS 2 nodes
from sensor_msgs.msg import Image,PointCloud2
from geometry_msgs.msg import PoseStamped,Pose


import cv2
from cv_bridge import CvBridge

import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
import torch
from Perception.model.unet import UNet
from Perception.evaluate import evaluate

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from .rotation_utils import transform_pose

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

device = "cpu"
 
class Perception(Node):
    
 
    def __init__(self,trained_model = "/home/ubuntu/workspace/ros2_ws/src/perception_ros/Perception/epoch_21.pt"):
        super().__init__('Perception')
        self.image_subscription = self.create_subscription(
            Image,
            '/oakd/rgb/image_raw', #You can change this topic to your image topic
            self.image_callback,
            10)
        self.pose_publisher = self.create_publisher(
            PoseStamped,
            '/pose_msg', 
            10                 
        )

        # tf buffer
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.br = CvBridge()

        save_dict = torch.load(trained_model, map_location=device)
        self.model =  UNet()
        self.model.load_state_dict(save_dict["model"])
        self.model.eval()
        self.pose_msg = PoseStamped()

        self.left_fb = False
        self.right_fb = False
        


    def calculate_path(self,current_frame,pred):
        self.left_fb = False
        self.right_fb = False
        (h,w) = pred.shape
        left = pred[:,:w//2]
        right = pred[:,w//2:]


        (l_h,l_w) = left.shape
        sum_horizontal = 0
        sum_l = 0
        for i in range(50,l_h):
            for j in range(l_w):
                if(left[i,j] == 0):
                    continue
                sum_horizontal += left[i,j]*j
                sum_l += 1

        if(sum_l != 0):
            left_middle = sum_horizontal/sum_l
        else:
            left_middle = 0
            self.left_fb = True

        (r_h,r_w) = right.shape
        sum_horizontal = 0
        sum_l = 0
        for i in range(50,r_h):
            for j in range(r_w):
                if(right[i,j] == 0):
                    continue
                sum_horizontal += right[i,j]*j
                sum_l += 1

        if(sum_l != 0):
            right_middle = w//2 + sum_horizontal/sum_l
        else:
            right_middle = w-1
            self.right_fb = True

        if self.left_fb and self.right_fb:
            return

        middle = (right_middle + left_middle)/2

        watch_pixel = 160

        current_frame = cv2.resize(current_frame,(w,h),cv2.INTER_AREA)

        current_frame[watch_pixel,:,0] = 255
        current_frame[watch_pixel,:,1] = 255
        current_frame[watch_pixel,:,2] = 255

        current_frame[pred.astype(bool),1:] = 0 
        current_frame[:,int(left_middle),0] = 255
        current_frame[:,int(right_middle),1] = 255
        current_frame[:,int(middle),2] = 255

        current_frame[:,int(w/2),0] = 255
        current_frame[:,int(w/2),1] = 255
        current_frame[:,int(w/2),2] = 255

  
        cv2.imshow("camera", current_frame)   
        cv2.waitKey(1)

        self.pose_msg.header.frame_id = "base_link"
        self.pose_msg.header.stamp = self.get_clock().now().to_msg()
        self.pose_msg.pose.position.x = 5.0
        self.pose_msg.pose.position.y = (w//2 - middle )/10
        self.pose_msg.pose.position.z = 0.0
        self.pose_msg.pose.orientation.w = 1.0

        self.transform_and_publish_pose(self.pose_msg)

    def image_callback(self,msg : Image):
        rgb = self.br.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        current_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        pred = evaluate(self.model,current_frame)
        self.calculate_path(current_frame,pred)
    
    
    def transform_and_publish_pose(self,pose_msg : PoseStamped):
        try:
            t = self.tf_buffer.lookup_transform(
                "world",
                pose_msg.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0),
            )
            pose_msg.pose = transform_pose(pose_msg.pose, t)
            pose_msg.header.frame_id = "world"
            

        except TransformException as ex:
            self.get_logger().info(
                f"Could not transform {pose_msg.header.frame_id} to base_link: {ex}"
            )
            return
        
        self.pose_publisher.publish(pose_msg)

        
 
def main(args=None):
 
    rclpy.init(args=args)

    perception = Perception()
 

    rclpy.spin(perception)
 

    perception.destroy_node()
 

    rclpy.shutdown()
 
if __name__ == '__main__':

    main()
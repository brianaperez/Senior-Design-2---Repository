import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, PoseArray, PointStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
import math
import time
import numpy as np

class Controller(Node):
    def __init__(self):
        super().__init__('controller')

        # Initialize variables
        self.targetPose = None
        self.currentPose = None
        self.speed = 0.0

        #Vehicle State Variables
        self.vehicle_x = 0.0
        self.vehicle_y = 0.0
        self.vehicle_orient = 0.0
        self.vehicle_yaw = 0.0

        #Pure Pursuit Parameters
        self.L = 1.0
        self.curve_gain = 0.0 #Higher values mean slow down more on sharper turns

        #PD Control Parameters
        self.kp = 0.8
        self.kd = 0.8

        #Default State Variables
        self.error = 0.0
        self.current_error = 0.0
        self.previous_error = 0.0
        self.current_time = time.time()
        self.previous_time = self.current_time
        self.dt = 0.1

        self.P = 0.0
        self.D = 0.0

        self.kf_initialized = False
        self.last_target_x = 0.0
        self.last_target_y = 0.0

        #Subscribers from Perception /pose_msg and Car Info from Gazebo /pose_info
        self.perception_sub = self.create_subscription(PoseStamped, '/pose_msg', self.perception_callback, 10)
        self.car_sim_sub = self.create_subscription(PoseArray, '/pose_info', self.carinfo_callback, 10)
        self.odometry_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        #Publisher to Gazebo /cmd_vel
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.current_point_pub = self.create_publisher(PointStamped, '/current_point', 10)
        self.target_point_pub = self.create_publisher(PointStamped, '/target_point',10)
        self.error_publisher = self.create_publisher(PointStamped, '/error', 10)
        self.dt_publisher = self.create_publisher(Float64, '/pid_dt', 10)

        self.steer_time_pub = self.create_publisher(PointStamped, '/steering_time', 10) #Steering with time publishing
        self.timer = self.create_timer(0.1, self.control_loop) #Timer to run control loop every 0.1 seconds

    def perception_callback(self, pose_msg): #Callback extracts pose from topic
        self.targetPose = pose_msg.pose
        #self.get_logger().info('Received target pose')
    
    def carinfo_callback(self, msg): #Callback extracts pose from topic
        pose = msg.poses[1]
        self.currentPose = msg.poses[1]

        self.vehicle_x = pose.position.x
        self.vehicle_y = pose.position.y
        self.vehicle_z = pose.position.z
        self.vehicle_orient = pose.orientation
        self.vehicle_yaw = self.euler_from_quaternion(self.vehicle_orient.x, self.vehicle_orient.y, self.vehicle_orient.z, self.vehicle_orient.w)

    def odom_callback(self, odom_msg):
        self.speed = odom_msg.twist.twist.linear.x

    def euler_from_quaternion(self, x, y, z, w):
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw
    
    def car_offset_to_world(self, x_off, y_off, xcar, ycar, yaw):
        xw = xcar + x_off * math.cos(yaw) - y_off * math.sin(yaw)
        yw = ycar + x_off * math.cos(yaw) - y_off * math.cos(yaw)
        return xw, yw
    
    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))
    
    def get_alpha(self, x, y, yaw, lap_x, lap_y):
        dx = lap_x - x
        dy = lap_y - y
        target_angle = math.atan2(dy, dx)
        
        return self.normalize_angle(target_angle - yaw)
    
    def compute_curvature(self, alpha, ld):
        return 2.0 * math.sin(alpha) / ld if ld >1e-6 else 0.0
    
    def steering_angle_to_angular_velocity(self, steering_angle):
        v = self.speed
        L = self.L
        w = (v/L) * math.tan(steering_angle)
        return w
    
    def clamp(self, x, low, high):
        return max(low, min(x, high))
    
    def pure_pursuit(self, x, y, yaw, target_x, target_y, ld):
        alpha = self.get_alpha(x,y,yaw,target_x,target_y)
        curvature = self.compute_curvature(alpha, ld)
        steering_angle = math.atan(curvature * self.L)
        angular_velocity = self.steering_angle_to_angular_velocity(steering_angle)

        '''Publishing Steering Angular Velocity with Timestamp for Analysis'''
        steer_time = PointStamped()
        steer_time.point.x = angular_velocity
        sec, nanosec = self.get_clock().now().seconds_nanoseconds()
        time = sec + nanosec / 1e9
        steer_time.point.y = time
        self.steer_time_pub.publish(steer_time)

        return angular_velocity, curvature
    
    def scale_velocity(self, v_cmd, curvature):
        # Tune these
        v_max = 6.25          # max forward speed

        # Scale down as curvature magnitude increases
        scale = 1.0 / (1.0 + self.curve_gain * abs(curvature))
        v = v_cmd * scale

        # Clamp
        v = max(0.0, min(v, v_max))
        return v
    
    def pd_control(self, x, y, path_yaw, target_x, target_y):
        #Calculation of Error
        dx = target_x - x
        dy = target_y - y
        localx = dx * math.cos(path_yaw) + dy * math.sin(path_yaw)
        #localy = math.sin(-yaw) * dx + math.cos(-yaw) * dy

        self.error = localx

        '''Publishing Error for Graphing'''
        error_msg = PointStamped()
        error_msg.point.x = self.error
        sec, nanosec = self.get_clock().now().seconds_nanoseconds()
        time_e = sec + nanosec / 1e9
        error_msg.point.y = time_e
        self.error_publisher.publish(error_msg)
        
        #We have established error now we must update previous and current errors
        self.previous_error = self.current_error
        self.current_error = self.error

        #Now update timing for PD Calcualtion
        self.previous_time = self.current_time
        self.current_time = time.time()
        self.dt = self.current_time - self.previous_time
        self.dt = max(self.dt, 0.01)

        #Calculation of PD Terms
        self.P = self.kp * self.current_error
        self.D = self.kd * ((self.current_error - self.previous_error) / self.dt)

        output = self.P + self.D

        dt_msg = Float64()
        sec, nanosec = self.get_clock().now().seconds_nanoseconds()
        time_dt = sec + nanosec / 1e9
        dt_msg.data = time_dt
        self.dt_publisher.publish(dt_msg)

        return output

    def control_loop(self):
        #We are going to utilize previous controllers logic like PP and PD Control
        if self.targetPose is None or self.currentPose is None: #Check if we have BOTH a target and current car pose, if not do not run
            self.get_logger().info('Target or current pose is not available')
            return
        
        #Beyond this point we have a target pose and current pose, so we can proceed with the control logic
        #Extract Vehicle Pose Components and Orientation
        x = self.vehicle_x
        y = self.vehicle_y
        yaw = self.vehicle_yaw
        v_min = 0.0
        v_max = 6.25

        #Apply same concept to our target pose
        target_x = self.targetPose.position.x
        target_y = self.targetPose.position.y

        '''Publishing Car Point and Target Point'''
        carpoint = PointStamped()
        carpoint.header.frame_id = "world"
        carpoint.header.stamp = self.get_clock().now().to_msg()
        carpoint.point.x = x
        carpoint.point.y = y
        self.current_point_pub.publish(carpoint)

        targetpoint = PointStamped()
        targetpoint.header.frame_id = "world"
        targetpoint.header.stamp = self.get_clock().now().to_msg()
        targetpoint.point.x = target_x
        targetpoint.point.y = target_y
        self.target_point_pub.publish(targetpoint)

        dx = target_x - x
        dy = target_y - y
        ld = math.hypot(dx,dy)
        ld = max(ld,1e-6)

        path_yaw = matha.atan2(target_x - x, target_y - y)
        angle_v, curvature = self.pure_pursuit(x, y, yaw, target_x, target_y, ld)
        pd_vel = self.pd_control(x, y, path_yaw, target_x, target_y)

        pd_vel = self.clamp(pd_vel, v_min, v_max)
        v_scaled = self.scale_velocity(pd_vel, curvature)

        twist = Twist()
        twist.linear.x = v_scaled
        twist.angular.z = angle_v
        self.cmd_vel_pub.publish(twist)
        self.get_logger().info(f"Velocity: {self.speed:.2f} Ang Velocity: {angle_v:.2f}")

def main(args=None):

    rclpy.init(args=args)
    controller = Controller()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



        
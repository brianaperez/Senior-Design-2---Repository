import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PointStamped, PoseArray
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float64
import math
import time

class PurePursuitController(Node):
    def __init__(self):
        super().__init__('pure_pursuit_controller')

        # Parameters
        self.declare_parameter('wheelbase', 1.8)
        self.declare_parameter('lookahead_min', 11.0)
        self.declare_parameter('lookahead_gain', 3.0)

        self.L = self.get_parameter('wheelbase').value
        self.kv = self.get_parameter('lookahead_gain').value
        self.ld_min = self.get_parameter('lookahead_min').value

        # Default States
        self.lookahead_point = None #no lookahead found first
        self.target_x = None
        self.target_y = None

        #Vehicle State Variables
        self.vehicle_x = 0.0
        self.vehicle_y = 0.0
        self.vehicle_orient = 0.0
        self.vehicle_yaw = 0.0
        self.vehicle_v = 0.0
        
        #Subscribers
        self.create_subscription(PointStamped, '/waypoints', self.lookahead_callback, 10) #Subscription to Waypoints
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10) #Subscription to Odometry for Velocity of Vehicle
        self.create_subscription(PoseArray, '/pose_info', self.pose_callback, 10) #Subscription to Pose Info for Vehicle Position

        #Publishers
        self.curvature_pub = self.create_publisher(Float64, '/curvature', 10) #For Curvature Publishing
        self.steer_time_pub = self.create_publisher(PointStamped, '/steering_time', 10) #Steering with time publishing
        self.cmd_vel_pub = self.create_publisher(Twist, '/pp_cmd_vel', 10) #Pure Pursuit Command Velocity Publishing

        #Timer for control loop
        self.create_timer(0.05, self.purepursuit_callback) #20Hz

    #Callbacks
    def lookahead_callback(self, msg):
        self.target_x = msg.point.x
        self.target_y = msg.point.y

    def odom_callback(self, odom_msg: Odometry):
        #Get Vehicle Velocity
        self.vehicle_v = odom_msg.twist.twist.linear.x

    def pose_callback(self, msg):
        pose = msg.poses[1] #Car position stored here

        #Storing Car X, Y and Z as well as the Yaw
        self.vehicle_x = pose.position.x
        self.vehicle_y = pose.position.y
        self.vehicle_z = pose.position.z
        self.vehicle_orient = pose.orientation
        self.vehicle_yaw = self.euler_from_quaternion(self.vehicle_orient.x, self.vehicle_orient.y, self.vehicle_orient.z, self.vehicle_orient.w)

    def euler_from_quaternion(self, x, y, z, w):
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return yaw

    #Functions Needed for Calculations
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
        v = self.vehicle_v
        L = self.L
        w = (v/L) * math.tan(steering_angle)
        return w
    
    #Pure Pursuit Control Loop Logic is contained below
    def purepursuit_callback(self):
        if self.target_x is None and self.target_y is None:
            return
    
        #Lookahead Point is recovered here at this point
        #Establish the angle alpha from my car to target point and its yaw
        alpha = self.get_alpha(self.vehicle_x, self.vehicle_y, self.vehicle_yaw, self.target_x, self.target_y)

        #Recopy lookahead distance formula from waypoint to be used here for curvature calculation
        ld = max(self.ld_min, self.kv * self.vehicle_v)
        #Testing Pure Pursuit Alone set ld not have minimum
        #ld = self.kv * self.vehicle_v

        #With alpha calculated and normalized, we can compute curvature using our alpha value and lookahead distance
        curvature = self.compute_curvature(alpha, ld)

        #Publish our curvature value to be shared with PD Controller for Scaling Velocity
        self.curvature_pub.publish(Float64(data=curvature)) #Publishing Curvature

        #From curvature, we can compute steering angle
        steering_angle = math.atan(curvature * self.L)

        #Gazebo expects a Twist message for steering control as angular velocity so convert steering angle to angular velocity
        angular_velocity = self.steering_angle_to_angular_velocity(steering_angle)

        '''Publishing Steering Angular Velocity with Timestamp for Analysis'''
        steer_time = PointStamped()
        steer_time.point.x = angular_velocity
        sec, nanosec = self.get_clock().now().seconds_nanoseconds()
        time = sec + nanosec / 1e9
        steer_time.point.y = time
        self.steer_time_pub.publish(steer_time)

        #Now with angular velocity, we can publish to pp__cmd_vel topic or known as pure pursuit command velocity topic
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = angular_velocity
        self.cmd_vel_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    controller = PurePursuitController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
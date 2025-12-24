import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PointStamped, PoseArray
from nav_msgs.msg import Path
from std_msgs.msg import Float64
import math
import time

class PIDController(Node):
    def __init__(self):
        super().__init__('pid_controller')

        # Default State Variables
        self.error = 0
        self.current_error = 0
        self.track_x = 0.0
        self.track_y = 0.0
        self.closest_point_x = 0.0
        self.closest_point_y = 0.0
        self.vehicle_x = 0.0
        self.vehicle_y = 0.0
        self.next_closest_x = 0.0
        self.next_closest_y = 0.0

        #Isolated PID Controller Parameters
        self.use_fixed_target = False  # Set to True to use fixed target
        self.fixed_target_x = 380.00    # Fixed target X coordinate
        self.fixed_target_y = -205.2557208     # Fixed target Y coordinate so we travel in straight line

        # PID parameters - tune as needed
        self.Kp = 0.4  # Proportional gain  
        self.Kd = 0.4   # Derivative gain
        
        # PID state variables
        self.current_time = time.time()
        self.previous_time = self.current_time
        self.current_error = 0
        self.previous_error = 0
        self.P = 0
        self.D = 0
        
        # Sampling time for discrete implementation
        self.dt = 0.1  # Expected sampling time (100ms)

        # Subscriptions
        self.track_point_sub = self.create_subscription(PointStamped, '/waypoints', self.tracking_point_callback, 10)
        self.subscription = self.create_subscription(PoseArray, '/pose_info', self.pose_callback, 10)
        # Publishers
        self.cmd_vel_publisher = self.create_publisher(Twist, '/pid_cmd_vel', 10)
        self.dt_publisher = self.create_publisher(Float64, '/pid_dt', 10)
        self.error_publisher = self.create_publisher(PointStamped, '/error', 10)
        
        #Timer for Control Loop
        self.timer = self.create_timer(0.05, self.generate_control_output)  # 10 Hz
        
    def tracking_point_callback(self, msg):
        self.track_x = msg.point.x
        self.track_y = msg.point.y
    
    def pose_callback(self, msg):
        pose = msg.poses[1] #Car position stored here
        self.vehicle_x = pose.position.x
        self.vehicle_y = pose.position.y
    
    def calc_positional_error(self, lookahead_point, vehicle_pos, path_yaw):
        dx = lookahead_point[0] - vehicle_pos[0]
        dy = lookahead_point[1] - vehicle_pos[1]
        error = dx * math.cos(path_yaw) + dy * math.sin(path_yaw)
        return error
    
    def calc_x_error(self, lookahead_point, vehicle_pos):
        dx = lookahead_point[0] - vehicle_pos[0]
        return dx
    
    def generate_control_output(self):
        if self.track_x == 0.0 and self.track_y == 0.0: 
            return  # Wait until we have valid target point

        current_x = self.vehicle_x #should be updated from pose_callback
        current_y = self.vehicle_y #should be updated from pose_callback
        vehicle_pos = (current_x, current_y) #Ties Vehicle Position represented as coordinate point

        #If we want to test PID isolated, set lookahead point to be fixed target, else set lookaheadpoint made as a tuple from callback
        if self.use_fixed_target:
            lookahead_point = (self.fixed_target_x, self.fixed_target_y)
        else:
            lookahead_point = (self.track_x, self.track_y) #Ties Lookahead Point Position represented as coordinate point

        path_yaw = math.atan2(self.track_y - self.vehicle_y, self.track_x - self.vehicle_x)
        
        # Calculate error based on mode if testing for Isolated PID or integrated
        if self.use_fixed_target:
            self.error = self.calc_x_error(lookahead_point, vehicle_pos)
        else:
            self.error = self.calc_positional_error(lookahead_point, vehicle_pos, path_yaw)

        # Update error terms
        #Initially Previous error is 0 and Current Error takes on the value of the calculated error
        #Later Iterations it will shift on its own
        self.previous_error = self.current_error
        self.current_error = self.error

        # Publish error with timestamp for analysis of Error vs Time during Simulation
        error_msg = PointStamped()
        error_msg.point.x = self.error
        sec, nanosec = self.get_clock().now().seconds_nanoseconds()
        time_e = sec + nanosec / 1e9
        error_msg.point.y = time_e
        self.error_publisher.publish(error_msg)
        
        # Update timing for PID calculation
        self.previous_time = self.current_time 
        self.current_time = time.time()
        self.dt = self.current_time - self.previous_time
        self.dt = max(self.dt, 0.01)
        
        # Calculate PID terms
        # Proportional term: Kp * e(t)
        self.P = self.Kp * self.current_error
        
        # Derivative term: Kd * de(t)/dt
        self.D = self.Kd * (self.current_error - self.previous_error) / self.dt
        
        # Calculate target output for PID Controller
        control_output = self.P + self.D
            
        # Convert control output to speed (always move forward toward waypoint)
        speed = control_output

        '''Publishing time for Error Graph'''
        dt_msg = Float64()
        sec, nanosec = self.get_clock().now().seconds_nanoseconds()
        time_dt = sec + nanosec / 1e9
        dt_msg.data = time_dt
        self.dt_publisher.publish(dt_msg)

        # Publish Linear X value as Speed to /pid_cmd_vel topic
        twist = Twist()
        twist.linear.x = speed
        twist.angular.z = 0.0  # No angular velocity for simple point-to-point control
        self.cmd_vel_publisher.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    controller = PIDController()
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
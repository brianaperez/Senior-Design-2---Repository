import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64

class CmdVelCombiner(Node):
    def __init__(self):
        super().__init__('cmd_vel_combiner')

        #Default State Variables
        self.gain = 0.0 #K gain for velocity scaling based on curvature 15.0 for Pure Pursuit and 8.0 for Stanley
        self.curvature = 0.0
        self.pure_pursuit_cmd = None
        self.pid_cmd = None
        self.stanley_cmd = None

        #Create Subscribers to the Pure Pursuit Steering Commands
        #They will have linear movement set to 0.0 and only contain steering angles
        self.pp_sub = self.create_subscription(Twist, '/pp_cmd_vel', self.pp_callback, 10)

        #Create Subscribers to the PID Controller Commands
        #They will set the speed of the car and set its angles to 0.0
        self.pid_sub = self.create_subscription(Twist, '/pid_cmd_vel', self.pid_callback, 10)

        self.stanley_sub = self.create_subscription(Twist, '/stanley_cmd_vel', self.stanley_callback, 10)

        self.curvature_sub = self.create_subscription(Float64, '/curvature', self.curvature_callback, 10)

        #Create publisher for sending commands to the car itself
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        #Timer for Control Loop
        self.timer = self.create_timer(0.05, self.timer_callback)  # 20 Hz

    #Callback function to set the msg from pure_pursuit_cmd_vel to the class variables set in the beginning
    def pp_callback(self, msg):
        self.pure_pursuit_cmd = msg

    #Callback function to set the msg from pid_cmd_vel to the class variable set in the beginning
    def pid_callback(self, msg):
        self.pid_cmd = msg

    def stanley_callback(self, msg):
        self.stanley_cmd = msg

    def curvature_callback(self, msg):
        self.curvature = msg.data

    def scale_velocity(self):
        v_max = 6.25  # maximum velocity
        scale_magnitude = 1 / (1+ self.gain * abs(self.curvature))
        v = self.pid_cmd.linear.x * scale_magnitude
        v = max(0.0, min(v, v_max))  # Clamp velocity to [0, v_max]

        return v
        

    def timer_callback(self):
        #No commands will be set unless we have both steering and speed commands together
        #if self.pure_pursuit_cmd is None or self.pid_cmd is None:
        if self.pid_cmd is None or self.stanley_cmd is None:
            return  # wait for both commands

        #Create a new twist message
        combined = Twist()
        combined.linear.x = self.scale_velocity()  # longitudinal from PID being scaled first 
        #combined.angular.z = self.pure_pursuit_cmd.angular.z  # steering from Pure Pursuit
        combined.angular.z = self.stanley_cmd.angular.z  # steering from Stanley

        #Fixed or Constant Velocity for Testing
        #combined.linear.x = 3.0 #Constant Velocity

        #Fixed Steering for Testing
        #combined.angular.z = 0.0 #locked steering for testing
        self.get_logger().info(f"V:{combined.linear.x:.2f}, Steering: {combined.angular.z:.2f}")

        self.cmd_vel_pub.publish(combined)


def main(args=None):
    rclpy.init(args=args)
    combiner = CmdVelCombiner()
    rclpy.spin(combiner)
    combiner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
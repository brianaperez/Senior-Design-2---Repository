import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PointStamped, PoseStamped, PoseArray
from std_msgs.msg import Float64
import csv
import os
import math
from ament_index_python.packages import get_package_share_directory

class WaypointPublisher(Node):
    def __init__(self):
        super().__init__("waypoint_publisher")

        packageName = 'pure_pursuit_pid'
        csvFile = 'trajectory.csv'
        packagePath = get_package_share_directory(packageName)
        csvPath = os.path.join(packagePath, csvFile)
        self.waypoints = self.readCSV(csvPath)

        self.L = 1.8
        self.ld_min = 11.0
        self.kv = 3.0

        self.vehicle_x = 0.0
        self.vehicle_y = 0.0
        self.vehicle_z = 0.0
        self.vehicle_orient = 0.0
        self.vehicle_yaw = 0.0
        self.vehicle_v = 0.0

        self.wp_idx = 0
        self.last_waypoint_select = False
        self.lookahead_point = None

        #Waypoint Publisher publishes a target point for controller
        #Stanley needs to calculate path segment so we give it a path represented by all waypoints
        self.point_publish = self.create_publisher(PointStamped, '/waypoints', 10)
        self.path_publish = self.create_publisher(Path, '/path', 10)
        self.car_point = self.create_publisher(PointStamped, '/car_point',10) #Car Point Publishing

        self.create_subscription(Odometry, '/odom', self.odom_callback, 10) #Subscription to Odometry for Velocity of Vehicle
        self.create_subscription(PoseArray, '/pose_info', self.pose_callback, 10) #Subscription to Pose Info for Vehicle Position

        self.timer = self.create_timer(0.02, self.control_loop)
        self.path_timer = self.create_timer(1.0, self.publish_path)

    def euler_from_quaternion(self, x, y, z, w):
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return yaw

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

    def readCSV(self, csvPath):
        waypoints = []
        try:
            with open(csvPath, 'r') as file:
                read = csv.reader(file)
                for row in read:
                    if len(row) < 2:
                        continue
                    x = float(row[0])
                    y = float(row[1])
                    z = float(row[2]) if len(row) > 2 else 0.0 #ignore Z but keep it in tuple
                    waypoints.append((x, y))
        except FileNotFoundError:
            self.get_logger().error(f"File {csvPath} was not found.")
        except Exception as e:
            self.get_logger().error(f"Error reading waypoints: {e}")

        #self.get_logger().info(f"Loaded {len(waypoints)} waypoints.")
        return waypoints
    
    def find_closest_waypoint_idx(self, x, y, yaw):
        close_dist = float('inf')
        close_idx = 0
        for i in range(self.wp_idx, len(self.waypoints)):
            wp_x, wp_y = self.waypoints[i]
            dist = math.hypot(wp_x - x, wp_y - y)
            dx = wp_x - x
            dy = wp_y - y
            dot = dx * math.cos(yaw) + dy * math.sin(yaw)
            if dot < 0.0:
                continue  # waypoint is behind the vehicle

            if dist < close_dist:
                close_dist = dist
                close_idx = i
        return close_idx

    def find_lookaheadpoint(self, x, y, ld, future_wp):
        if self.last_waypoint_select:
            return future_wp[-1] #Always return last waypoint if already selected

        for wp in future_wp[1:]:
            dx = wp[0] - x
            dy = wp[1]- y
            distance = math.hypot(dx, dy)
            if distance >= ld:
                return wp
            
        self.last_waypoint_select = True    
        return future_wp[-1]  # Return last waypoint if none found

    def publish_path(self):
        path_msg = Path()
        path_msg.header.frame_id = "world"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for waypoint in self.waypoints:
            pose = PoseStamped()
            pose.header.frame_id = "world"
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = waypoint[0]
            pose.pose.position.y = waypoint[1]
            pose.pose.position.z = 0.0
            path_msg.poses.append(pose)

        self.path_publish.publish(path_msg)


    def control_loop(self):
        #Get Vehicle Position and Yaw
        x = self.vehicle_x
        y = self.vehicle_y
        yaw = self.vehicle_yaw
        
        #Vehicle Speed
        v = self.vehicle_v

        #Lookahead distance
        ld = max(self.ld_min, self.kv * v)

        #Adjust for testing to isolate ld for Pure Pursuit
        #ld = self.kv * v

        #Waypoints Stored in self.waypoints so lets find the closest waypoint index from waypoints
        self.wp_idx = self.find_closest_waypoint_idx(x, y, yaw)

        #Closest waypoint index is found: create a subpath list of waypoints from closest index and onward
        future_wp = self.waypoints[self.wp_idx:]

        #Now we have sub waypoints that are in front of the vehicle
        #Now search for a point in sub waypoints that is beyond our lookahead distance
        self.lookahead_point = self.find_lookaheadpoint(x, y, ld, future_wp)

        if self.lookahead_point is None:
            self.get_logger().info("No lookahead point found")
            return
        #Beyond this point we have found a valid lookahead point so extract its x and y coordinates
        lap_x, lap_y = self.lookahead_point

        '''Publishing Lookahead Point and Car Current Point for RViz2 Visualization'''
        point_msg = PointStamped()
        point_msg.header.frame_id = "world"
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.point.x = lap_x
        point_msg.point.y = lap_y
        point_msg.point.z = 0.0

        p_msg = PointStamped()
        p_msg.header.frame_id = "world"
        p_msg.header.stamp = self.get_clock().now().to_msg()
        p_msg.point.x = x
        p_msg.point.y = y
        p_msg.point.z = 0.0
        self.car_point.publish(p_msg)

        #Publish lookahead point to topic /waypoint
        self.point_publish.publish(point_msg)
        

def main():
    rclpy.init()
    node = WaypointPublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
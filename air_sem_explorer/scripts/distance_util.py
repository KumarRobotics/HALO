import rclpy
from rclpy.node import Node
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA


def quat_msg_to_rot_mat(orientation) -> np.ndarray:
    # Extract quaternion [x, y, z, w]
    quat = [
        orientation.x,
        orientation.y,
        orientation.z,
        orientation.w
    ]
    
    # Convert to 3x3 rotation matrix
    return R.from_quat(quat).as_matrix()

class DistanceUtil(Node):
    def __init__(self):
        super().__init__('distance_util_node')
        self.declare_parameter('odom_topic', '/pose_map')

        self.odom_world_ = None
        self.odom_pos_ = None

        self.total_distance_ = 0.0
        self.last_odom_pos_ = None

        ############ Callback groups ############
        odom_group = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()
    
        self.odom_topic = self.get_parameter('odom_topic').value

        self.odom_sub_ = self.create_subscription(
            PoseStamped, self.odom_topic, self.odom_callback,
            10, callback_group=odom_group
        )
        self.marker_pub_ = self.create_publisher(
            Marker,
            '/trajectory_marker',
            10
        )
        # Initialize trajectory marker
        self.traj_marker_ = Marker()
        self.traj_marker_.header.frame_id = "map"
        self.traj_marker_.ns = "odometry_trajectory"
        self.traj_marker_.id = 0
        self.traj_marker_.type = Marker.LINE_STRIP
        self.traj_marker_.action = Marker.ADD
        
        # Set line appearance
        self.traj_marker_.scale.x = 0.4  # Line width in meters
        self.traj_marker_.color.r = 1.0
        self.traj_marker_.color.g = 0.0
        self.traj_marker_.color.b = 1.0
        self.traj_marker_.color.a = 1.0
        
        # Set pose (identity)
        self.traj_marker_.pose.orientation.w = 1.0
        # Distance threshold for adding points (avoid cluttering)
        self.min_distance_threshold_ = 0.05  # minimum distance between points

        # Timer for visualization (1 Hz)
        self.viz_timer_ = self.create_timer(1.0, self.traj_visualize)

        self.get_logger().info("Distance util node initialized")

    def odom_callback(self, msg):
        """
        Callback for odometry updates.
        """
        tmp_odom = msg.pose
        # Update the odometry in world and map frames
        body2world = np.eye(4, dtype=np.float32)
        body2world[:3, :3] = quat_msg_to_rot_mat(tmp_odom.orientation)
        body2world[:3, 3] = np.array([tmp_odom.position.x, 
                                      tmp_odom.position.y, 
                                      tmp_odom.position.z])

        self.odom_world_ = body2world
        self.last_odom_pos_ = copy.deepcopy(self.odom_pos_)
        self.odom_pos_ = np.array([self.odom_world_[0, 3], self.odom_world_[1, 3], self.odom_world_[2, 3]], dtype=np.float32)

        # compute accumulated distance
        if self.last_odom_pos_ is not None:
            distance_moved = np.linalg.norm(self.odom_pos_[:2] - self.last_odom_pos_[:2])
            self.total_distance_ += distance_moved
            
            # Only add point if moved enough (reduce marker density)
            if distance_moved > self.min_distance_threshold_:
                # Add current position to trajectory marker
                p = Point()
                p.x = float(self.odom_pos_[0])
                p.y = float(self.odom_pos_[1])
                p.z = float(self.odom_pos_[2] - 25)
                self.traj_marker_.points.append(p)
                
                color = ColorRGBA()
                color.r = 1.0
                color.g = 0.0
                color.b = 0.0
                color.a = 1.0
                self.traj_marker_.colors.append(color)
                
            self.get_logger().info(
                f"Total distance: {self.total_distance_:.2f}m, "
                f"Points: {len(self.traj_marker_.points)}"
            )

        else:
            # First point - initialize trajectory
            p = Point()
            p.x = float(self.odom_pos_[0])
            p.y = float(self.odom_pos_[1])
            p.z = float(self.odom_pos_[2] - 25)
            self.traj_marker_.points.append(p)
            
            # First point color (green)
            color = ColorRGBA()
            color.r = 1.0
            color.g = 0.0
            color.b = 0.0
            color.a = 1.0
            self.traj_marker_.colors.append(color)
            
            self.get_logger().info("Trajectory visualization started")

    def traj_visualize(self):
        """
        Visualize the trajectory by publishing the marker.
        """
        if len(self.traj_marker_.points) > 0:
            # Update timestamp
            self.traj_marker_.header.stamp = self.get_clock().now().to_msg()
            
            # Publish the marker
            self.marker_pub_.publish(self.traj_marker_)
        

if __name__ == '__main__':
    rclpy.init()
    distance_util_node = DistanceUtil()
    rclpy.spin(distance_util_node)
    distance_util_node.destroy_node()
    rclpy.shutdown()
import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Pose, Twist, PoseStamped
from nav_msgs.msg import Odometry
from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation as R
import tf2_geometry_msgs  # Needed for do_transform_pose

class SimTracker(Node):
    def __init__(self):
        super().__init__('sim_tracker')
        
        # Two use cases:
        # 1. transform_wp = True. Waypoints are in local map frame, we transform it to global world frame, and track using odom in world frame
        # 2. transform_wp = False. Waypoints are already in world frame, we track using odom in world frame
        # Parameters
        self.declare_parameter('max_velocity', 5.0)  # m/s
        self.declare_parameter('waypoint_tolerance', 0.5)  # meters
        self.declare_parameter('control_frequency', 10.0)  # Hz
        self.declare_parameter('use_odom_msg', False)
        self.declare_parameter('transform_wp', True)
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('map_frame', 'map')

        self.current_pos = np.zeros(3)
        self.current_orientation = None
        self.current_waypoint = None  # Only tracks [x, y]
        
        # Velocity model
        self.max_vel = self.get_parameter('max_velocity').value
        self.waypoint_tol = self.get_parameter('waypoint_tolerance').value
        self.use_odom_msg = self.get_parameter('use_odom_msg').value
        self.world_frame = self.get_parameter('world_frame').value
        self.map_frame = self.get_parameter('map_frame').value
        self.transform_wp = self.get_parameter('transform_wp').value

        # Print debug info
        self.get_logger().info(f"Waypoint tolerance: {self.waypoint_tol} m")
        self.get_logger().info(f"Using odom message: {self.use_odom_msg}")
        self.get_logger().info(f"Transforming waypoints from map to world frame: {self.transform_wp}")
        self.get_logger().info(f"World frame: {self.world_frame}, Map frame: {self.map_frame}")

        # TF listener for map→world transform
        self.tf_buffer = Buffer()     
        self.tf_listener = TransformListener(self.tf_buffer, self)        
        self.map_to_world_ = None

        # ROS interfaces
        # self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.control_odom_pub = self.create_publisher(Pose, '/quadrotor/pose_cmd', 10)
        
        self.waypoint_sub = self.create_subscription(
            PoseStamped,
            '/planner/waypoint',
            self.waypoint_callback,
            10)
            
        if self.use_odom_msg:
            self.get_logger().info("Using odometry messages for pose updates")
            self.odom_sub = self.create_subscription(
                Odometry,
                '/odom',
                self.odom_callback,
                10)
        else:
            self.odom_sub = self.create_subscription(
                PoseStamped,
                '/odom',
                self.odom_callback,
                10)
        
        # Control timer
        self.dt = 1.0 / self.get_parameter('control_frequency').value
        self.control_timer = self.create_timer(self.dt, self.control_loop)
        
        self.get_logger().info("Orientation-aware Holonomic SimTracker initialized")

    def lookup_map_to_world(self):
        """Lookup TF transform (world→map) and store its inverse (map→world)."""
        try:
            # Get transform: target = map, source = world
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,   # target frame
                self.world_frame, # source frame
                rclpy.time.Time()  # latest
            )

            # Extract translation and rotation
            t = transform.transform.translation
            q = transform.transform.rotation

            # Homogeneous matrix world→map
            T_world_map = np.eye(4)
            T_world_map[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            T_world_map[:3, 3] = np.array([t.x, t.y, t.z])

            # Invert to get map→world
            self.map_to_world_ = np.linalg.inv(T_world_map)

            self.get_logger().info("map_to_world transform set.")

        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")


    def transform_pose_to_world(self, header, pose_msg):
        """Transform pose (in map frame) to world frame using stored transform"""
        if self.map_to_world_ is None:
            self.get_logger().warn("map_to_world transform not set yet.")
            return None

        # Pose in map frame → homogeneous transform
        pos = np.array([pose_msg.position.x,
                        pose_msg.position.y,
                        pose_msg.position.z])
        ori = pose_msg.orientation
        rot = R.from_quat([ori.x, ori.y, ori.z, ori.w]).as_matrix()
        T_map = np.eye(4)
        T_map[:3, :3] = rot
        T_map[:3, 3] = pos
        T_world = self.map_to_world_ @ T_map

        # Convert back to PoseStamped
        transformed_pose = PoseStamped()
        transformed_pose.header.stamp = header.stamp
        transformed_pose.header.frame_id = self.world_frame

        transformed_pose.pose.position.x = T_world[0, 3]
        transformed_pose.pose.position.y = T_world[1, 3]
        transformed_pose.pose.position.z = T_world[2, 3]

        quat = R.from_matrix(T_world[:3, :3]).as_quat()
        transformed_pose.pose.orientation.x = quat[0]
        transformed_pose.pose.orientation.y = quat[1]
        transformed_pose.pose.orientation.z = quat[2]
        transformed_pose.pose.orientation.w = quat[3]

        return transformed_pose


    def waypoint_callback(self, msg):
        """Store new waypoint position (ignores orientation)"""
        # Check if TF is there
        if self.transform_wp:
            if self.map_to_world_ is None:
                self.lookup_map_to_world()
                if self.map_to_world_ is None:
                    self.get_logger().warn("No map_to_world TF yet, cannot transform waypoint.")
                    return
            # Transform waypoint to world frame
            transformed_waypoint = self.transform_pose_to_world(msg.header, msg.pose)
            self.current_waypoint = np.array([
                transformed_waypoint.pose.position.x,
                transformed_waypoint.pose.position.y,
            ])
            self.get_logger().info(f"New waypoint received -- world coord: {self.current_waypoint}, map coord: {[msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]}")

        else:
            self.current_waypoint = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
            ])
            self.get_logger().info(f"New waypoint received -- world coord: {self.current_waypoint}")

    def odom_callback(self, msg):
        """Update full pose from odometry (position + orientation)"""
        if self.use_odom_msg:
            tmp_odom = msg.pose.pose
        else:
            tmp_odom = msg.pose
        # Position
        self.current_pos[0] = tmp_odom.position.x
        self.current_pos[1] = tmp_odom.position.y
        self.current_pos[2] = tmp_odom.position.z
        
        self.current_orientation = tmp_odom.orientation
        # Orientation (quaternion to yaw)
        # q = msg.pose.pose.orientation
        # _, _, self.current_pose[2] = euler_from_quaternion([q.x, q.y, q.z, q.w])

    def control_loop(self):
        """Holonomic control with orientation pass-through"""
        if self.current_waypoint is None:
            return
    
        # Calculate vector to waypoint
        direction = self.current_waypoint - self.current_pos[:2]
        distance = np.linalg.norm(direction)
        
        # Check if waypoint reached
        if distance < self.waypoint_tol:
            cmd_vel = Twist()
            self.current_waypoint = None
            self.get_logger().info("Waypoint reached!")
        else:
            # Normalize direction and scale by max velocity
            cmd_vel = Twist()
            if distance > 0:
                norm_direction = direction / distance
                velocity = norm_direction * self.max_vel
                
                cmd_vel.linear.x = velocity[0]
                cmd_vel.linear.y = velocity[1]
        
        # self.cmd_vel_pub.publish(cmd_vel)
        self.update_sim_odom(cmd_vel)

    def update_sim_odom(self, cmd_vel):
        """Update odometry while preserving orientation"""
        
        # Update position only (orientation comes from odom callback)
        next_pos_x = self.current_pos[0] + cmd_vel.linear.x * self.dt
        next_pos_y = self.current_pos[1] + cmd_vel.linear.y * self.dt
        # Create message
        pose_msg = Pose()
        pose_msg.position.x = float(next_pos_x)
        pose_msg.position.y = float(next_pos_y)
        pose_msg.position.z = float(self.current_pos[2])
        if self.current_orientation is not None:
            pose_msg.orientation = self.current_orientation
        else:
            pose_msg.orientation.w = 1.0

        self.control_odom_pub.publish(pose_msg)


#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped, PointStamped
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
import tf2_geometry_msgs
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import threading

import numpy as np
from scipy.spatial.transform import Rotation as R


class TfHandlerNode(Node):
    def __init__(self):
        super().__init__('tf_handler_node')
        
        # Declare parameters
        self.declare_parameter('set_first_pose', True, 
                             ParameterDescriptor(description='Set first odom pose as map origin'))
        self.declare_parameter('use_odom_msg', True,
                             ParameterDescriptor(description='Use Odometry message type (false for PoseStamped)'))
        self.declare_parameter('odom_topic', '/odom',
                             ParameterDescriptor(description='Input odometry topic'))
        self.declare_parameter('output_topic', '/transformed_odom',
                             ParameterDescriptor(description='Output transformed pose topic'))
        self.declare_parameter('map_frame', 'map',
                             ParameterDescriptor(description='Map frame name'))
        self.declare_parameter('world_frame', 'odom',
                             ParameterDescriptor(description='Original world frame name'))
        self.declare_parameter('gps_mode', False, 
                               ParameterDescriptor(description='Use GPS mode'))
        self.declare_parameter('filter_yaw', False,
                               ParameterDescriptor(description='Filter yaw in GPS mode'))
        # self.declare_parameter('base_link_frame', 'base_link',
        #                      ParameterDescriptor(description='Base link frame name'))
        
        # Get parameters
        self.set_first_pose_ = self.get_parameter('set_first_pose').value
        self.use_odom_msg_ = self.get_parameter('use_odom_msg').value
        self.odom_topic_ = self.get_parameter('odom_topic').value
        self.output_topic_ = self.get_parameter('output_topic').value
        self.map_frame_ = self.get_parameter('map_frame').value
        self.world_frame_ = self.get_parameter('world_frame').value
        self.gps_mode_ = self.get_parameter('gps_mode').value
        self.filter_yaw_ = self.get_parameter('filter_yaw').value
        # self.base_link_frame_ = self.get_parameter('base_link_frame').value
        
        # Initialize variables
        self.first_odom_received_ = False
        self.first_odom_pose_ = None
        self.map_origin_transform_ = None
        self.static_tf_sent_ = False
        self.world_to_map_ = None
        
        # Setup callback groups
        odom_group = MutuallyExclusiveCallbackGroup()
        
        # Setup TF
        self.tf_buffer_ = Buffer()
        self.tf_listener_ = TransformListener(self.tf_buffer_, self)
        self.tf_broadcaster_ = TransformBroadcaster(self)
        self.static_tf_broadcaster_ = StaticTransformBroadcaster(self)
        
        # Setup subscribers and publishers
        if self.use_odom_msg_:
            self.odom_sub_ = self.create_subscription(
                Odometry, 
                self.odom_topic_, 
                self.odom_callback,
                10,
                callback_group=odom_group
            )
        else:
            self.odom_sub_ = self.create_subscription(
                PoseStamped, 
                self.odom_topic_, 
                self.pose_callback,
                10,
                callback_group=odom_group
            )
        
        # Setup publisher - always publish as PoseStamped
        self.pose_pub_ = self.create_publisher(PoseStamped, self.output_topic_, 10)        
        
        # Subscribe to posestamped waypoint  
        self.wp_sub_ = self.create_subscription(
            PoseStamped,
            '/planner/waypoint',
            self.waypoint_callback,
            10
        )
        # For air-router interface
        self.wp_pub_ = self.create_publisher(PointStamped, '/planner/gps_waypoint', 10)

        # Broadcast static TF if map frame is set and we're NOT using first pose
        # If we're using first pose, we'll broadcast the static TF after receiving the first message
        if self.map_frame_ and self.map_frame_ != "" and not self.set_first_pose_:
            self.broadcast_static_tf_identity()
        
        self.init_pose_buffer = []
        
        self.get_logger().info(f"TF Handler Node initialized")
        self.get_logger().info(f"Set first pose: {self.set_first_pose_}")
        self.get_logger().info(f"Use odom msg: {self.use_odom_msg_}")
        self.get_logger().info(f"Input topic: {self.odom_topic_}")
        self.get_logger().info(f"Output topic: {self.output_topic_}")
        self.get_logger().info(f"Map frame: {self.map_frame_}")
        self.get_logger().info(f"World frame: {self.world_frame_}")
        self.get_logger().info(f"GPS mode: {self.gps_mode_}")
        self.get_logger().info(f"Filter yaw: {self.filter_yaw_}")

        
    def broadcast_static_tf_identity(self):
        """Broadcast static transform from world to map frame with identity transform"""
        static_transform = TransformStamped()
        static_transform.header.stamp = self.get_clock().now().to_msg()
        static_transform.header.frame_id = self.world_frame_
        static_transform.child_frame_id = self.map_frame_
        
        # Identity transform
        static_transform.transform.translation.x = 0.0
        static_transform.transform.translation.y = 0.0
        static_transform.transform.translation.z = 0.0
        static_transform.transform.rotation.x = 0.0
        static_transform.transform.rotation.y = 0.0
        static_transform.transform.rotation.z = 0.0
        static_transform.transform.rotation.w = 1.0
        
        self.static_tf_broadcaster_.sendTransform(static_transform)
        self.static_tf_sent_ = True
        self.get_logger().info(f"Broadcasting static TF (identity): world -> {self.map_frame_}")


    def broadcast_static_tf_from_first_pose(self, pose):
        """Broadcast static transform from world to map frame using first pose"""
        static_transform = TransformStamped()
        static_transform.header.stamp = self.get_clock().now().to_msg()
        static_transform.header.frame_id = self.world_frame_
        static_transform.child_frame_id = self.map_frame_
        
        # If in GPS Mode, first, keep the fixed map to world rotation, only set translation
        

        # Use the first pose as the transform from world to map
        static_transform.transform.translation.x = pose.position.x
        static_transform.transform.translation.y = pose.position.y
        static_transform.transform.translation.z = pose.position.z
        static_transform.transform.rotation.x = pose.orientation.x
        static_transform.transform.rotation.y = pose.orientation.y
        static_transform.transform.rotation.z = pose.orientation.z
        static_transform.transform.rotation.w = pose.orientation.w
        
        self.static_tf_broadcaster_.sendTransform(static_transform)
        self.static_tf_sent_ = True
        self.get_logger().info(f"Broadcasting static TF (from first pose): world -> {self.map_frame_}")
        self.get_logger().info(f"Transform: [{pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f}]")
        

    def odom_callback(self, msg):
        """Handle Odometry message"""
        self.process_pose(msg.header, msg.pose.pose)
        

    def pose_callback(self, msg):
        """Handle PoseStamped message"""
        self.process_pose(msg.header, msg.pose)

    def process_pose(self, header, pose_msg):
        """Process incoming pose message"""
        if not self.set_first_pose_:
            # publish dummy odom with correct frame
            self.publish_dummy_odom(header, pose_msg)
            return
        
        # check that pose is valid
        if not self.first_odom_received_:
            curr_pos = np.array([pose_msg.position.x,
                                 pose_msg.position.y,
                                 pose_msg.position.z])
            self.init_pose_buffer.append(curr_pos)
            if len(self.init_pose_buffer) > 100:
                first_pos = self.init_pose_buffer[0]
                dist = np.linalg.norm(curr_pos - first_pos)
                self.get_logger().info(f"Dist: {dist:.3f} m")
                self.get_logger().info(f"Curr pose {curr_pos}")
                if dist > 5.0:
                    self.init_pose_buffer = []                
                    return
                self.get_logger().info(f"Converged, curr pose {curr_pos}")
            else:
                return

        # Transform and publish
        if not self.first_odom_received_:
            self.set_first_pose_as_origin(pose_msg)
            
            # Broadcast static TF using the first pose if map frame is set
            if self.map_frame_ and self.map_frame_ != "" and not self.static_tf_sent_:
                self.broadcast_static_tf_from_first_pose(pose_msg)
            
        # Transform current pose to map frame
        transformed_pose = self.transform_pose_to_map(header, pose_msg)
        if transformed_pose:
            self.pose_pub_.publish(transformed_pose)
            

    def set_first_pose_as_origin(self, pose_msg):
        """Set the first received pose as the map origin"""
        self.first_odom_pose_ = pose_msg
        
        pos = np.array([self.first_odom_pose_.position.x,
                        self.first_odom_pose_.position.y,
                        self.first_odom_pose_.position.z])
        ori = self.first_odom_pose_.orientation
        self.first_yaw_ = R.from_quat([ori.x, ori.y, ori.z, ori.w]).as_euler('xyz')[2]

        # If in GPS Mode, first, keep the fixed map to world rotation, only set translation
        if self.gps_mode_:
            # the rotation is FLU to ENU
            map2world_r = R.from_euler('xyz', [0, 0, np.pi/2])  # 90 deg yaw
            world2map_r = map2world_r.inv()
        else:
            map2world_r = R.from_quat([ori.x, ori.y, ori.z, ori.w])
            world2map_r = map2world_r.inv()

        world2map = np.eye(4)
        world2map[0:3, 0:3] = world2map_r.as_matrix() # R^T
        world2map[0:3, 3] = -world2map_r.as_matrix() @ pos # -R^T * t
        self.world_to_map_ = world2map
        self.map_to_world_ = np.linalg.inv(world2map)

        self.first_odom_received_ = True
        self.get_logger().info(f"Set first pose as map origin: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        

    def transform_pose_to_map(self, header, pose_msg):
        """Transform pose to map frame using the stored origin transform"""
        if self.world_to_map_ is None:
            self.get_logger().warn("World to map transform not set yet.")
            return None

        # Current pose → homogeneous transform
        pos = np.array([pose_msg.position.x,
                        pose_msg.position.y,
                        pose_msg.position.z])
        ori = pose_msg.orientation
        rot = R.from_quat([ori.x, ori.y, ori.z, ori.w]).as_matrix()
        
        # If in GSP, since yaw is aligned with Y in ENU, we should convert to ROS convention
        if self.gps_mode_:
            # Extract yaw and +90
            euler = R.from_matrix(rot).as_euler('xyz')
            euler[2] += np.pi/2  # +90 deg
            rot = R.from_euler('xyz', euler).as_matrix()
        if self.filter_yaw_:
            # When filtering yaw. We assume first yaw is 0.0, compensate all subsequent yaw
            euler = R.from_matrix(rot).as_euler('xyz')
            euler[2] -= self.first_yaw_
            rot = R.from_euler('xyz', euler).as_matrix()

        T_world = np.eye(4)
        T_world[:3, :3] = rot
        T_world[:3, 3] = pos

        # world to map x robot to world = robot to map
        T_map = self.world_to_map_ @ T_world

        # Convert back to PoseStamped
        transformed_pose = PoseStamped()
        transformed_pose.header.stamp = header.stamp
        transformed_pose.header.frame_id = self.map_frame_

        transformed_pose.pose.position.x = T_map[0, 3]
        transformed_pose.pose.position.y = T_map[1, 3]
        transformed_pose.pose.position.z = T_map[2, 3]

        quat = R.from_matrix(T_map[:3, :3]).as_quat()
        transformed_pose.pose.orientation.x = quat[0]
        transformed_pose.pose.orientation.y = quat[1]
        transformed_pose.pose.orientation.z = quat[2]
        transformed_pose.pose.orientation.w = quat[3]

        return transformed_pose


    def publish_dummy_odom(self, header, pose_msg):
        """Publish dummy odometry in map frame"""
        dummy_pose = PoseStamped()
        dummy_pose.header.stamp = header.stamp
        dummy_pose.header.frame_id = self.map_frame_
        # same pose
        dummy_pose.pose = pose_msg
        self.pose_pub_.publish(dummy_pose)


    def waypoint_callback(self, msg):
        """This is mainly for GPS mode, converting ROS FLU to GPS ENU retaining yaw"""
        # Get the original msg position and change frames
        # Keep orientation
        if not self.gps_mode_:
            return
        if not self.first_odom_received_:
            self.get_logger().warn("No first odom received yet, cannot transform waypoint.")
            return
        # Transform waypoint from map to world frame
        # Then add -90 degrees in yaw
        wp_pos_map = np.array([msg.pose.position.x,
                               msg.pose.position.y,
                               msg.pose.position.z])
        wp_ori_map = msg.pose.orientation
        rot_map = R.from_quat([wp_ori_map.x, wp_ori_map.y, wp_ori_map.z, wp_ori_map.w]).as_matrix()
        wp_map_mat = np.eye(4)
        wp_map_mat[0:3, 0:3] = rot_map
        wp_map_mat[0:3, 3] = wp_pos_map
        wp_world_mat = self.map_to_world_ @ wp_map_mat
        # PointStamped does not have orientation, so we don't touch it
        # but it needs to be minus 90 degrees in yaw to GPS convention,
        # Then if filter yaw, we also has to add it back
        wp_world = PointStamped()
        wp_world.header.stamp = msg.header.stamp
        wp_world.header.frame_id = self.world_frame_
        wp_world.point.x = wp_world_mat[0, 3]
        wp_world.point.y = wp_world_mat[1, 3]
        wp_world.point.z = wp_world_mat[2, 3]
        self.get_logger().info(f"Publishing gps waypoint: [{wp_world.point.x:.3f}, {wp_world.point.y:.3f}, {wp_world.point.z:.3f}]")
        self.wp_pub_.publish(wp_world)




def main(args=None):
    rclpy.init(args=args)
    
    node = TfHandlerNode()
    executor = MultiThreadedExecutor()
    
    executor.add_node(node)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    try:
        executor_thread.start()
    finally:
        executor_thread.join()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
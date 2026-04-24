"""
    Yuezhan Tao
    July 2025

    occ_map class for ros2 node
"""

import os
import rclpy
from collections import deque
import numpy as np
import cv2
import struct
import copy
import math
import threading

from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import TransformStamped, PoseStamped, Pose, Point32
from sensor_msgs.msg import PointCloud2, PointField, PointCloud
from std_msgs.msg import Header, Bool
from air_sem_explorer.utils.utils import quat_msg_to_rot_mat, frontier_to_msg, load_config
from air_sem_explorer.mapper.frontier_detector import FrontierDetector
from air_sem_explorer.mapper.frontier_visualizer import FrontierVisualizer
from air_sem_explorer_interfaces.msg import Frontiers
from ament_index_python.packages import get_package_share_directory

from message_filters import Subscriber, ApproximateTimeSynchronizer
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


class OccMap(Node):

    def __init__(self) -> None:
        super().__init__("occ_map_node")

        self.alt_ = 0.0
        self.cvbridge_ = CvBridge()

        self.global_pc_ = None
        self.map_origin_ = None
        self.process_bag_ = False

        self.cam2body_ = np.array([[0.0, -1.0, 0.0, 0.0],
                                   [-1.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, -1.0, 0.0],
                                   [0.0, 0.0, 0.0, 1.0]])

        # For common param, load through yaml
        config_file = os.path.join(get_package_share_directory('air_sem_explorer'), 'config', 'map_common.yaml')
        map_config = load_config(config_file)
        
        self.depth_scale_ = map_config['depth_scale']
        self.map_min_x_ = map_config['map_min_x']
        self.map_max_x_ = map_config['map_max_x']
        self.map_min_y_ = map_config['map_min_y']
        self.map_max_y_ = map_config['map_max_y']
        self.resolution_ = map_config['resolution']
        self.use_odom_msg_ = map_config['use_odom_msg']

        # ROS2
        self.declare_parameter('depth_topic', '/img') 
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('img_interval', 0.5)  # seconds
        self.declare_parameter('frontier_size_min', 3.0)
        self.declare_parameter('frontier_size_max', 5.0)
        self.declare_parameter('intrinsics_topic', '/camera_info')
        # For vggt/robot use
        self.declare_parameter('clear_close_frontiers', False)
        self.declare_parameter('clear_distance', 3.0)  # meters
        self.declare_parameter('clear_frontier_odom_topic', '/clear_odom')

        # Retrieve the parameters
        self.depth_topic_ = self.get_parameter('depth_topic').value
        self.odom_topic_ = self.get_parameter('odom_topic').value
        self.intrinsics_topic_ = self.get_parameter('intrinsics_topic').value
        self.img_interval_ = self.get_parameter('img_interval').value
        self.ftr_size_min_ = self.get_parameter('frontier_size_min').value
        self.ftr_size_max_ = self.get_parameter('frontier_size_max').value
        self.clear_close_frontiers_ = self.get_parameter('clear_close_frontiers').value
        self.clear_distance_ = self.get_parameter('clear_distance').value
        self.clear_frontier_odom_topic_ = self.get_parameter('clear_frontier_odom_topic').value

        # Print params for debugging:
        rclpy.logging.get_logger('occ_map_node').info(f'Depth topic: {self.depth_topic_}')
        rclpy.logging.get_logger('occ_map_node').info(f'Odom topic: {self.odom_topic_}')
        rclpy.logging.get_logger('occ_map_node').info(f'Intrinsics topic: {self.intrinsics_topic_}')
        rclpy.logging.get_logger('occ_map_node').info(f'Depth scale: {self.depth_scale_}')
        rclpy.logging.get_logger('occ_map_node').info(f'Map min x: {self.map_min_x_}')
        rclpy.logging.get_logger('occ_map_node').info(f'Map max x: {self.map_max_x_}')
        rclpy.logging.get_logger('occ_map_node').info(f'Map min y: {self.map_min_y_}')
        rclpy.logging.get_logger('occ_map_node').info(f'Map max y: {self.map_max_y_}')
        rclpy.logging.get_logger('occ_map_node').info(f'Resolution: {self.resolution_}')
        rclpy.logging.get_logger('occ_map_node').info(f'Use odom msg: {self.use_odom_msg_}')
        rclpy.logging.get_logger('occ_map_node').info(f'Image interval: {self.img_interval_}')
        rclpy.logging.get_logger('occ_map_node').info(f'Frontier size min: {self.get_parameter("frontier_size_min").value}')
        rclpy.logging.get_logger('occ_map_node').info(f'Frontier size max: {self.get_parameter("frontier_size_max").value}')
        rclpy.logging.get_logger('occ_map_node').info(f'---------------VGGT / ROBOT----------------')
        rclpy.logging.get_logger('occ_map_node').info(f'Clear close frontiers: {self.clear_close_frontiers_}')
        rclpy.logging.get_logger('occ_map_node').info(f'Clear distance: {self.clear_distance_}')
        rclpy.logging.get_logger('occ_map_node').info(f'Clear frontier odom topic: {self.clear_frontier_odom_topic_}')

        # Init buffers
        self.map_origin_ = np.array((self.map_min_x_, self.map_min_y_))
        self.map_size_ = np.array((self.map_max_x_ - self.map_min_x_,
                          self.map_max_y_ - self.map_min_y_))
        self.map_voxel_num_ = (np.ceil(self.map_size_[0] / self.resolution_),
                               np.ceil(self.map_size_[1] / self.resolution_))
        self.map_voxel_num_ = np.array(self.map_voxel_num_, dtype=int)
        # Init as unknown (-1)
        self.occ_map_ = np.full((int(self.map_voxel_num_[0]),
                                 int(self.map_voxel_num_[1])), -1, dtype=np.int8)
        self.x_coords = self.map_origin_[0] + (np.arange(self.map_voxel_num_[0]) + 0.5) * self.resolution_
        self.y_coords = self.map_origin_[1] + (np.arange(self.map_voxel_num_[1]) + 0.5) * self.resolution_
        # Bounding box for map updates - will be used by frontier detection
        self.update_min_ = np.array([0, 0])
        self.update_max_ = np.array([0, 0])
        self.update_box_reset_ = True

        self.ftr_lock_ = threading.Lock()

        img_group = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()
        intrinsics_group = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()
        odom_group = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()
        pub_group = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()
        ftr_vis_group = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()
        ftr_group = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()


        # PC field
        self.pc_fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)
        ]

        self.occ_map_pub_ = self.create_publisher(OccupancyGrid, "/occ_map/occ_map", 10)
        self.pc_pub_ = self.create_publisher(PointCloud, "/occ_map/occ_map_pc", 10)
        self.ftr_pub_ = self.create_publisher(Frontiers, "/occ_map/frontiers", 10)

        # keyframe callback based on image buffers
        self.last_img_time_ = self.get_clock().now()


        # The frontier detector
        self.ftr_detector_ = FrontierDetector(self, self.ftr_size_min_, self.ftr_size_max_)
        self.ftr_vis_ = FrontierVisualizer(self)

        self.map_visualize_ = self.create_timer(1.0, self.map_vis_callback, callback_group=pub_group)
        self.ftr_visualize_ = self.create_timer(1.0, self.ftr_vis_callback, callback_group=ftr_vis_group)
        self.ftr_ = self.create_timer(1.0, self.ftr_callback, callback_group=ftr_group)

        self.intrinsics_sub_ = Subscriber(self, CameraInfo, self.intrinsics_topic_, callback_group=intrinsics_group)
        self.img_sub_ = Subscriber(self, Image, self.depth_topic_, callback_group=img_group)
        if self.use_odom_msg_:
            self.odom_sub_ = Subscriber(self, Odometry, self.odom_topic_, callback_group=odom_group)
            self.ts = ApproximateTimeSynchronizer(
                [self.img_sub_, self.odom_sub_, self.intrinsics_sub_],
                queue_size=50,
                slop=0.02)
            self.ts.registerCallback(self.image_odom_callback)
        else:
            self.odom_sub_ = Subscriber(self, PoseStamped, self.odom_topic_, callback_group=odom_group)
            self.ts = ApproximateTimeSynchronizer(
                [self.img_sub_, self.odom_sub_, self.intrinsics_sub_],
                queue_size=50,
                slop=0.02)
            self.ts.registerCallback(self.image_odom_callback)
        
        self.clear_pos_ = None
        if self.clear_close_frontiers_:
            self.clear_odom_sub_ = self.create_subscription(
                PoseStamped,
                self.clear_frontier_odom_topic_,
                self.odom_callback,
                10)

        # no message filter
        self.reset_sub_ = self.create_subscription(Bool, "/occ_map/reset_map", self.reset_callback, 1)

        rclpy.logging.get_logger('occ_map_node').info('Subscribing to: {} and {}'.format(
            self.img_sub_.topic, self.odom_sub_.topic))
        rclpy.logging.get_logger('occ_map_node').info('DataHandler initialized.')

    def odom_callback(self, odom_msg) -> None:
        tmp_odom = odom_msg.pose
        self.clear_pos_ = np.array([tmp_odom.position.x,
                                    tmp_odom.position.y])

    def image_odom_callback(self, img_msg: Image, odom_msg: Odometry, intrinsics_msg: CameraInfo) -> None:
        """
        Callback function for image and odometry messages.
        Processes the incoming data and updates the global point cloud.
        """
        # Convert compressed image to OpenCV format
                # Skip the image if within the last image interval
        if (self.get_clock().now() - self.last_img_time_).nanoseconds < self.img_interval_ * 1e9:
            return

        self.last_img_time_ = self.get_clock().now()

        rclpy.logging.get_logger('occ_map_node').info(f'Received image at {self.last_img_time_.nanoseconds} ns')

        if self.use_odom_msg_:
            tmp_odom = copy.deepcopy(odom_msg.pose.pose)
        else:
            tmp_odom = copy.deepcopy(odom_msg.pose)

        
        self.last_img_time_ = self.get_clock().now()
            
        # In the bag we are dealing with compressed image:
        if self.process_bag_:
            return
        else:
            # Convert the image
            try:
                img = self.cvbridge_.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
            except Exception as e:
                self.get_logger().error(f'Failed to convert image: {e}')
                return
        # Find cam2world
        cam2world = self.get_cam2world(tmp_odom)
        intrinsics = np.array(intrinsics_msg.k).reshape(3, 3)
        s_time = self.get_clock().now()
        pc_cam = self.process_depth_image(img, intrinsics, skip_num=1)
        if pc_cam is None:
            self.get_logger().error('No valid points in depth image.')
            return
        # Transform points to world frame
        # Convert to homogeneous coordinates (Nx3 -> Nx4)
        pc_cam_homo = np.column_stack([pc_cam, np.ones(len(pc_cam))])
        pc_world_homo = (cam2world @ pc_cam_homo.T).T
        pc_world = pc_world_homo[:, :3]

        # Update map
        self.update_map(pc_world)
        e_time = self.get_clock().now()
        rclpy.logging.get_logger('occ_map_node').info(f'Point cloud processing took {(e_time - s_time).nanoseconds / 1e6} ms')

        # Get the bounding box of the point update region
        min_x, max_x = np.min(pc_world[:, 0]), np.max(pc_world[:, 0])
        min_y, max_y = np.min(pc_world[:, 1]), np.max(pc_world[:, 1])

        # Maintain the update bounding box unless reset externally
        if self.update_box_reset_:
            self.update_min_ = np.array([min_x, min_y])
            self.update_max_ = np.array([max_x, max_y])
            self.update_box_reset_ = False
        else:
            self.update_min_ = np.minimum(self.update_min_, [min_x, min_y])
            self.update_max_ = np.maximum(self.update_max_, [max_x, max_y])



    def get_cam2world(self, odom_msg: Pose) -> np.ndarray:
        """
        Convert the odometry message to a camera-to-world transformation matrix.
        """
        # Use cam2body_ to get the camera to world transformation
        # odom is body2world
        pos = odom_msg.position
        quat = odom_msg.orientation
        body2world = np.eye(4)
        body2world[:3, :3] = quat_msg_to_rot_mat(quat)
        body2world[:3, 3] = np.array([pos.x, pos.y, pos.z])
        # Convert to camera to world
        cam2world = body2world @ self.cam2body_
        return cam2world
    
    def map_vis_callback(self) -> None:
        rclpy.logging.get_logger('occ_map_node').info('Visualizing occupancy map...')
        self.publish_occ_map()
        self.publish_pointcloud()

    def ftr_vis_callback(self) -> None:
        rclpy.logging.get_logger('occ_map_node').info('Visualizing frontiers...')
        with self.ftr_lock_:
            if len(self.ftr_detector_.frontiers_) == 0:
                rclpy.logging.get_logger('occ_map_node').info('No frontiers to visualize.')
                return
            self.ftr_vis_.visualize_frontiers(self.ftr_detector_.frontiers_)


    def process_depth_image(self, depth_image, intrinsics, skip_num = 1) -> np.ndarray:
        """
        Process depth image to generate 3D points in camera frame.
        
        Args:
            depth_image: 2D numpy array of depth values (height x width)
            skip_num: int, interval for skipping pixels (e.g., 1 means take every other pixel)
            
        Returns:
            points_3d: Nx3 numpy array of 3D points in camera frame
        """
        # Get camera intrinsics
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        # Create grid of pixel coordinates with skipping
        height, width = depth_image.shape
        x_coords = np.arange(0, width, skip_num + 1)
        y_coords = np.arange(0, height, skip_num + 1)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        depths = depth_image[y_grid, x_grid] / self.depth_scale_

        # Filter out invalid depths (assuming 0 is invalid)
        valid_mask = depths > 0
        x_valid = x_grid[valid_mask]
        y_valid = y_grid[valid_mask]
        z_valid = depths[valid_mask]
        
        # Project to 3D (vectorized)
        x_3d = (x_valid - cx) * z_valid / fx
        y_3d = (y_valid - cy) * z_valid / fy
        z_3d = z_valid
        
        # Stack into Nx3 array
        points_3d = np.column_stack((x_3d, y_3d, z_3d))
        
        return points_3d
    

    def update_map(self, pc: np.ndarray) -> None:
        """
        Update the occupancy map with the new point cloud data.
        Vectorized update for occupancy grid (No log-odds, any point is a hit).
        
        Args:
            pc: Nx3 numpy array of points in world frame
        """
        # Calculate voxel indices for all points at once
        xy_points = pc[:, :2]
        voxel_indices = np.floor((xy_points - np.array(self.map_origin_)) / self.resolution_).astype(int)
        
        # Filter out points outside the map bounds
        in_bounds_mask = (voxel_indices >= 0).all(axis=1) & \
                        (voxel_indices < np.array(self.map_voxel_num_)).all(axis=1)
        valid_voxels = voxel_indices[in_bounds_mask]
        
        # Mark occupied cells (1 = occupied)
        if valid_voxels.size > 0:
            self.occ_map_[valid_voxels[:, 0], valid_voxels[:, 1]] = 1


    def pos_to_index(self, pos: np.ndarray) -> np.ndarray:
        """
        Convert world position to grid index.
        
        Args:
            pos: 2D or 3D numpy array (x,y) or (x,y,z) position in world frame
            
        Returns:
            index: numpy array of grid indices, clamped to map bounds
        """
        index = np.floor((pos - np.array(self.map_origin_)) / self.resolution_).astype(int)
        # Clamp to map bounds (handles both 2D and 3D cases)
        for i in range(len(index)):
            index[i] = np.clip(index[i], 0, self.map_voxel_num_[i] - 1)
        return index

    def index_to_pos(self, index: np.ndarray) -> np.ndarray:
        """
        Convert grid index to world position (center of voxel).
        
        Args:
            index: numpy array of grid indices
            
        Returns:
            pos: numpy array of world position (x,y) or (x,y,z)
        """
        return (index + 0.5) * self.resolution_ + np.array(self.map_origin_)

    def get_occupancy_by_index(self, index: np.ndarray) -> int:
        """
        Get occupancy status by grid index.
        
        Args:
            index: numpy array of grid indices
            
        Returns:
            occupancy: -1 (unknown), 0 (free), 1 (occupied)
        """
        if not self.is_in_map(index):
            return -1
        occ_value = self.occ_map_[tuple(index)]
        # Simplified version without log-odds (adjust thresholds as needed)
        if occ_value == -1:
            return -1  # UNKNOWN
        elif occ_value == 1:
            return 1    # OCCUPIED
        else:
            return 0    # FREE

    def get_occupancy_by_pos(self, pos: np.ndarray) -> int:
        """
        Get occupancy status by world position.
        
        Args:
            pos: numpy array of world position (x,y) or (x,y,z)
            
        Returns:
            occupancy: -1 (unknown), 0 (free), 1 (occupied)
        """
        index = self.pos_to_index(pos)
        return self.get_occupancy_by_index(index)

    def is_in_map(self, index: np.ndarray) -> bool:
        """
        Check if index is within map bounds.
        
        Args:
            index: numpy array of grid indices
            
        Returns:
            bool: True if index is within map bounds
        """
        return all((0 <= index) & (index < np.array(self.map_voxel_num_)))

    def is_known_free(self, index: np.ndarray) -> bool:
        """
        Check if the cell at the given index is known and free.
        
        Args:
            index: numpy array of grid indices
            
        Returns:
            bool: True if cell is known and free
        """
        return self.get_occupancy_by_index(index) == 0
    
    def is_known(self, index: np.ndarray) -> bool:
        """
        Check if the cell at the given index is known.
        
        Args:
            index: numpy array of grid indices
            
        Returns:
            bool: True if cell is known (not -1)
        """
        return self.get_occupancy_by_index(index) != -1

    def is_neighbor_unknown(self, index: np.ndarray) -> bool:
        """
        Check if any neighbor of the cell at the given index is unknown.
        
        Args:
            index: numpy array of grid indices
            
        Returns:
            bool: True if any neighbor is unknown
        """
        # Check if any neighbor is unknown
        for nbr in self.get_neighbors(index):
            if not self.is_in_map(nbr):
                continue
            if self.get_occupancy_by_index(nbr) == -1:
                return True
        return False

    def get_neighbors(self, index: np.ndarray) -> list[np.ndarray]:
        # Returns 4-connected neighbors
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        return [index + np.array(off) for off in offsets]


    def publish_occ_map(self):
        """
        Publish the occupancy map as OccupancyGrid.
        """
        
        # Create OccupancyGrid message
        occ_grid = OccupancyGrid()
        occ_grid.header.stamp = self.get_clock().now().to_msg()
        occ_grid.header.frame_id = "map"
        
        # Set metadata
        occ_grid.info.resolution = float(self.resolution_)
        occ_grid.info.width = int(self.map_voxel_num_[0])
        occ_grid.info.height = int(self.map_voxel_num_[1])
        occ_grid.info.origin.position.x = float(self.map_origin_[0])
        occ_grid.info.origin.position.y = float(self.map_origin_[1])
        occ_grid.info.origin.position.z = 0.0
        occ_grid.info.origin.orientation.w = 1.0
        occ_grid.data = self.occ_map_.flatten(order='F').tolist()

        self.occ_map_pub_.publish(occ_grid)

    def publish_pointcloud(self):
        """
        Publish the map as a pointcloud message
        """
        
        # Create PointCloud message
        pc_msg = PointCloud()
        pc_msg.header.stamp = self.get_clock().now().to_msg()
        pc_msg.header.frame_id = "map"
        
        # Fill points
        pc_msg.points = []
        i, j = np.where(self.occ_map_ == 1)
        points = np.column_stack([
            self.x_coords[i],
            self.y_coords[j],
            np.zeros_like(i)
        ])
        pc_msg.points = [Point32(x=p[0], y=p[1], z=p[2]) for p in points]
        # Publish the point cloud
        self.pc_pub_.publish(pc_msg)

    def create_depth_pointcloud(self, points, frame_id="world"):
        """
        Convert numpy array of 3D points to PointCloud2 message
        
        Args:
            points: Nx3 numpy array containing [x, y, z] coordinates
            frame_id: Reference frame for the point cloud
            
        Returns:
            PointCloud2 message
        """
        # Ensure points is a numpy array
        points = np.asarray(points, dtype=np.float32)
        
        # Create PointCloud2 message
        msg = PointCloud2()
        
        # Header
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        
        # Point cloud dimensions
        msg.height = 1  # Unorganized point cloud
        msg.width = points.shape[0]  # Number of points
        
        # Point fields (x, y, z coordinates)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        
        # Point step and row step
        msg.point_step = 12  # 3 * 4 bytes (float32)
        msg.row_step = msg.point_step * msg.width
        
        # Convert points to bytes
        msg.data = points.tobytes()
        
        # Endianness
        msg.is_bigendian = False
        msg.is_dense = True
        
        return msg


    def ftr_callback(self) -> None:
        """
        Callback for frontier detection.
        """
        if self.update_box_reset_:
            if self.clear_close_frontiers_ and self.clear_pos_ is not None:
                with self.ftr_lock_:
                    self.ftr_detector_.clear_nearby_frontiers(self.clear_pos_, self.clear_distance_)
            rclpy.logging.get_logger('occ_map_node').warn('Update bounding box is reset, skipping frontier detection.')
            return
        with self.ftr_lock_:
            s_time = self.get_clock().now()
            rclpy.logging.get_logger('occ_map_node').info('Detecting frontiers...')
            self.ftr_detector_.search_frontiers(self.update_min_, self.update_max_)
            self.update_box_reset_ = True
            rclpy.logging.get_logger('occ_map_node').info(f'Found {len(self.ftr_detector_.frontiers_)} active frontiers.')

            self.ftr_detector_.sample_viewpoints()
            e_time = self.get_clock().now()
            rclpy.logging.get_logger('occ_map_node').info(f'Frontier detection took {(e_time - s_time).nanoseconds / 1e6} ms')

            # remove close frontiers if needed
            if self.clear_close_frontiers_ and self.clear_pos_ is not None:
                self.ftr_detector_.clear_nearby_frontiers(self.clear_pos_, self.clear_distance_)

            # Publish frontiers
            ftr_msg = frontier_to_msg(self.ftr_detector_.frontiers_)
            ftr_msg.header.stamp = self.get_clock().now().to_msg()
            ftr_msg.header.frame_id = "map"
            self.ftr_pub_.publish(ftr_msg)

    
    def reset_callback(self, msg) -> None:
        """
        Reset the occupancy map and frontier detector.
        """
        self.occ_map_ = np.full((int(self.map_voxel_num_[0]),
                                 int(self.map_voxel_num_[1])), -1, dtype=np.int8)
        self.update_box_reset_ = True
        with self.ftr_lock_:
            self.ftr_detector_.reset_frontiers()
            # Publish frontiers
            ftr_msg = frontier_to_msg(self.ftr_detector_.frontiers_)
            ftr_msg.header.stamp = self.get_clock().now().to_msg()
            ftr_msg.header.frame_id = "map"
            self.ftr_pub_.publish(ftr_msg)
            self.update_box_reset_ = True
        rclpy.logging.get_logger('occ_map_node').info('Occupancy map and frontiers have been reset.')

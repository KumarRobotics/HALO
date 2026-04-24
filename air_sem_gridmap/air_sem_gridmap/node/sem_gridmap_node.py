#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header, Bool
import sensor_msgs_py.point_cloud2 as pc2
from rclpy.callback_groups import ReentrantCallbackGroup
import cv2
from cv_bridge import CvBridge

import numpy as np
import torch
import matplotlib.cm as cm
import yaml
import pathlib
from sklearn.decomposition import PCA

from air_sem_gridmap.sem_gridmap import SemGridMap
from air_sem_gridmap_interfaces.srv import SetPrompt
from air_sem_gridmap_interfaces.msg import RelevancyMap

from air_sem_explorer.utils.utils import map_to_msg
from ament_index_python.packages import get_package_share_directory
import os
import threading


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class SemGridMapNode(Node):
    def __init__(self):
        super().__init__('sem_gridmap_node')
        config_file = os.path.join(get_package_share_directory('air_sem_gridmap'), 'config.yaml')
        node_config = load_config(config_file)

        map_config_file = os.path.join(get_package_share_directory('air_sem_explorer'), 'config', 'map_common.yaml')
        map_config = load_config(map_config_file)

        # merge configs
        config = {**node_config, **map_config}

        self.gridmap = SemGridMap(config)
        self.cv_bridge = CvBridge()

        # ROS2
        self.declare_parameter('depth_topic', '/vggt_mapper/depth/image') 
        self.declare_parameter('rgb_topic', '/vggt_mapper/color/image')
        self.declare_parameter('odom_topic', '/vggt_mapper/pose')
        self.declare_parameter('intrinsic_topic', '/vggt_mapper/camera_info')
        # Retrieve the parameters
        self.depth_topic_ = self.get_parameter('depth_topic').value
        self.rgb_topic_ = self.get_parameter('rgb_topic').value
        self.odom_topic_ = self.get_parameter('odom_topic').value
        self.intrinsic_topic_ = self.get_parameter('intrinsic_topic').value

        # Subscribers
        self.rgb_sub = Subscriber(self, Image, self.rgb_topic_)
        self.depth_sub = Subscriber(self, Image, self.depth_topic_)
        self.pose_sub = Subscriber(self, PoseStamped, self.odom_topic_)
        self.intrinsics_sub = Subscriber(self, CameraInfo, self.intrinsic_topic_)
        # Synchronizer
        self.ts = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.pose_sub, self.intrinsics_sub],
            queue_size=10,
            slop=0.01
        )
        self.ts.registerCallback(self.rgbd_odom_callback)
        # Publishers
        self.relevancy_map_pub = self.create_publisher(RelevancyMap, 'relevancy_map', 1)
        self.publish_pointcloud = config.get('publish_pointcloud', False)
        self.publish_relevancy_viz = config.get('publish_relevancy_viz', False)
        self.publish_feature_viz = config.get('publish_feature_viz', False)
        # Use a shared callback group for all publishers and timers
        self.viz_callback_group = ReentrantCallbackGroup()
        if self.publish_pointcloud:
            self.pc_pub = self.create_publisher(PointCloud2, 'sem_gridmap/pointcloud', 1, callback_group=self.viz_callback_group)
            self.pc_pub_timer = self.create_timer(1.0, self.publish_all_points, callback_group=self.viz_callback_group)
        if self.publish_relevancy_viz:            
            self.relevancy_pub = self.create_publisher(PointCloud2, 'sem_gridmap/relevancy_viz', 1, callback_group=self.viz_callback_group)
            self.relevancy_pub_timer = self.create_timer(1.0, self.publish_relevancy_voxels, callback_group=self.viz_callback_group)
            self.relevancy_viz_height = config.get('relevancy_viz_height', -10.0)
        if self.publish_feature_viz:
            self.feature_viz_pub = self.create_publisher(PointCloud2, 'sem_gridmap/feature_viz', 1, callback_group=self.viz_callback_group)
            self.feature_viz_pub_timer = self.create_timer(5.0, self.publish_feature_map_viz, callback_group=self.viz_callback_group)
            self.feature_viz_height = config.get('feature_viz_height', -20.0)
        self.publish_viz_at_offset = config.get('publish_viz_at_offset', False)
        # Service to update text prompt
        self.prompt_service = self.create_service(
            SetPrompt, 'set_prompt', self.handle_set_text_prompt,
            callback_group=ReentrantCallbackGroup()
        )
        # sub to reset map
        if config.get('enable_reset_map', False):
            self.reset_map_sub = self.create_subscription(Bool, '/occ_map/reset_map', self.reset_map_callback, 1)
        self.task_id = 0
        self.last_pose = None
        self.processing_lock = threading.Lock()
        self.get_logger().info("SemGridMapNode initialized!")

    def handle_set_text_prompt(self, request, response):
        """
        Handle service call to set the text prompt for the semantic grid map.
        The prompt is expected to be a comma-separated list of objects.
        """
        self.task_id = request.task_id
        prompt = request.task_prompt
        # comma-separated list of objects, convert to list of strings
        prompt = prompt.split(',')
        prompt = [p.strip() for p in prompt]
        if not prompt:
            response.success = False
            response.message = 'No prompt provided.'
            return response
        self.processing_lock.acquire()
        self.gridmap.set_text_prompt(prompt)
        self.processing_lock.release()
        response.success = True
        
        # publish relevancy map with new task
        self.publish_relevancy_map(local_update=False)
        
        response.message = f'Prompt set to: {prompt}'
        self.get_logger().info(f"Text prompt updated to: {prompt}")
        return response

    @torch.no_grad()
    def rgbd_odom_callback(self, rgb_msg, depth_msg, pose_msg, intrinsics_msg):
        """
        Callback for synchronized RGB, depth, pose, and intrinsics messages.
        """
        # return if pose has not changed by more than 1m
        if self.last_pose is not None:
            pos_diff = np.linalg.norm(np.array([pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z]) - self.last_pose)
            if pos_diff < 1.0:
                return
        self.last_pose = np.array([pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z])

        # Convert ROS Image messages to numpy arrays
        rgb = self.cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
        depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        intrinsics = np.array(intrinsics_msg.k).reshape(3, 3)
        # Convert ROS PoseStamped to [x, y, z, qx, qy, qz, qw]
        pos = pose_msg.pose.position
        quat = pose_msg.pose.orientation
        pose_arr = np.array([pos.x, pos.y, pos.z, quat.w, quat.x, quat.y, quat.z], dtype=np.float32)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        self.processing_lock.acquire()
        self.gridmap.process_measurement(rgb, depth, pose_arr, intrinsics)        
        self.processing_lock.release()
        
        # publish relevancy map
        self.publish_relevancy_map(local_update=True)
        # # publish relevancy map visualization
        # if self.publish_relevancy_viz:
        #     voxel_points, voxel_values = self.gridmap.get_relevancy_voxels()
        #     if voxel_points is not None and voxel_values is not None and len(voxel_points) > 0:
        #         self.publish_relevancy_voxels(voxel_points, voxel_values, frame_id="world", scale=1.0)
                
        # # full pointcloud viz
        # if self.publish_pointcloud:
        #     self.publish_all_points()

        # # feature viz
        # if self.publish_feature_viz:
        #     self.publish_feature_map_viz()        
        
        end.record()
        torch.cuda.synchronize()
        self.get_logger().info(f"Time taken: {start.elapsed_time(end)} ms")

    def publish_all_points(self):
        """
        Publish the global colored point cloud for visualization.
        """
        self.processing_lock.acquire()
        all_points = self.gridmap.global_points.detach().clone()
        all_colors = self.gridmap.global_colors.detach().clone()
        self.processing_lock.release()
        if all_points is None or all_colors is None or all_points.shape[0] == 0:
            return

        points_np = all_points.cpu().numpy() if hasattr(all_points, 'cpu') else all_points
        colors_np = (all_colors.cpu().numpy() * 255).astype(np.uint8) if hasattr(all_colors, 'cpu') else (all_colors * 255).astype(np.uint8)
        rgb = colors_np[:, 0].astype(np.uint32) << 16 | colors_np[:, 1].astype(np.uint32) << 8 | colors_np[:, 2].astype(np.uint32)
        rgb = rgb.view(np.float32)
        points = np.hstack([points_np, rgb.reshape(-1, 1)])
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"
        pc_msg = pc2.create_cloud(header, fields, points)
        self.pc_pub.publish(pc_msg)

    def publish_relevancy_voxels(self, frame_id="map", scale=1.0):
        """
        Publish the relevancy voxels as a PointCloud2 message for visualization in RViz.
        The color of each voxel is determined by its relevancy value using a colormap.
        """
        self.processing_lock.acquire()
        points, values = self.gridmap.get_relevancy_voxels()
        self.processing_lock.release()
        if points is None or len(points) == 0:
            # self.get_logger().info("No relevancy voxels to publish.")
            return
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "relevancy_voxels"
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.scale.x = scale * self.gridmap.grid_resolution
        marker.scale.y = scale * self.gridmap.grid_resolution
        marker.scale.z = scale * self.gridmap.grid_resolution
        marker.pose.orientation.w = 1.0
        cmap = cm.get_cmap('jet')
        colors = cmap(values)[:, :3]

        # Normalize to 0-1 range for float conversion
        colors = colors.astype(np.float32)

        # Create structured array with proper field names
        points_3d = points.reshape(-1, 3)
        if self.publish_viz_at_offset:
            height = self.gridmap.highest_point + self.relevancy_viz_height
        else:
            height = 0.0
        points_3d[:, 2] = height  # set z to fixed height
        points_with_color = np.zeros(len(points_3d), dtype=[
            ('x', np.float32),
            ('y', np.float32), 
            ('z', np.float32),
            ('rgb', np.float32)
        ])

        points_with_color['x'] = points_3d[:, 0]
        points_with_color['y'] = points_3d[:, 1]
        points_with_color['z'] = points_3d[:, 2]

        # Pack RGB into float32 (standard ROS format: 0x00RRGGBB)
        rgb_packed = ((colors[:, 0] * 255).astype(np.uint32) << 16 | 
                    (colors[:, 1] * 255).astype(np.uint32) << 8 | 
                    (colors[:, 2] * 255).astype(np.uint32)).view(np.float32)

        points_with_color['rgb'] = rgb_packed

        # Create PointField definitions
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id

        # Create cloud from structured array
        pc_msg = pc2.create_cloud(header, fields, points_with_color)
        self.relevancy_pub.publish(pc_msg)

    def publish_relevancy_map(self, local_update=True):
        """
        Publish the relevancy map as a RelevancyMap message.
        """
        self.processing_lock.acquire()
        try:
            grid_info = self.gridmap.get_grid_info(local_update)
        except Exception as e:
            self.get_logger().error(f"Error getting grid info: {e}")
            grid_info = None
        self.processing_lock.release()
        if grid_info is None:
            self.get_logger().warn("No relevancy data available to publish")
            return
        
        origin, size, voxel_num, resolution, data = grid_info
        
        relevancy_msg = map_to_msg(
            origin=origin,
            size=size,
            voxel_num=voxel_num,
            resolution=resolution,
            data=data
        )
        relevancy_msg.header.stamp = self.get_clock().now().to_msg()
        relevancy_msg.header.frame_id = "map"
        relevancy_msg.task_id = self.task_id
        self.relevancy_map_pub.publish(relevancy_msg)
        self.get_logger().info(f"Published RelevancyMap: task_id={self.task_id}, origin={origin}, size={size}, voxel_num={voxel_num}, resolution={resolution}")
        
    def publish_feature_map_viz(self):
        """Publish a visualization of the feature map using PCA."""
        # get PCA on feature grid and publish
        self.processing_lock.acquire()
        feature_grid = self.gridmap.grid_features.detach().cpu().numpy()
        self.processing_lock.release()
        # reshape to (N, D)
        W, H, D = feature_grid.shape
        feature_grid_flat = feature_grid.reshape(-1, D)
        # only keep cells whose channels are not all zeros
        valid_mask = feature_grid_flat.sum(axis=1) != 0
        if np.sum(valid_mask) < 5:
            # self.get_logger().info("No valid feature cells to publish.")
            return
        feature_grid_valid = feature_grid_flat[valid_mask]
        # PCA to 3 channels
        pca = PCA(n_components=3)
        feature_pca = pca.fit_transform(feature_grid_valid)
        # normalize to 0-1
        feature_pca = (feature_pca - feature_pca.min()) / (feature_pca.max() - feature_pca.min() + 1e-6)
        # get voxel coordinates for valid cells
        valid_indices = np.nonzero(valid_mask)[0]
        # convert flat indices to 2D indices
        y_indices = valid_indices // H
        x_indices = valid_indices % H
        voxel_coords_valid = np.stack([y_indices, x_indices], axis=1)
        # convert to world coordinates
        voxel_coords_valid = voxel_coords_valid * self.gridmap.grid_resolution + self.gridmap.grid_origin.cpu().numpy()
        # publish as point cloud
        points = voxel_coords_valid
        if self.publish_viz_at_offset:
            height = self.gridmap.highest_point + self.feature_viz_height
        else:
            height = 0.0
        points = np.hstack([points, np.full((points.shape[0], 1), height)])  # set z=height
        # points = np.hstack([points, np.zeros((points.shape[0], 1))])  # add z=0
        colors = (feature_pca * 255).astype(np.uint8)
        rgb = colors[:, 0].astype(np.uint32) << 16 | colors[:, 1].astype(np.uint32) << 8 | colors[:, 2].astype(np.uint32)
        rgb = rgb.view(np.float32)
        points = np.hstack([points, rgb.reshape(-1, 1)])
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"
        pc_msg = pc2.create_cloud(header, fields, points)
        self.feature_viz_pub.publish(pc_msg)

    def reset_map_callback(self, msg):
        """Callback to reset the semantic grid map."""
        self.processing_lock.acquire()
        self.gridmap.reset_map()
        self.processing_lock.release()
        self.last_pose = None
        self.get_logger().info("Semantic grid map has been reset.")


def main(args=None):
    rclpy.init(args=args)
    node = SemGridMapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 
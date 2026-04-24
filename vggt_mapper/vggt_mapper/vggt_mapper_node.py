#!/usr/bin/env python3

import os
import sys
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path, Odometry
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Header, ColorRGBA
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs
from message_filters import ApproximateTimeSynchronizer, Subscriber

import numpy as np
import torch
import cv2
from tqdm.auto import tqdm
import tempfile
import threading
from queue import Queue, Empty
from collections import deque
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2

from scipy.spatial.transform import Rotation as _R
import yaml
from ament_index_python.packages import get_package_share_directory

# Add the VGGT-SLAM path to sys.path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
vggt_slam_path = os.path.join(current_dir, '../../VGGT-SLAM')
if vggt_slam_path not in sys.path:
    sys.path.insert(0, vggt_slam_path)

# Import VGGT-SLAM modules
import vggt_slam.slam_utils as utils
from vggt_slam.solver import Solver
from vggt.models.vggt import VGGT

from vggt_mapper.scale_estimation import estimate_scale
from gtsam import NonlinearFactorGraph, Values, noiseModel

import time

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class VGGTSlamNode(Node):
    def __init__(self):
        super().__init__('vggt_slam_node')
        
        self.declare_parameter('config_path', 'config/vggt_mapper.yaml').get_parameter_value().string_value
        config_path = self.get_parameter('config_path').get_parameter_value().string_value

        config_file = os.path.join(
            get_package_share_directory('vggt_mapper'),
            config_path
        )
        config = load_config(config_file)
        
        # Load parameters from config dict
        self.use_sim = config.get('use_sim', False)
        self.sim_image_topic = config.get('sim_image_topic', '/quadrotor/color/image')
        self.sim_pose_topic = config.get('sim_pose_topic', '/quadrotor/pose')
        if self.use_sim:
            self.image_topic = self.sim_image_topic
            self.pose_topic = self.sim_pose_topic
        else:
            self.image_topic = config.get('image_topic', '/quadrotor/color/image')
            self.pose_topic = config.get('pose_topic', '/quadrotor/pose')
        self.image_distance = config.get('image_distance', 2.0)
        self.submap_size = config.get('submap_size', 16)
        self.overlapping_window_size = config.get('overlapping_window_size', 1)
        self.max_loops = config.get('max_loops', 1)
        self.conf_threshold = config.get('conf_threshold', 25.0)
        self.use_point_map = config.get('use_point_map', False)
        self.use_sim3 = config.get('use_sim3', False)
        self.ptcloud_viz_resolution = config.get('ptcloud_viz_resolution', 0.1)
        self.use_external_poses = config.get('use_external_poses', True)
        self.pose_sync_tolerance = config.get('pose_sync_tolerance', 0.1)
        self.enable_loop_closure = config.get('enable_loop_closure', False)
        self.enable_icp = config.get('enable_icp', True)
        self.enable_gps_prior = config.get('enable_gps_prior', False)
        self.loop_distance = config.get('loop_distance', 30.0)
        self.loop_id_threshold = config.get('loop_id_threshold', 4)
        
        self.get_logger().info(f'Subscribing to image topic: {self.image_topic}')
        self.get_logger().info(f'Subscribing to pose topic: {self.pose_topic}')
        self.get_logger().info(f'Use external poses: {self.use_external_poses}')
        self.get_logger().info(f'Pose sync tolerance: {self.pose_sync_tolerance}s')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Initialize device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Using device: {self.device}")
        
        # Initialize VGGT-SLAM solver
        self.get_logger().info("Initializing VGGT-SLAM solver...")
        self.solver = Solver(
            init_conf_threshold=self.conf_threshold,
            use_point_map=self.use_point_map,
            use_sim3=self.use_sim3,
            gradio_mode=False,
            enable_loop_closure=self.enable_loop_closure,
            enable_icp=self.enable_icp,
            overlap_window_size=self.overlapping_window_size
        )

        # Initialize and load VGGT model
        self.get_logger().info("Loading VGGT model...")
        self.model = VGGT()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        self.model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
        self.model.eval()
        self.model = self.model.to(self.device)
        self.get_logger().info("VGGT model loaded successfully")
        
        # Image processing variables
        self.temp_dir = tempfile.mkdtemp()
        self.image_counter = 0
        self.image_names_subset = []
        self.image_queue = Queue(maxsize=100)
        self.processing_lock = threading.Lock()
        
        # Synchronized data storage
        self.synchronized_data = []  # Store (image_filename, pose, timestamp) tuples
        
        # Rate limiting for synchronized callback (2Hz)
        self.last_sync_time = 0.0
        self.sync_callback_rate = 0.5  # seconds (2Hz)
        
        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Publishers
        self.pointcloud_publisher = self.create_publisher(PointCloud2, 'vggt_mapper/pointcloud', 1)
        self.rgb_publisher = self.create_publisher(Image, 'vggt_mapper/color/image', 1)
        self.depth_publisher = self.create_publisher(Image, 'vggt_mapper/depth/image', 1)
        self.pose_publisher = self.create_publisher(PoseStamped, 'vggt_mapper/pose', 1)
        self.camera_info_publisher = self.create_publisher(CameraInfo, 'vggt_mapper/camera_info', 1)

        # Subscribers
        if self.use_external_poses:
            # Use message filters for synchronized callbacks
            if self.use_sim:                
                self.image_sub = Subscriber(self, Image, self.image_topic)
                self.pose_sub = Subscriber(self, PoseStamped, self.pose_topic)
            else:
                if 'compressed' in self.image_topic:
                    self.image_sub = Subscriber(self, CompressedImage, self.image_topic)
                else:
                    self.image_sub = Subscriber(self, Image, self.image_topic)
                self.pose_sub = Subscriber(self, Odometry, self.pose_topic)
            
            # Approximate time synchronizer
            self.sync = ApproximateTimeSynchronizer(
                [self.image_sub, self.pose_sub],
                queue_size=500,
                slop=self.pose_sync_tolerance
            )
            self.sync.registerCallback(self.synchronized_callback)
            
            self.get_logger().info("Using synchronized image+pose callback")
        else:
            # Use regular image-only subscription
            self.image_subscription = self.create_subscription(
                Image,
                self.image_topic,
                self.image_callback
            )
            self.get_logger().info("Using image-only callback")
                        
        # Processing thread
        self.processing_thread = threading.Thread(target=self.processing_worker, daemon=True)
        self.processing_thread.start()

        # Buffer for external poses
        self.pose_buffer = []  
        # First external pose (world->camera) reference for relative transforms
        self.T_0_w = None

        self.scale = None
        self.last_pos = None
        
        self.get_logger().info("VGGT-SLAM node initialized successfully")

        self.first_batch = True
        self.first_batch_pub = True

        # Track external pose of each submap
        self.all_submap_poses = []  # list of 4x4 numpy matrices

        self.received_first_synced = False
        
    @torch.no_grad()
    def synchronized_callback(self, image_msg, pose_msg):
        """Synchronized callback for image and pose messages"""
        # Rate limiting - only process synchronized data at 2Hz

        if self.received_first_synced == False:
            self.get_logger().info("Received synced rgb/pose")
            self.received_first_synced = True

        current_time = self.get_clock().now().nanoseconds / 1e9
        if current_time - self.last_sync_time < self.sync_callback_rate:
            return
        self.last_sync_time = current_time

        if 'compressed' in self.image_topic:
            np_arr = np.frombuffer(image_msg.data, dtype=np.uint8)
            raw = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            if raw.ndim == 2:
                cv_image = cv2.cvtColor(raw, cv2.COLOR_BayerRG2RGB)
            else:
                cv_image = raw
        else:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        
        # Check disparity directly in callback if optical flow is enabled
        should_process = True
        
        # determine should_process if more than 2m translation from last processed frame
        if self.last_pos is not None:
            if self.use_sim:
                curr_pos = np.array([pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z])
            else:
                curr_pos = np.array([pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y, pose_msg.pose.pose.position.z])
            dist = np.linalg.norm(curr_pos - self.last_pos)
            if dist < self.image_distance:
                should_process = False
            else:
                self.last_pos = curr_pos
        else:
            if self.use_sim:
                self.last_pos = np.array([pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z])
            else:
                self.last_pos = np.array([pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y, pose_msg.pose.pose.position.z])

        if should_process:
            # Save image to temporary file with timestamp
            image_filename = os.path.join(self.temp_dir, f"frame_{self.image_counter:06d}.jpg")
            cv2.imwrite(image_filename, cv_image)
            
            # Store pose data (for future use)
            if self.use_sim:
                # Pose msg
                pose_data = {
                    'position': [pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z],
                    'orientation': [pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, 
                                    pose_msg.pose.orientation.z, pose_msg.pose.orientation.w],
                    'frame_id': pose_msg.header.frame_id,
                    'timestamp': pose_msg.header.stamp
                }
            else:
                # Odometry msg
                pose_data = {
                    'position': [pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y, pose_msg.pose.pose.position.z],
                    'orientation': [pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, 
                                    pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w],
                    'frame_id': pose_msg.header.frame_id,
                    'timestamp': pose_msg.header.stamp
                }
            # Compute homogeneous transform matrix for external pose
            quat = pose_data['orientation']  # [x, y, z, w]
            R_mat = _R.from_quat(quat).as_matrix()
            t_vec = np.array(pose_data['position'], dtype=float)
            # self.get_logger().info(f"position: {t_vec}")
            # R_euler = _R.from_matrix(R_mat).as_euler('xyz', degrees=True)
            # self.get_logger().info(f"rotation (euler): {R_euler}")
            
            T_w_body = np.eye(4, dtype=float)
            T_w_body[:3, :3] = R_mat
            T_w_body[:3, 3] = t_vec

            if self.use_sim:
                # cam to body (FLU)
                R_body_cam = np.array([[0, -1, 0],
                                       [-1, 0, 0],
                                       [0, 0, -1]], dtype=float)
            else:
                # cam to body (FLU)
                R_body_cam = np.array([[0, -1, 0],
                                       [-1, 0, 0],
                                       [0, 0, -1]], dtype=float)
                # ENU to FLU
                R_flu_enu = np.array([[0, 1, 0],
                                      [-1, 0, 0],
                                      [0, 0, 1]], dtype=float)
                T_flu_enu = np.eye(4, dtype=float)
                T_flu_enu[:3, :3] = R_flu_enu
                # convert ENU frame to FLU frame
                T_enu_body = T_w_body
                T_flu_body = T_flu_enu @ T_enu_body @ np.linalg.inv(T_flu_enu)
                T_w_body = T_flu_body
            self.T_body_cam = np.eye(4, dtype=float)
            self.T_body_cam[:3, :3] = R_body_cam # T_x_cam
            self.T_cam_body = np.linalg.inv(self.T_body_cam)
            T_w_cam = T_w_body @ self.T_body_cam # T_w_cam = T_w_x @ T_x_cam

            # Initialize base external pose for relative transforms
            if self.T_0_w is None:
                self.T_w_0 = T_w_cam
                self.T_0_w = np.linalg.inv(T_w_cam)
                T_0_cam = np.eye(4, dtype=float)
            else:
                # Compute relative transform from base external pose
                T_0_cam = self.T_0_w @ T_w_cam # T_0_cam = T_0_w @ T_w_cam
            
            # Use relative transform as pose
            pose_data['matrix'] = T_0_cam
            # Update position to relative translation
            pose_data['position'] = T_0_cam[:3, 3].tolist()
            
            # Add to processing queue with pose data
            if not self.image_queue.full():
                self.image_queue.put((image_filename, image_msg.header.stamp, pose_data))
                self.image_counter += 1
                
                self.get_logger().debug(f"Synchronized frame {self.image_counter}: "
                                        f"img_time={image_msg.header.stamp.sec}.{image_msg.header.stamp.nanosec}, "
                                        f"pose_time={pose_msg.header.stamp.sec}.{pose_msg.header.stamp.nanosec}")
            else:
                self.get_logger().warn("Image queue is full, dropping synchronized frame")
    
    def image_callback(self, msg):
        """Callback for incoming images (used when external poses are disabled)"""
        try:
            # Rate limiting - only process images at 2Hz
            current_time = self.get_clock().now().nanoseconds / 1e9
            if current_time - self.last_sync_time < self.sync_callback_rate:
                return
            self.last_sync_time = current_time
            
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Check disparity directly in callback if optical flow is enabled
            should_process = True
            if should_process:
                # Save image to temporary file with timestamp
                image_filename = os.path.join(self.temp_dir, f"frame_{self.image_counter:06d}.jpg")
                cv2.imwrite(image_filename, cv_image)
                
                # Add to processing queue without pose data
                if not self.image_queue.full():
                    self.image_queue.put((image_filename, msg.header.stamp, None))  # None for pose data
                    self.image_counter += 1
                else:
                    self.get_logger().warn("Image queue is full, dropping frame")
                    
        except Exception as e:
            self.get_logger().error(f"Error in image callback: {str(e)}")
    
    def processing_worker(self):
        """Background thread for processing images through VGGT-SLAM"""
        while True:
            try:
                # Get image from queue (blocking)
                data = self.image_queue.get(timeout=1.0)
                
                if len(data) == 2:
                    image_filename, timestamp = data
                    pose_data = None
                else:
                    image_filename, timestamp, pose_data = data
                
                with self.processing_lock:
                    self.process_image(image_filename, timestamp, pose_data)
                    
            except Empty:
                continue
            # except Exception as e:
            #     self.get_logger().error(f"Error in processing worker: {str(e)}")
    
    def process_image(self, image_filename, timestamp, pose_data=None):
        """Add image and pose to buffer"""
        # Track incoming external poses per image
        if self.use_external_poses:
            self.pose_buffer.append(pose_data)
        self.image_names_subset.append(image_filename)

        if self.first_batch and len(self.image_names_subset) < 10:
            return
        else:
            self.first_batch = False
        
        # Process submap if we have enough images
        if len(self.image_names_subset) >= self.submap_size + self.overlapping_window_size:
        # if len(self.image_names_subset) == self.submap_size:
            if len(self.image_names_subset) > 0:
                self.get_logger().info(f"Processing submap with {len(self.image_names_subset)} images")
                # Determine external loop candidates by pose distance & temporal separation
                if self.enable_loop_closure and self.use_external_poses and self.pose_buffer:
                    current_pose = np.array(self.pose_buffer[0]['matrix'])
                    new_id = len(self.all_submap_poses)
                    cands = []
                    for prev_id, prev_pose in enumerate(self.all_submap_poses):
                        dist = np.linalg.norm(current_pose[:3,3] - prev_pose[:3,3])
                        if dist <= self.loop_distance and abs(new_id - prev_id) >= self.loop_id_threshold:
                            cands.append(prev_id)
                    external_loop_candidates = cands
                    self.get_logger().info(f"Found {len(external_loop_candidates) if external_loop_candidates else 0} external loop candidates")
                else:
                    external_loop_candidates = None                    
                
                # Run VGGT-SLAM predictions
                predictions = self.solver.run_predictions(
                    self.image_names_subset, self.model, self.max_loops, external_loop_candidates
                )             
                # If using external poses, run scale estimation before adding to map
                if self.use_external_poses and self.pose_buffer:
                    # Build world->camera priors for local batch
                    H_cam2world_global = [pd['matrix'] for pd in self.pose_buffer]
                    H0 = H_cam2world_global[0]
                    H_cam2world_priors = [np.linalg.inv(H0) @ H for H in H_cam2world_global]
                    H_world2cam_priors = [np.linalg.inv(H) for H in H_cam2world_priors]
                    # compute scale
                    predictions = estimate_scale(predictions, H_world2cam_priors, self.scale)
                    self.scale = predictions['scale']

                    # print out prior translations in c2w
                    H = H_cam2world_priors[-1]
                    t = H[:3, 3]
                    # self.get_logger().info(f"Prior translation: {t.tolist()}")
                    H = predictions['scaled_extrinsics'][-1]
                    H = np.linalg.inv(H)
                    t = H[:3, 3]
                    # self.get_logger().info(f"Scaled translation: {t.tolist()}")
                # Add points to map
                self.solver.add_points(predictions)                
                submap_id = self.solver.map.get_largest_key()
                # Anchor submap into metric scale using external pose
                if self.use_external_poses and self.pose_buffer:
                    # add GPS position prior
                    t = self.pose_buffer[0]['position']
                    R = self.pose_buffer[0]['orientation']
                    H_t = np.eye(4, dtype=float)
                    H_t[:3, 3] = np.array(t, dtype=float)
                    H_t[:3, :3] = _R.from_quat(R).as_matrix()
                    if self.use_sim3:
                        anchor_noise = [1] * 6
                        anchor_noise[3] = 3
                        anchor_noise[4] = 3
                        anchor_noise[5] = 3
                    else:
                        anchor_noise = [1e-1] * 15
                        anchor_noise[3] = 1e-9
                        anchor_noise[6] = 1e-9
                        anchor_noise[9] = 1e-9
                    if self.enable_gps_prior:
                        self.solver.graph.add_prior_factor(submap_id, H_t, noiseModel.Diagonal.Sigmas(anchor_noise))
                    # Save external translation for later comparison
                    self.last_ext_t = np.array(t, dtype=float)
                    # self.get_logger().info(f"Added translation-only prior for submap {submap_id}: t={t}")
                # Optimize graph
                self.solver.graph.optimize()
                # After optimize, compare optimized homography translation to external prior
                H_opt = self.solver.graph.get_homography(submap_id).matrix()
                opt_t = H_opt[:3, 3]
                # self.get_logger().info(
                #     f"Compare submap {submap_id}: external_t={self.last_ext_t.tolist()}, optimized_t={opt_t.tolist()}"
                # )
                self.solver.map.update_submap_homographies(self.solver.graph)

                self.publish_global_pointcloud()

                self.publish_optimized_submap(self.solver.map.get_latest_submap(), predictions)

                self.get_logger().info(f"Total submaps: {self.solver.map.get_num_submaps()}")
                # self.get_logger().info(f"Total loop closures: {self.solver.graph.get_num_loops()}")
                # Reset for next submap, keeping overlapping frames
                self.image_names_subset = self.image_names_subset[-self.overlapping_window_size:]
                if self.use_external_poses:
                    # Record this submap's external pose
                    if self.pose_buffer:
                        self.all_submap_poses.append(np.array(self.pose_buffer[0]['matrix']))
                    self.pose_buffer = self.pose_buffer[-self.overlapping_window_size:]                    
        
    def publish_optimized_submap(self, submap, predictions):

        if submap is None:
            self.get_logger().warn("No submap to publish")
            return
        # get rgb, depth, poses
        intrinsics = predictions['intrinsic']
        rgb_images = predictions['images']
        depth_images = predictions['scaled_depth'] * 1000.0 # convert to mm
        rel_poses_w2c = predictions['scaled_extrinsics'] # w2c
        submap_anchor = submap.get_reference_homography() # c2w T_0_submap

        rel_poses_c2w = [np.linalg.inv(p) for p in rel_poses_w2c] # T_submap_cam

        # convert to cam0 frame
        T_0_cam = [submap_anchor @ p for p in rel_poses_c2w] # T_0_cam = T_0_submap @ T_submap_cam

        # convert to world frame
        T_w_cam = [self.T_w_0 @ p for p in T_0_cam] # T_w_cam = T_w_0 @ T_0_cam
        T_w_body = [p @ self.T_cam_body for p in T_w_cam] # T_w_body = T_w_cam @ T_cam_body
        
        skip_overlapping_frames = True
        if self.first_batch_pub:
            skip_overlapping_frames = False
            self.first_batch_pub = False
        
        for i in range(len(rgb_images)):
            if skip_overlapping_frames:
                if i < self.overlapping_window_size:
                    self.get_logger().info(f"Skipping overlapping frame {i+1}/{len(rgb_images)}")
                    continue
            timestamp = self.get_clock().now().to_msg()
            rgb = rgb_images[i].transpose(1, 2, 0)
            rgb = (rgb * 255.0).astype(np.uint8)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            rgb_msg = self.bridge.cv2_to_imgmsg(rgb, encoding='bgr8')
            rgb_msg.header.stamp = timestamp
            rgb_msg.header.frame_id = 'map'

            depth_msg = self.bridge.cv2_to_imgmsg(depth_images[i], encoding='passthrough')
            depth_msg.header.stamp = timestamp
            depth_msg.header.frame_id = 'map'

            pose_msg = PoseStamped()
            pose_msg.header.stamp = timestamp
            pose_msg.header.frame_id = 'map'
            pose_msg.pose.position.x = T_w_body[i][0, 3]
            pose_msg.pose.position.y = T_w_body[i][1, 3]
            pose_msg.pose.position.z = T_w_body[i][2, 3]
            quat = _R.from_matrix(T_w_body[i][:3, :3]).as_quat()
            pose_msg.pose.orientation.x = quat[0]
            pose_msg.pose.orientation.y = quat[1]
            pose_msg.pose.orientation.z = quat[2]
            pose_msg.pose.orientation.w = quat[3]

            intrinsics_msg = CameraInfo()
            intrinsics_msg.header.stamp = timestamp
            intrinsics_msg.header.frame_id = 'map'
            intrinsics_msg.width = rgb.shape[1]
            intrinsics_msg.height = rgb.shape[0]
            intrinsics_msg.k = intrinsics[i].flatten().tolist()            

            # publish
            self.rgb_publisher.publish(rgb_msg)
            self.depth_publisher.publish(depth_msg)
            self.pose_publisher.publish(pose_msg)
            self.camera_info_publisher.publish(intrinsics_msg)

            # sleep
            time.sleep(0.1)
    
    def publish_global_pointcloud(self, latest_only=True):
        """Publish the combined point cloud of all submaps as a single MarkerArray"""
        # Gather all points and colors from every submap
        pts_list = []
        cols_list = []

        if latest_only:
            submap = self.solver.map.get_latest_submap()
            if submap is None:
                return
            pts = submap.get_points_in_world_frame()
            cols = submap.get_points_colors()
            if pts is not None and pts.size > 0:
                pts_list.append(pts.reshape(-1, 3))
                cols_list.append(cols.reshape(-1, 3))
        else:
            for submap in self.solver.map.get_submaps():
                pts = submap.get_points_in_world_frame()
                cols = submap.get_points_colors()
                if pts is None or pts.size == 0:
                    continue
                pts_list.append(pts.reshape(-1, 3))
                cols_list.append(cols.reshape(-1, 3))

        if not pts_list:
            return
        pts_all = np.vstack(pts_list)
        cols_all = np.vstack(cols_list)

        # convert pts_all back to body frame
        pts_world = self.T_w_0 @ np.hstack((pts_all, np.ones((pts_all.shape[0], 1)))).T
        pts_all = pts_world[:3, :].T  # Convert back to (N, 3) shape
        pts_all = pts_all.astype(np.float32)

        # voxelize to ptcloud_viz_resolution
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_all)
        pcd.colors = o3d.utility.Vector3dVector(cols_all / 255.0)
        pcd = pcd.voxel_down_sample(voxel_size=self.ptcloud_viz_resolution)
        pts_all = np.asarray(pcd.points)
        cols_all = np.asarray(pcd.colors) * 255.0  # Convert back
        cols_all = cols_all.astype(np.uint8)          

        # Build PointCloud2 message
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        data = []
        for (x, y, z), (r, g, b) in zip(pts_all, cols_all):
            rgb_uint = (int(r) << 16) | (int(g) << 8) | int(b)
            data.append((float(x), float(y), float(z), rgb_uint))
        cloud = pc2.create_cloud(header, fields, data)
        self.pointcloud_publisher.publish(cloud)
        self.get_logger().info(f"Published global pointcloud with {pts_all.shape[0]} points")
    
    def destroy_node(self):
        """Clean up when node is destroyed"""
        # Clean up temporary files
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = VGGTSlamNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in main: {str(e)}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

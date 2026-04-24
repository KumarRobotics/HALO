"""
    Yuezhan Tao
    July 2025

    occ_map class for ros2 node
"""

import rclpy
import numpy as np
import cv2
import struct
import copy
import math
import threading
from enum import Enum, auto
import os
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped, Pose, PointStamped
from std_msgs.msg import Bool, Header
from nav_msgs.msg import Odometry
from air_sem_gridmap_interfaces.msg import RelevancyMap
from air_sem_explorer_interfaces.msg import Frontiers
from air_sem_explorer.planner.exploration_planner import ExplorationPlanner
from air_sem_explorer.utils.utils import quat_msg_to_rot_mat, load_config
from air_sem_explorer.planner.path_tracker import ExplorationPath, PathTracker
from ament_index_python.packages import get_package_share_directory
from air_sem_explorer.utils.evaluation_logger import EvaluationLogger
from air_sem_explorer.utils.utils import get_area_discovered
from tf2_ros import Buffer, TransformListener

class ExplorationState(Enum):
    INIT = auto()
    WAIT_TRIGGER = auto()
    FINISH = auto() # Exploration finished, waiting for next trigger or retasking
    PLAN = auto()
    EXEC = auto()
    EXEC_INIT_WP = auto()  # Executing initial waypoints for robot initialization

class PlanStatus(Enum):
    REPLAN_GLOBAL = auto()  # Replan globally
    REPLAN_LOCAL = auto()   # Replan locally
    EXECUTING = auto()      # Executing the current plan

class StateMachine(Node):
    def __init__(self):
        super().__init__('state_machine')
        self.state_ = ExplorationState.INIT
        self.plan_state_ = PlanStatus.REPLAN_GLOBAL
        self.lock_state_ = threading.Lock()
        self.lock_msg_ = threading.Lock()
        self.flag_retrieval_ = False # internal flag to indicate if we consider retrieval, sampling waypoints from the map
        
        self.last_map_ = None  # Last received relevancy map
        self.last_frontiers_ = None # Last received frontiers

        # For common param, load through yaml
        config_file = os.path.join(get_package_share_directory('air_sem_explorer'), 'config', 'map_common.yaml')
        map_config = load_config(config_file)

        # Load common map param:
        self.map_min_x_ = map_config['map_min_x']
        self.map_max_x_ = map_config['map_max_x']
        self.map_min_y_ = map_config['map_min_y']
        self.map_max_y_ = map_config['map_max_y']
        self.resolution_ = map_config['resolution']
        self.use_odom_msg_ = map_config['use_odom_msg']

        # ROS2 params, subs and pubs
        self.declare_parameter("relevancy_map_topic", "/relevancy_map")
        self.declare_parameter("frontier_topic", "/frontiers")
        self.declare_parameter('odom_topic', '/pose')
        self.declare_parameter('region_size', 10.0)
        self.declare_parameter('global_replan_thres', 5.0)
        self.declare_parameter('local_replan_thres', 2.0)
        self.declare_parameter('global_relevancy_thres', 0.4)
        self.declare_parameter('retrieval_thres', 0.8)
        self.declare_parameter('ftr_relevancy_thres', 0.2)
        self.declare_parameter('goal_x', [0.0])
        self.declare_parameter('goal_y', [0.0])
        self.declare_parameter('goal_tol', [10.0])
        self.declare_parameter('enable_goal', False)
        self.declare_parameter('log_path', " ")
        self.declare_parameter('sim', True)
        # Used for real robot, initialization of scale estimation and IMU online calibration
        self.declare_parameter('robot_initialization', False)
        self.declare_parameter('init_waypoints_x', 1.0)
        self.declare_parameter('init_waypoints_y', 1.0)
        self.declare_parameter('init_timeout', 1.0)

        # Get map info
        self.region_size_ = self.get_parameter('region_size').value
        self.global_replan_thres_ = self.get_parameter('global_replan_thres').value
        self.local_replan_thres_ = self.get_parameter('local_replan_thres').value
        self.global_relevancy_thres_ = self.get_parameter('global_relevancy_thres').value
        self.ftr_relevancy_thres_ = self.get_parameter('ftr_relevancy_thres').value
        self.retrieval_thres_ = self.get_parameter('retrieval_thres').value
        self.goal_x_ = self.get_parameter('goal_x').value
        self.goal_y_ = self.get_parameter('goal_y').value
        self.goal_tol_ = self.get_parameter('goal_tol').value
        self.enable_goal_ = self.get_parameter('enable_goal').value
        self.log_path_ = self.get_parameter('log_path').value
        self.last_global_plan_t_ = self.get_clock().now()
        self.last_local_plan_t_ = self.get_clock().now()
        self.sim_ = self.get_parameter('sim').value

        self.robot_initialization_ = self.get_parameter('robot_initialization').value
        self.init_waypoints_x_ = self.get_parameter('init_waypoints_x').value
        self.init_waypoints_y_ = self.get_parameter('init_waypoints_y').value
        self.init_timeout_ = self.get_parameter('init_timeout').value
        self.init_finished_ = False
        self.init_wp_set_ = False
        self.init_wp_published_ = False

        self.num_goals_ = len(self.goal_x_)
        self.current_goal_ = 0

        self.odom_world_ = None
        self.odom_pos_ = None
        self.path_ = None
        self.next_region_ = None
        self.next_region_retrieval_ = False  # Whether the next region is for retrieval

        ############ Map info setup ############
        self.map_size_ = np.array([self.map_max_x_ - self.map_min_x_,
                                    self.map_max_y_ - self.map_min_y_], dtype=np.float32)
        self.map_origin_ = np.array([self.map_min_x_, self.map_min_y_], dtype=np.float32)
        self.map_voxel_num_ = np.array(
            [int(self.map_size_[0] / self.resolution_), int(self.map_size_[1] / self.resolution_)],
            dtype=np.int32
        )

        # Init the exploration_planner
        self.explorer_ = ExplorationPlanner(
            logger=self.get_logger(),
            map_origin=self.map_origin_,
            map_size=self.map_size_,
            resolution=self.resolution_,
            map_voxel_num=self.map_voxel_num_,
            region_size=self.region_size_,
            global_relevancy_thres=self.global_relevancy_thres_,
            retrieval_thres=self.retrieval_thres_,
            ftr_relevancy_thres=self.ftr_relevancy_thres_
            
        )

        self.total_distance_ = 0.0
        self.last_odom_pos_ = None
        self.task_id_ = None
        # save region segment for visualization
        self.region_segs_ = self.explorer_.get_region_segments()

        # DEBUG print:
        self.get_logger().info(f"Map info: min_x={self.map_min_x_}, max_x={self.map_max_x_}, "
                                f"min_y={self.map_min_y_}, max_y={self.map_max_y_}, "
                                f"resolution={self.resolution_}, region_size={self.region_size_}, "
                                f"global_replan_thres={self.global_replan_thres_}, "
                                f"local_replan_thres={self.local_replan_thres_}, "
                                f"use_odom_msg={self.use_odom_msg_}, "
                                f"global_relevancy_thres={self.global_relevancy_thres_}, "
                                f"retrieval_thres={self.retrieval_thres_}, "
                                f"ftr_relevancy_thres={self.ftr_relevancy_thres_}, "
                                f"enable_goal={self.enable_goal_}, "
                                f"map_voxel_num={self.map_voxel_num_}, "
                                f"map_origin={self.map_origin_}, "
                                f"map_size={self.map_size_},"
                                f"running sim={self.sim_}, "
                                # f"gps_wp_topic={self.gps_wp_topic_}"
                                f"robot_initialization={self.robot_initialization_},"
                                f"init_waypoints=({self.init_waypoints_x_}, {self.init_waypoints_y_}),"
                                f"init_timeout={self.init_timeout_}"
        )
        if self.log_path_ == " ":
            self.enable_log_ = False
            self.get_logger().warn("No log path provided, logging disabled.")
        else:
            self.enable_log_ = True
            self.get_logger().info(f"Logging enabled, log path: {self.log_path_}")
            self.eval_log_ = EvaluationLogger(f"{self.log_path_}/ours_planner", "Polycity_hardrel")
            self.eval_log_.log(f"Config: global_replan_thres: {self.global_replan_thres_}")
            self.eval_log_.log(f"Config: local_replan_thres: {self.local_replan_thres_}")
            self.eval_log_.log(f"Config: retrieval_thres: {self.retrieval_thres_}")
            self.eval_log_.log(f"Config: global_relevancy_thres: {self.global_relevancy_thres_}")
            self.eval_log_.log(f"Config: ftr_relevancy_thres: {self.ftr_relevancy_thres_}")
            self.eval_log_.log(f"Config: region size: {self.region_size_}")
            self.eval_log_.log(f"Goal location: ({self.goal_x_[self.current_goal_]}, {self.goal_y_[self.current_goal_]})")

        self.tracker_ = PathTracker(self, distance_threshold=5.0, time_threshold=10.0, limit_time=False)
        ############ Callback groups ############
        odom_group = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()
        statemachine_group = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()
        map_sub_group = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()
        ftr_sub_group = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()
        trigger_group = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()

        self.rel_map_topic = self.get_parameter("relevancy_map_topic").value
        self.frontier_topic = self.get_parameter("frontier_topic").value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.relevancy_map_sub_ = self.create_subscription(
            RelevancyMap, self.rel_map_topic, self.relevancy_map_callback,
            10, callback_group=map_sub_group
        )
        self.frontier_sub_ = self.create_subscription(
            Frontiers, self.frontier_topic, self.frontier_callback, 
            10, callback_group=ftr_sub_group
        )
        if self.use_odom_msg_:
            # Use odometry message for position updates
            self.odom_sub_ = self.create_subscription(
                Odometry, self.odom_topic, self.odom_callback,
                10, callback_group=odom_group
            )
        else:
            # Use pose message for position updates
            self.odom_sub_ = self.create_subscription(
                PoseStamped, self.odom_topic, self.odom_callback,
                10, callback_group=odom_group
            )
        self.sm_timer_ = self.create_timer(0.5, self.sm_callback, callback_group=statemachine_group)
        self.trigger_sub_ = self.create_subscription(
            Bool, '/state_machine/trigger', self.trigger_callback,
            2, callback_group=trigger_group
        )


        # Publishers
        self.path_pub_ = self.create_publisher(MarkerArray, '/planner/exploration_path', 10)
        self.wp_pub_ = self.create_publisher(PoseStamped, '/planner/waypoint', 10)
        self.region_pub_ = self.create_publisher(MarkerArray, '/planner/regions', 10)
        self.goal_pub_ = self.create_publisher(Marker, '/planner/goal', 10)
        self.get_logger().info("State Machine Node Initialized")

        # TF listener for map→world transform
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def transit_state(self, new_state: ExplorationState):
        with self.lock_state_:
            if self.state_ != new_state:
                self.get_logger().info(f"Transitioning State Mchine from {self.state_.name} to {new_state.name}")
                self.state_ = new_state
    

    def set_plan_state(self, new_plan_state: PlanStatus):
        if self.plan_state_ != new_plan_state:
            self.get_logger().info(f"Transitioning Plan State from {self.plan_state_.name} to {new_plan_state.name}")
            self.plan_state_ = new_plan_state


    def sm_callback(self):
        if self.state_ == ExplorationState.INIT:
            self.get_logger().info("State Machine Initialized")
            self.transit_state(ExplorationState.WAIT_TRIGGER)
        
        elif self.state_ == ExplorationState.WAIT_TRIGGER:
            pass # Waiting for trigger through trigger callback
        
        elif self.state_ == ExplorationState.FINISH:
            # Pub current position as waypoint
            waypoint = PoseStamped()
            waypoint.header.stamp = self.get_clock().now().to_msg()
            waypoint.header.frame_id = 'map'
            waypoint.pose.position.x = float(self.odom_pos_[0])
            waypoint.pose.position.y = float(self.odom_pos_[1])
            waypoint.pose.position.z = float(self.odom_pos_[2])
            waypoint.pose.orientation.w = 1.0
            self.wp_pub_.publish(waypoint)
            self.get_logger().info("Task Finished, waiting for next trigger or retasking")
        
        elif self.state_ == ExplorationState.PLAN:
            # if global plan, update planning graph, call global + local.
            # if retrieval, pass arg to global planner. 
            if self.plan_state_ == PlanStatus.REPLAN_GLOBAL:
                # Update planning graph, call global planner
                if self.odom_pos_ is None:
                    self.get_logger().warn("No odometry position available, cannot replan globally.")
                    return
                if self.last_frontiers_ is None or self.last_map_ is None:
                    self.get_logger().warn("No frontiers or map available, cannot replan globally.")
                    return
                self.get_logger().info("Global Replan.")
                # with self.lock_msg_:
                self.next_region_, self.next_region_retrieval_ = self.explorer_.global_planner(self.odom_pos_[:2], self.last_map_,
                                                                                                self.last_frontiers_, self.flag_retrieval_)
                # Here, assume the global planner should have flipped 
                # the flags for regions need retrieval, so we can reset the flag
                if self.flag_retrieval_:
                    self.flag_retrieval_ = False
                    self.get_logger().info("Retrieval flag reset.")
                if self.next_region_ is None:
                    self.get_logger().warn("No valid region found, transit to FINISH.")
                    self.transit_state(ExplorationState.FINISH)
                    return
                rclpy.logging.get_logger('StateMachine').info(f"Next region: {self.next_region_}, Retrieval: {self.next_region_retrieval_}")
                path = self.explorer_.local_planner_ortools(self.odom_pos_[:2], self.last_map_, self.last_frontiers_, self.next_region_, self.next_region_retrieval_)
                if path is None or len(path) == 0:
                    self.get_logger().warn("No valid path found, transit to global plan.")
                    self.set_plan_state(PlanStatus.REPLAN_GLOBAL)
                    return
                # Set path and tracker
                rclpy.logging.get_logger('StateMachine').info(f"Path: {path}")
                self.path_ = ExplorationPath(path, self.get_clock().now())
                self.tracker_.set_path(self.path_)
                self.visualize_plan(path)
                self.publish_segment_boundaries(self.region_segs_)
                self.last_global_plan_t_ = self.get_clock().now()
                self.last_local_plan_t_ = self.get_clock().now()
                self.set_plan_state(PlanStatus.EXECUTING)
                self.transit_state(ExplorationState.EXEC)
            # if local plan, update planning graph, call local planner
            elif self.plan_state_ == PlanStatus.REPLAN_LOCAL:
                # Update planning graph, call local planner
                self.get_logger().info("Local Replan.")
                # with self.lock_msg_:
                path = self.explorer_.local_planner_ortools(self.odom_pos_[:2], self.last_map_, self.last_frontiers_, self.next_region_, self.next_region_retrieval_)
                if path is None or len(path) == 0:
                    self.get_logger().warn("No valid local path found, transit to global plan.")
                    self.set_plan_state(PlanStatus.REPLAN_GLOBAL)
                    return
                # Set path and tracker
                self.path_ = ExplorationPath(path, self.get_clock().now())
                self.tracker_.set_path(self.path_)
                self.visualize_plan(path)
                self.last_local_plan_t_ = self.get_clock().now()
                self.set_plan_state(PlanStatus.EXECUTING)
                self.transit_state(ExplorationState.EXEC)
            elif self.plan_state_ == PlanStatus.EXECUTING:
                # Shouldn't happen
                self.get_logger().error("Executing state should not be in PLAN state.")
                self.transit_state(ExplorationState.EXEC)

        elif self.state_ == ExplorationState.EXEC:
            # execute plan
            # Reset region retrieval flag if needed
            self.explorer_.check_region_flag(self.odom_pos_[:2])
            t_now = self.get_clock().now()
            if (t_now - self.last_global_plan_t_).nanoseconds / 1e9 > self.global_replan_thres_:
                # Replan globally if the last global plan is too old
                self.get_logger().info("Replanning globally.")
                self.set_plan_state(PlanStatus.REPLAN_GLOBAL)
                self.transit_state(ExplorationState.PLAN)
            elif (t_now - self.last_local_plan_t_).nanoseconds / 1e9 > self.local_replan_thres_:
                self.get_logger().info("Replanning locally.")
                self.set_plan_state(PlanStatus.REPLAN_LOCAL)
                self.transit_state(ExplorationState.PLAN)
            else:
                # call tracker
                self.tracker_path()

        elif self.state_ == ExplorationState.EXEC_INIT_WP:
            # Execute initial waypoints for robot initialization
            if self.odom_pos_ is None:
                self.get_logger().warn("[StateMachine: EXEC_INIT_WP] No odometry position available.")
                return
            if not self.init_wp_set_:
                init_wps = self.generate_init_wp(self.odom_pos_[:2])
                # Set path and tracker
                rclpy.logging.get_logger('StateMachine').info(f"Initial waypoints: {init_wps}")
                self.path_ = ExplorationPath(init_wps, self.get_clock().now())
                self.tracker_.set_path(self.path_)
                self.visualize_plan(init_wps)
                self.publish_segment_boundaries(self.region_segs_)
                self.init_wp_set_ = True
                self.init_wp_start_time_ = self.get_clock().now()
            if not self.init_finished_:
                elapsed_time = (self.get_clock().now() - self.init_wp_start_time_).nanoseconds / 1e9
                if elapsed_time > self.init_timeout_:
                    self.get_logger().info("Initial waypoint execution timeout.")
                    self.init_finished_ = True
                    self.set_plan_state(PlanStatus.REPLAN_GLOBAL)
                    self.transit_state(ExplorationState.PLAN)
                else:
                    self.get_logger().info(f"Tracking init waypoint...")
                    if not self.init_wp_published_:
                        # call tracker
                        self.track_init_path()
                        self.init_wp_published_ = True
            else:
                # Shouldn't happen
                self.get_logger().warn("Initial waypoint execution already finished, transitioning to PLAN state.")
                self.set_plan_state(PlanStatus.REPLAN_GLOBAL)
                self.transit_state(ExplorationState.PLAN)               
                    
        else:
            self.get_logger().error(f"Unknown state: {self.state_.name}")

    def tracker_path(self):
        self.get_logger().info("Executing current plan.")
        ret = self.tracker_.update(self.odom_pos_)
        if ret['status'] == 'completed' or ret['status'] == 'no_path':
            self.get_logger().info("Path complete / No path!")
            self.set_plan_state(PlanStatus.REPLAN_GLOBAL)
            self.transit_state(ExplorationState.PLAN)
        if ret['goal_updated']:
            self.get_logger().info(f"Track new waypoint: {ret['current_waypoint']}")
        # Use the current waypoint for navigation
        if ret['current_waypoint'] is not None:
            # If sim, pub to sim tracker
            # if self.sim_:
            waypoint = PoseStamped()
            waypoint.header.stamp = self.get_clock().now().to_msg()
            waypoint.header.frame_id = 'map'
            waypoint.pose.position.x = float(ret['current_waypoint'][0])
            waypoint.pose.position.y = float(ret['current_waypoint'][1])
            waypoint.pose.position.z = float(self.odom_pos_[2])
            waypoint.pose.orientation.w = 1.0
            self.wp_pub_.publish(waypoint)

    def track_init_path(self):
        self.get_logger().info("Executing initialization path.")
        ret = self.tracker_.update(self.odom_pos_)
        if ret['status'] == 'completed' or ret['status'] == 'no_path':
            self.get_logger().info("Init path complete / No path!")
            self.init_finished_ = True
            self.set_plan_state(PlanStatus.REPLAN_GLOBAL)
            self.transit_state(ExplorationState.PLAN)
        if ret['goal_updated']:
            self.get_logger().info(f"Track new waypoint: {ret['current_waypoint']}")
        # Use the current waypoint for navigation
        if ret['current_waypoint'] is not None:
            # If sim, pub to sim tracker
            # if self.sim_:
            waypoint = PoseStamped()
            waypoint.header.stamp = self.get_clock().now().to_msg()
            waypoint.header.frame_id = 'map'
            waypoint.pose.position.x = float(ret['current_waypoint'][0])
            waypoint.pose.position.y = float(ret['current_waypoint'][1])
            waypoint.pose.position.z = float(self.odom_pos_[2])
            waypoint.pose.orientation.w = 1.0
            self.wp_pub_.publish(waypoint)



    def trigger_callback(self, msg):
        """
        Trigger the exploration.
        """
        if msg.data == True and self.state_ == ExplorationState.WAIT_TRIGGER:
            self.get_logger().info("Trigger received, starting exploration.")
            # log
            if self.enable_log_:
                if self.odom_pos_ is not None:
                    self.eval_log_.log(f"Exploration triggered. Start location: ({self.odom_pos_}).")
                else:
                    self.eval_log_.log("Exploration triggered.")
                self.mission_start_time_ = self.get_clock().now()
                self.eval_log_.log("---------------------------------------------------")
                self.eval_log_.log(f"Exploration started at: {self.mission_start_time_}")
            
            # If robot initialization, generate initial waypoints
            if self.robot_initialization_:
                self.set_plan_state(PlanStatus.EXECUTING)
                self.transit_state(ExplorationState.EXEC_INIT_WP)
            else:
                self.set_plan_state(PlanStatus.REPLAN_GLOBAL)
                self.transit_state(ExplorationState.PLAN)

    def retask_callback(self):
        """
        If retask, use the existing map to replan.
        """
        # force global replan and set retrieval
        self.flag_retrieval_ = True
        self.set_plan_state(PlanStatus.REPLAN_GLOBAL)
        self.transit_state(ExplorationState.PLAN)
        # Log the start time of the new task
        if self.enable_log_:
            self.mission_start_time_ = self.get_clock().now()
            self.eval_log_.log("---------------------------------------------------")
            self.eval_log_.log(f"New task started at: {self.mission_start_time_}")

    def relevancy_map_callback(self, map_msg):
        """
        Callback for relevancy map updates.
        """
        # with self.lock_msg_:
        self.last_map_ = map_msg
        # Now check if the task id changed
        if self.task_id_ is None:
            self.task_id_ = map_msg.task_id
            self.get_logger().info(f"Set initial task id to {self.task_id_}.")
        else:
            if self.task_id_ != map_msg.task_id:
                self.get_logger().info(f"Task id changed from {self.task_id_} to {map_msg.task_id}")
                self.task_id_ = map_msg.task_id
                if self.enable_log_:
                    self.eval_log_.log(f"Task id changed to {self.task_id_}.")
                self.retask_callback()
                # If we have more goals
                if self.current_goal_ < self.task_id_ and self.current_goal_ < self.num_goals_ - 1:
                    self.current_goal_ += 1
                    self.get_logger().info(f"Update current goal to {self.current_goal_}, location: ({self.goal_x_[self.current_goal_], self.goal_y_[self.current_goal_]})")
                    if self.enable_log_:
                        self.eval_log_.log(f"Update current goal to {self.current_goal_}, location: ({self.goal_x_[self.current_goal_], self.goal_y_[self.current_goal_]})")

    def frontier_callback(self, frontier_msg):
        """
        Callback for frontier updates.
        """
        # with self.lock_msg_:
        self.last_frontiers_ = frontier_msg


    def odom_callback(self, msg):
        """
        Callback for odometry updates.
        """
        if not self.use_odom_msg_:
            tmp_odom = msg.pose
        else:
            tmp_odom = msg.pose.pose
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
            self.total_distance_ += np.linalg.norm(self.odom_pos_ - self.last_odom_pos_)

        if self.state_ != ExplorationState.FINISH:
            self.terminate_callback()

    
    def visualize_plan(self, path):
        """
        Visualize path as both line strip and individual sphere points.
        """

        # delete all
        delete_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        delete_marker.ns = "exploration"
        delete_marker.id = 0
        delete_marker.action = Marker.DELETEALL
        delete_array.markers.append(delete_marker)
        self.path_pub_.publish(delete_array)

        marker_array = MarkerArray()
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"

        line_strip = Marker()
        line_strip.header = header
        line_strip.ns = "exploration"
        line_strip.id = 0
        line_strip.type = Marker.LINE_STRIP
        line_strip.action = Marker.ADD
        line_strip.pose.orientation.w = 1.0
        line_strip.scale.x = 0.2
        line_strip.color.a = 1.0
        line_strip.color.r = 0.0
        line_strip.color.g = 1.0
        line_strip.color.b = 0.0

        for i, point in enumerate(path):
            pt = Point()
            pt.x = float(point[0])
            pt.y = float(point[1])
            pt.z = float(self.odom_pos_[2])
            line_strip.points.append(pt)

            sphere = Marker()
            sphere.header = header
            sphere.ns = "exploration"
            sphere.id = i + 1
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position = pt
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.3
            sphere.scale.y = 0.3
            sphere.scale.z = 0.3
            sphere.color.a = 1.0

            # Color logic: green -> yellow -> red gradient
            sphere.color.r = min(1.0, i / max(1, len(path)-1))
            sphere.color.g = 1.0 - sphere.color.r
            sphere.color.b = 0.0

            marker_array.markers.append(sphere)

        marker_array.markers.append(line_strip)
        self.path_pub_.publish(marker_array)
    

    def publish_segment_boundaries(self, segments_dict, frame_id='map', z_height=0.5):
        """
        Publishes segment boundaries as line strips for visualization in RViz
        Args:
            segments_dict (dict): Dictionary of segments
            frame_id (str): Frame ID for the markers
            z_height (float): Height at which to draw the boundaries
        """
        marker_array = MarkerArray()
        
        for idx, (segment_id, segment_data) in enumerate(segments_dict.items()):
            # Extract position bounds
            pos_bounds = segment_data['pos_bounds']
            min_x, max_x = pos_bounds[0]
            min_y, max_y = pos_bounds[1]
            
            # Create a line strip marker for the boundary rectangle
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "segment_boundaries"
            marker.id = idx
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            
            # Set the scale (line width)
            marker.scale.x = 0.3  # Line width
            # Set color 
            marker.color.b = 1.0
            marker.color.a = 0.8
            
            # Create rectangle points (clockwise from bottom-left)
            points = [
                Point(x=float(min_x), y=float(min_y), z=float(z_height)),  # Bottom-left
                Point(x=float(max_x), y=float(min_y), z=float(z_height)),  # Bottom-right
                Point(x=float(max_x), y=float(max_y), z=float(z_height)),  # Top-right
                Point(x=float(min_x), y=float(max_y), z=float(z_height)),  # Top-left
                Point(x=float(min_x), y=float(min_y), z=float(z_height)),  # Back to bottom-left to close
            ]
            
            marker.points = points
            marker_array.markers.append(marker)
            
            # Optionally add text label at region center
            text_marker = self._create_text_marker(
                segment_data, 
                idx + len(segments_dict), 
                frame_id, 
                z_height
            )
            marker_array.markers.append(text_marker)
        
        # Clear old markers if we have fewer segments now
        self._clear_old_markers(marker_array, len(segments_dict) * 2)
        
        # Publish the marker array
        self.region_pub_.publish(marker_array)
        
        self.get_logger().info(f'Published {len(segments_dict)} segment boundaries')
    
    
    def _create_text_marker(self, segment_data, marker_id, frame_id, z_height):
        """Create a text marker showing the segment ID at the region center"""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "segment_labels"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        # Position at region center
        center = segment_data['region_center']
        marker.pose.position.x = center[0]
        marker.pose.position.y = center[1]
        marker.pose.position.z = z_height + 0.5  # Slightly above the boundary
        
        # Set text
        segment_id = segment_data['id']
        marker.text = f"({segment_id[0]},{segment_id[1]})"
        
        # Set scale (text size)
        marker.scale.z = 0.5
        
        # Set color (white text)
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        
        return marker
    
    def _clear_old_markers(self, marker_array, current_count):
        """Add DELETE markers to clear any old markers that are no longer needed"""
        # This is a simple approach - in practice you might want to track
        # the maximum number of markers you've ever published
        max_old_markers = 100  # Adjust based on your expected max segments
        
        for i in range(current_count, max_old_markers):
            # Clear old boundary markers
            delete_marker = Marker()
            delete_marker.header.frame_id = 'map'
            delete_marker.ns = "segment_boundaries"
            delete_marker.id = i
            delete_marker.action = Marker.DELETE
            marker_array.markers.append(delete_marker)
            
            # Clear old label markers
            delete_marker = Marker()
            delete_marker.header.frame_id = 'map'
            delete_marker.ns = "segment_labels"
            delete_marker.id = i
            delete_marker.action = Marker.DELETE
            marker_array.markers.append(delete_marker)

    def terminate_callback(self):
        """Serve as the "object detector" for terminating mission"""
        if not self.enable_goal_:
            return
        if self.odom_pos_ is None:
            self.get_logger().info('No odometry data available for termination.')
            return
        # check if odom_pos is within range of goal
        self.visualize_goal((self.goal_x_[self.current_goal_], self.goal_y_[self.current_goal_]), radius=self.goal_tol_[self.current_goal_], frame_id='map', z_height=float(self.odom_pos_[2]))
        goal = np.array([self.goal_x_[self.current_goal_], self.goal_y_[self.current_goal_]], dtype=np.float32)
        if np.linalg.norm(self.odom_pos_[:2] - goal) < self.goal_tol_[self.current_goal_]:
            self.get_logger().info('Goal reached!')
            self.transit_state(ExplorationState.FINISH)
            if self.enable_log_:
                self.eval_log_.log("Goal reached!")
                self.eval_log_.log(f"Task ended at : {self.get_clock().now()}")
                self.eval_log_.log(f"Task time (this may not be the mission total time): {(self.get_clock().now() - self.mission_start_time_).nanoseconds / 1e9} seconds")
                area_covered = get_area_discovered(self.last_map_)
                self.eval_log_.log(f"Mission total area covered: {area_covered} m^2")
                self.eval_log_.log(f"Mission total distance traveled: {self.total_distance_} m")

    def visualize_goal(self, position, radius=0.5, alpha=0.5, frame_id='map', z_height=1.0):
        """
        Visualizes a 2D goal position as a flat cylinder marker
        
        Args:
            position (tuple): 2D position as (x, y)
            radius (float): Radius of the goal circle
            alpha (float): Transparency (0.0 = invisible, 1.0 = opaque)
            frame_id (str): Frame ID for the marker
            z_height (float): Height at which to draw the circle
        """
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "goal"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        # Set position
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = z_height
        
        # Set orientation (no rotation needed)
        marker.pose.orientation.w = 1.0
        
        # Set scale (2D flat circle)
        marker.scale.x = radius * 2.0  # Diameter in x
        marker.scale.y = radius * 2.0  # Diameter in y
        marker.scale.z = 0.05  # Very thin height for 2D appearance
        
        # Set color (bright green for goal)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = float(alpha)
        # Publish the marker
        self.goal_pub_.publish(marker)
        # self.get_logger().info(f'Published goal at ({position[0]:.2f}, {position[1]:.2f})')
    

    def generate_init_wp(self, current_pos):
        """Generate a set of initial waypoints for robot initialization;
        output shape n x 2"""
        waypoints = []
        # Single waypoint in front of the robot
        wp = np.array([current_pos[0] + self.init_waypoints_x_, 
                       current_pos[1] + self.init_waypoints_y_])
        return [current_pos, wp]
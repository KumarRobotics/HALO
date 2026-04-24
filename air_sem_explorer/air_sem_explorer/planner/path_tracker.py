import numpy as np

class ExplorationPath:
    def __init__(self, waypoints_2d, timestamp, ignore_first=True):
        """
        Stores and manages an exploration path
        Args:
            waypoints_2d: List of 2D numpy arrays representing the path
        """
        self.waypoints = np.array(waypoints_2d)  # Nx2 array
        self.creation_time = timestamp
        if ignore_first and len(self.waypoints) >= 2:
            self.current_wp_index = 1  # Skip the first point
        else:
            self.current_wp_index = 0  # Index of currently tracked waypoint, skip the start point
        self.completed = False
        
    def current_waypoint(self):
        """Returns the current target waypoint"""
        if not self.completed:
            return self.waypoints[self.current_wp_index]
        return None
    
    def next_waypoint(self):
        """Returns the next waypoint without advancing the index"""
        if self.current_wp_index + 1 < len(self.waypoints):
            return self.waypoints[self.current_wp_index + 1]
        return None
    
    def advance_waypoint(self):
        """Move to the next waypoint in the path"""
        if self.current_wp_index + 1 < len(self.waypoints):
            self.current_wp_index += 1
            return True
        else:
            self.completed = True
            return False
        
    def get_path(self):
        """Returns the full path"""
        return self.waypoints
    
    def remaining_waypoints(self):
        """Returns all waypoints from current position onward"""
        return self.waypoints[self.current_wp_index:]
    
    def __len__(self):
        return len(self.waypoints)
    
    def __repr__(self):
        return f"ExplorationPath(waypoints={len(self.waypoints)}, current={self.current_wp_index})"


class PathTracker():
    def __init__(self, parent_node, distance_threshold=0.5, time_threshold=10.0, limit_time=False):
        """
        Enhanced path tracker with waypoint information in returns
        Args:
            parent_node: ROS2 node for clock access
            distance_threshold: Proximity threshold for waypoint advancement (meters)
            time_threshold: Maximum time to spend on a waypoint (seconds)
            limit_time: Whether to enforce time thresholds
        """
        self.parent_node = parent_node
        self.current_path = None
        self.dist_thresh = distance_threshold
        self.time_thresh = time_threshold
        self.limit_time = limit_time
        self.current_goal = None
        self.waypoint_reached_time = None
        self.current_waypoint_start_time = None
        
    def set_path(self, path: ExplorationPath):
        """Assign a new path to track"""
        self.current_path = path
        if path:
            self.current_goal = path.current_waypoint()
            self.waypoint_reached_time = None
            self.current_waypoint_start_time = self.parent_node.get_clock().now()
        
    def update(self, current_pose):
        """
        Update tracker state and return comprehensive information
        Args:
            current_pose: Current robot position (2D/3D numpy array)  
        Returns:
            dict: {
                'goal_updated': bool,
                'status': str,
                'current_waypoint': np.ndarray or None,
                'next_waypoint': np.ndarray or None,
                'remaining_waypoints': list
            }
        """
        current_time = self.parent_node.get_clock().now()
        
        # Default return structure
        result = {
            'goal_updated': False,
            'status': 'inactive',
            'current_waypoint': None,
            'next_waypoint': None,
            'remaining_waypoints': []
        }
        
        if not self.current_path or self.current_path.completed:
            result['status'] = 'completed' if self.current_path else 'no_path'
            return result
            
        # Populate waypoint information
        result['current_waypoint'] = self.current_path.current_waypoint()
        result['next_waypoint'] = self.current_path.next_waypoint()
        result['remaining_waypoints'] = self.current_path.remaining_waypoints()
        
        dist_to_goal = np.linalg.norm(current_pose[:2] - self.current_goal)
        
        # Condition 1: Immediate advancement when close enough
        if dist_to_goal < self.dist_thresh:
            if self._advance_waypoint(current_time):
                result['goal_updated'] = True
                result['status'] = 'waypoint_reached'
                # Update waypoint info after advancement
                result['current_waypoint'] = self.current_path.current_waypoint()
                result['next_waypoint'] = self.current_path.next_waypoint()
                result['remaining_waypoints'] = self.current_path.remaining_waypoints()
            return result
            
        # Condition 2: Timeout-based advancement
        elapsed_time = (current_time - self.current_waypoint_start_time).nanoseconds * 1e-9
        if self.limit_time and elapsed_time > self.time_thresh:
            if self._advance_waypoint(current_time):
                result['goal_updated'] = True
                result['status'] = 'timeout_advance'
                # Update waypoint info after advancement
                result['current_waypoint'] = self.current_path.current_waypoint()
                result['next_waypoint'] = self.current_path.next_waypoint()
                result['remaining_waypoints'] = self.current_path.remaining_waypoints()
            return result
            
        result['status'] = 'tracking'
        return result
        
    def _advance_waypoint(self, current_time):
        """Internal waypoint advancement handler"""
        if not self.current_path:
            return False
            
        if self.waypoint_reached_time is None:
            self.waypoint_reached_time = current_time
            
        if self.current_path.advance_waypoint():
            self.current_goal = self.current_path.current_waypoint()
            self.waypoint_reached_time = None
            self.current_waypoint_start_time = current_time
            return True
            
        self.current_goal = None
        return False

    def status(self):
        """Get current tracking status"""
        if not self.current_path:
            return 'no_path'
        elif self.current_path.completed:
            return 'completed'
        else:
            return 'tracking'
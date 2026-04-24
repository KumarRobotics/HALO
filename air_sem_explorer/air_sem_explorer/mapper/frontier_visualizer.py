import numpy as np
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import rclpy
from builtin_interfaces.msg import Time

class FrontierVisualizer():
    def __init__(self, parent_node):
        self.parent_node_ = parent_node
        # Publisher for visualization markers
        self.marker_pub_ = self.parent_node_.create_publisher(Marker, '/frontier_markers', 10)
        
        # Parameters
        self.cube_scale_ = 0.5  # Size of each frontier cube
        self.last_marker_count_ = 0  # Track previous marker count
        
    def visualize_frontiers(self, frontiers):
        """
        Visualize frontier clusters with different colors using MarkerArray
        
        Args:
            frontiers: List of frontier clusters, each containing 3D points
        """

        # Use single marker for all
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.parent_node_.get_clock().now().to_msg()
        marker.ns = "frontiers"
        marker.id = 0  # Single marker ID
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.scale.x = self.cube_scale_
        marker.scale.y = self.cube_scale_
        marker.scale.z = self.cube_scale_
        marker.pose.orientation.w = 1.0  # No rotation

        for i, frontier in enumerate(frontiers):
            # Add each frontier's points to the marker
            self.add_frontier_to_marker(frontier.cells, i, len(frontiers), marker)
        
        rclpy.logging.get_logger('FrontierVisualizer').info(f"Visualizing {len(frontiers)} frontiers")
        # Publish the marker array
        self.marker_pub_.publish(marker)
        self.last_marker_count_ = len(frontiers)
    
    def create_frontier_marker(self, points, frontier_id, total_frontiers):
        """
        Create a Marker message for a single frontier cluster
        
        Args:
            points: List of 3D points in the frontier
            frontier_id: Index of this frontier
            total_frontiers: Total number of frontiers for color mapping
        """
        marker = Marker()
        
        # Basic marker info
        marker.header.frame_id = "map"
        marker.header.stamp = self.parent_node_.get_clock().now().to_msg()
        marker.ns = "frontiers"
        marker.id = frontier_id
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        
        # Scale and orientation
        marker.scale.x = self.cube_scale_
        marker.scale.y = self.cube_scale_
        marker.scale.z = self.cube_scale_
        marker.pose.orientation.w = 1.0
        
        # Color (hue varies with frontier ID)
        hue = float(frontier_id) / max(1, total_frontiers - 1)
        color = self.hsv_to_rgb(hue, 1.0, 1.0)
        marker.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=0.8)
        
        # Add all points
        for pt in points:
            point = Point()
            point.x = float(pt[0])
            point.y = float(pt[1])
            point.z = 1.0  # Assuming 2D points with z=1.0
            marker.points.append(point)
            
        return marker
    

    def add_frontier_to_marker(self, points, frontier_id, total_frontiers, marker):
        """
        Add a frontier cluster to an existing Marker message
        
        Args:
            points: List of 3D points in the frontier
            frontier_id: Index of this frontier
            total_frontiers: Total number of frontiers for color mapping
            marker: Existing Marker message to add points to
        """
        # Color (hue varies with frontier ID)
        hue = float(frontier_id) / max(1, total_frontiers - 1)
        color = self.hsv_to_rgb(hue, 1.0, 1.0)
        
        # Add all points
        for pt in points:
            point = Point()
            point.x = float(pt[0])
            point.y = float(pt[1])
            point.z = 1.0
            marker.points.append(point)
            marker.colors.append(ColorRGBA(r=color[0], g=color[1], b=color[2], a=0.8))


    @staticmethod
    def hsv_to_rgb(h, s, v):
        """Convert HSV to RGB color (all values 0-1)"""
        if s == 0.0:
            return (v, v, v)
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        if i == 0:
            return (v, t, p)
        if i == 1:
            return (q, v, p)
        if i == 2:
            return (p, v, t)
        if i == 3:
            return (p, q, v)
        if i == 4:
            return (t, p, v)
        if i == 5:
            return (v, p, q)
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Set

@dataclass
class FrontierInfo:
    cells: np.ndarray  # Nx2 array of frontier points
    average: np.ndarray  # 2D centroid
    box_min: np.ndarray  # Bounding box min
    box_max: np.ndarray  # Bounding box max
    normal: np.ndarray  # Principal component normal
    viewpoint: np.ndarray  # Viewpoint for visualization
    id: int = -1  # Frontier ID

class FrontierDetector:
    def __init__(self, mapper, cluster_size_min = 3.0, cluster_size_max = 5.0):
        self.mapper_ = mapper
        self.map_origin_ = mapper.map_origin_
        self.map_size_ = mapper.map_size_
        self.resolution_ = mapper.resolution_
        self.map_voxel_num_ = mapper.map_voxel_num_

        # Frontier parameters
        self.cluster_min_ = cluster_size_min  # Minimum frontier size
        self.cluster_size_xy_ = cluster_size_max  # Max size before splitting
        self.frontier_flag_ = np.zeros(self.map_voxel_num_, dtype=np.uint8)  # 2D for 2D map
        
        # Storage
        self.frontiers_ = []
        self.inactive_frontiers_ = []
        self.tmp_frontiers_ = []
        self.removed_ids_ = []
        print("FrontierDetector initialized with map size:", self.map_size_)

    def search_frontiers(self, update_min: np.ndarray, update_max: np.ndarray) -> None:
        """Main frontier detection pipeline"""
        self.tmp_frontiers_.clear()
        
        # Remove changed frontiers in updated region
        self.remove_changed_frontiers(update_min, update_max)

        # Search new frontiers in inflated update region
        search_min = update_min - np.array([3, 3])
        search_max = update_max + np.array([3, 3])
        search_min = np.maximum(search_min, self.map_origin_)
        search_max = np.minimum(search_max, self.map_origin_ + self.map_size_)

        min_id = self.mapper_.pos_to_index(search_min)
        max_id = self.mapper_.pos_to_index(search_max)

        # Scan updated region for frontier seeds
        for x in range(min_id[0], max_id[0] + 1):
            for y in range(min_id[1], max_id[1] + 1):
                idx = np.array([x, y])
                if (self.frontier_flag_[x, y] == 0 and 
                    self.mapper_.is_known(idx) and 
                    self.mapper_.is_neighbor_unknown(idx)):
                    self.expand_frontier(idx)
        self.mapper_.get_logger().info("before split, tmp ftr size: {}".format(len(self.tmp_frontiers_)))
        # print tmp frontier averages
        for frontier in self.tmp_frontiers_:
            self.mapper_.get_logger().info(f"Frontier average: {frontier.average}, size: {len(frontier.cells)}")
        self.split_large_frontiers()
        self.frontiers_.extend(self.tmp_frontiers_)

    def remove_changed_frontiers(self, update_min: np.ndarray, update_max: np.ndarray) -> None:
        """Remove frontiers that overlap with changed map areas"""
        to_remove = []
        # before remove, print size
        self.mapper_.get_logger().info(f"Before remove, ftr size: {len(self.frontiers_)}")
        # First identify frontiers to remove
        for i, frontier in enumerate(self.frontiers_):
            if self.have_overlap(frontier.box_min, frontier.box_max, update_min, update_max):
                self.reset_frontier_flag(frontier)
                to_remove.append(i)
        # remove in reverse order
        for i in sorted(to_remove, reverse=True):
            del self.frontiers_[i]
        self.mapper_.get_logger().info(f"After remove, ftr size: {len(self.frontiers_)}")

    def remove_inactive_frontiers(self) -> None:
        """Remove frontiers that are inactive for too long"""
        to_remove = []
        for i, frontier in enumerate(self.inactive_frontiers_):
            if self._is_frontier_changed(frontier):
                self.reset_frontier_flag(frontier)
                to_remove.append(i)
        # remove in reverse order
        for i in sorted(to_remove, reverse=True):
            del self.inactive_frontiers_[i]

    def expand_frontier(self, first_idx: np.ndarray) -> None:
        """Region growing to find complete frontier cluster"""
        cell_queue = deque()
        expanded = []
        self.frontier_flag_[tuple(first_idx)] = 1
        
        cell_queue.append(first_idx)
        expanded.append(self.mapper_.index_to_pos(first_idx))
        while cell_queue:
            cur = cell_queue.popleft()
            
            # In our case, there is no known_free, so only known vs unknown
            for nbr in self.mapper_.get_neighbors(cur):
                # If already marked as frontier or not in map or not a frontier
                if (not self.mapper_.is_in_map(nbr) or 
                    self.frontier_flag_[tuple(nbr)] == 1 or
                    not (self.mapper_.is_known(nbr) and self.mapper_.is_neighbor_unknown(nbr))):
                    continue
                
                self.frontier_flag_[tuple(nbr)] = 1
                expanded.append(self.mapper_.index_to_pos(nbr))
                cell_queue.append(nbr)
        
        if len(expanded) >= self.cluster_min_:
            frontier = FrontierInfo(
                cells=np.array(expanded),
                average=np.zeros(2),
                box_min=np.zeros(2),
                box_max=np.zeros(2),
                normal=np.zeros(2),
                viewpoint=np.zeros(2),
            )
            self.compute_frontier_info(frontier)
            self.tmp_frontiers_.append(frontier)

    def split_large_frontiers(self) -> None:
        """Split frontiers that exceed size threshold"""
        splits = []
        for frontier in self.tmp_frontiers_:
            if self.split_along_principal_component(frontier, splits):
                continue
            splits.append(frontier)
        self.tmp_frontiers_ = splits

    def split_along_principal_component(self, frontier: FrontierInfo, splits: list[FrontierInfo]) -> bool:
        """Recursively split frontier along principal component"""
        mean = frontier.average
        need_split = any(np.linalg.norm(cell - mean) > self.cluster_size_xy_ for cell in frontier.cells)
        if not need_split:
            return False
        
        # Compute principal component
        cov = np.cov(frontier.cells.T)
        eigvals, eigvecs = np.linalg.eig(cov)
        first_pc = eigvecs[:, np.argmax(eigvals)]
        
        # Split frontier
        ftr1_cells = [c for c in frontier.cells if (c - mean).dot(first_pc) >= 0]
        ftr2_cells = [c for c in frontier.cells if (c - mean).dot(first_pc) < 0]

        ftr1 = FrontierInfo(
            cells=np.array(ftr1_cells),
            average=np.zeros(2),
            box_min=np.zeros(2),
            box_max=np.zeros(2),
            normal=np.zeros(2),
            viewpoint=np.zeros(2)
        )
        ftr2 = FrontierInfo(
            cells=np.array(ftr2_cells),
            average=np.zeros(2),
            box_min=np.zeros(2),
            box_max=np.zeros(2),
            normal=np.zeros(2),
            viewpoint=np.zeros(2)
        )
        self.compute_frontier_info(ftr1)
        self.compute_frontier_info(ftr2)
        
        # Recursive splitting
        new_splits_1 = []
        if self.split_along_principal_component(ftr1, new_splits_1):
            splits.extend(new_splits_1)
        else:
            splits.append(ftr1)
            
        new_splits_2 = []
        if self.split_along_principal_component(ftr2, new_splits_2):
            splits.extend(new_splits_2)
        else:
            splits.append(ftr2)
            
        return True
    

    def compute_frontier_info(self, frontier):
        """
        Vectorized computation of frontier information for 2D frontiers.
        """
        if len(frontier.cells) == 0:
            self.mapper_.get_logger().warn("Empty frontier cluster!")
            return
        # Convert to numpy array if not already
        cells = np.asarray(frontier.cells)   
        frontier.average = np.mean(cells, axis=0)  # [avg_x, avg_y]
        frontier.box_min = np.min(cells, axis=0)   # [min_x, min_y]
        frontier.box_max = np.max(cells, axis=0)  # [max_x, max_y]
        frontier.size = len(cells)
    
    
    def have_overlap(self, min1: np.ndarray, max1: np.ndarray, 
                     min2: np.ndarray, max2: np.ndarray) -> bool:
        # Compute intersection box bounds
        bmin = np.maximum(min1, min2)
        bmax = np.minimum(max1, max2)
        
        # Check if intersection is valid in all dimensions
        return np.all(bmin <= bmax + 1e-3)
    
    def _is_frontier_changed(self, frontier: FrontierInfo) -> bool:
        # Implement change detection logic
        return False
    
    def reset_frontier_flag(self, frontier: FrontierInfo) -> None:
        if len(frontier.cells) == 0:
            return
        indices = np.floor((frontier.cells[:, :2] - self.map_origin_[:2]) / self.resolution_).astype(int)
        # Clamp indices to map bounds
        indices = np.clip(indices, 
                        np.array([0, 0]), 
                        np.array(self.map_voxel_num_[:2]) - 1)
        # Convert to tuple of arrays for advanced indexing
        rows, cols = indices[:, 0], indices[:, 1]
        # Vectorized flag reset
        self.frontier_flag_[rows, cols] = 0
    
    def reset_all_frontier_flags(self) -> None:
        self.frontier_flag_.fill(0)

    def reset_frontiers(self) -> None:
        self.frontiers_.clear()
        self.tmp_frontiers_.clear()
        self.inactive_frontiers_.clear()
        self.reset_all_frontier_flags()
    
    def sample_viewpoints(self):
        for frontier in self.tmp_frontiers_:
            # Sample viewpoint as the average position
            frontier.viewpoint = frontier.average.copy()
            # # Add to main frontiers list
            # self.frontiers_.append(frontier)
    
    def clear_nearby_frontiers(self, position: np.ndarray, clear_distance: float) -> None:
        """Clear frontiers within a certain distance from the given position"""
        to_remove = []
        for i, frontier in enumerate(self.frontiers_):
            dist = np.linalg.norm(frontier.average - position)
            if dist < clear_distance:
                self.reset_frontier_flag(frontier)
                to_remove.append(i)
        # Remove in reverse order
        for i in sorted(to_remove, reverse=True):
            del self.frontiers_[i]
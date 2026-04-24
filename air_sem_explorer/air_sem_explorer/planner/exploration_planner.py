"""
    Yuezhan Tao
    July 2025

    occ_map class for ros2 node
"""

import numpy as np
import networkx as nx
from air_sem_explorer.utils.utils import msg_to_map, msg_to_frontier
import pickle
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def solve_tsp_ortools(G, timeout=1):
    """Solve ATSP using OR-Tools"""
    print("\n=== OR-Tools ATSP ===")

    # --- Create distance matrix ---
    nodes = list(G.nodes)
    n = len(nodes)

    distance_matrix = np.zeros((n, n))
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if G.has_edge(u, v):
                distance_matrix[i, j] = G[u][v]['weight']
    
    # OR-Tools requires integers
    int_matrix = (distance_matrix * 1000).astype(int)

    # --- OR-Tools setup ---
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)  # start and end at node 0
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        # Convert routing indices to node indices
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # --- Search parameters ---
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = int(timeout)

    # --- Solve ---
    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        print("No solution found!")
        return None, None

    # --- Extract route ---
    path = []
    total_distance = 0
    index = routing.Start(0)
    while not routing.IsEnd(index):
        node_index = manager.IndexToNode(index)
        path.append(nodes[node_index])
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        if not routing.IsEnd(index):
            arc_cost = routing.GetArcCostForVehicle(previous_index, index, 0)
            total_distance += arc_cost

    # Include return to start (last arc)
    arc_cost = routing.GetArcCostForVehicle(previous_index, routing.Start(0), 0)
    total_distance += arc_cost
    path.append(path[0])  # complete the cycle

    actual_distance = total_distance / 1000.0

    return path, actual_distance

class ExplorationPlanner:
    def __init__(self, logger, map_origin, map_size, resolution, map_voxel_num, region_size=10, 
                 global_relevancy_thres=0.4, retrieval_thres=0.8, ftr_relevancy_thres=0.2):
        # Store map metadata
        self.logger = logger
        self.map_origin_ = np.array(map_origin, dtype=np.float32)  # Map origin in world coordinates
        self.map_size_ = np.array(map_size, dtype=np.float32)  # Map size in world coordinates
        self.resolution_ = resolution  # Map resolution in meters per voxel
        self.map_voxel_num_ = np.array(map_voxel_num, dtype=np.int32)  # Number of voxels in the map
        self.viewpoint_graph_id_ = 0 # Unique ID for each viewpoint graph node
        self.global_relevancy_thres_ = global_relevancy_thres  # Threshold for relevancy map to consider a voxel as relevant
        self.ftr_relevancy_thres_ = ftr_relevancy_thres # Threshold for frontier relevancy to be considered valid
        self.retrieval_thres_ = retrieval_thres

        self.relevancy_map_ = None  # A 2D numpy array representing the relevancy map
        self.frontiers_ = None  # A list of frontier points to explore
        self.region_size_ = region_size # Segment the map into regions for high-level global planning
        self.region_graph_ = nx.Graph()
        self.viewpoint_graph_ = nx.DiGraph()

        self.segment_map() # Segment the map into regions, pre-compute the bounds of each segment
        self.init_region_graph()


    def segment_map(self):
        """
        Compute the bounds of the segments and store them in a dictionary with (i,j) indices as keys.
        Each entry contains:
        - world_bounds: ((min_x, max_x), (min_y, max_y)) in world coordinates
        - voxel_bounds: ((min_vx, max_vx), (min_vy, max_vy)) in voxel indices
        
        Returns:
            dict: Dictionary containing segment information keyed by (i,j) indices
        """
        self.segments_ = {}  # Dictionary to store all segments
        
        self.num_segments_x_ = int(np.ceil(self.map_size_[0] / self.region_size_))
        self.num_segments_y_ = int(np.ceil(self.map_size_[1] / self.region_size_))

        for i in range(self.num_segments_x_):
            for j in range(self.num_segments_y_):
                # Calculate world coordinates bounds
                min_x = self.map_origin_[0] + i * self.region_size_
                min_y = self.map_origin_[1] + j * self.region_size_
                max_x = min(min_x + self.region_size_, self.map_origin_[0] + self.map_size_[0])
                max_y = min(min_y + self.region_size_, self.map_origin_[1] + self.map_size_[1])
                
                # Calculate corresponding voxel indices bounds
                min_vx = int((min_x - self.map_origin_[0]) / self.resolution_)
                min_vy = int((min_y - self.map_origin_[1]) / self.resolution_)
                max_vx = int((max_x - self.map_origin_[0]) / self.resolution_)
                max_vy = int((max_y - self.map_origin_[1]) / self.resolution_)
                # make sure indices are within bounds
                min_vx = max(min_vx, 0)
                min_vy = max(min_vy, 0)
                max_vx = min(max_vx, self.map_voxel_num_[0])
                max_vy = min(max_vy, self.map_voxel_num_[1])

                # Store in dictionary with (i,j) as key
                self.segments_[(i, j)] = {
                    'pos_bounds': ((min_x, max_x), (min_y, max_y)),
                    'voxel_bounds': ((min_vx, max_vx), (min_vy, max_vy)),
                    'region_center': ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0),
                    'id': (i, j)
                }

        return self.segments_

    def get_region_segments(self):
        return self.segments_

    def init_region_graph(self):
        """
        Initialize the region graph.
        """
        # Iterate through all segments in the dictionary
        for _, seg_data in self.segments_.items():  
            # Add node with segment information
            self.region_graph_.add_node(
                seg_data['id'],
                seg_pos=seg_data['pos_bounds'],
                seg_idx=seg_data['voxel_bounds'],
                region_center=seg_data['region_center'],
                utility=0.0,
                has_frontier=False,
                retrieval=False
            )
        
    
    def reset_region_graph(self):
        """
        Reset the region graph for a new exploration session.
        """
        for node in self.region_graph_.nodes:
            self.region_graph_.nodes[node]['utility'] = 0.0
            self.region_graph_.nodes[node]['has_frontier'] = False

    def reset_viewpoint_graph(self):
        """
        Reset the viewpoint graph for a new exploration session.
        """
        self.viewpoint_graph_ = nx.DiGraph()  # Reset viewpoint graph
        self.viewpoint_graph_id_ = 0  # Reset viewpoint graph ID for new session


    def update_global_graph(self, relevancy_map_msg, frontiers_msg, retrieval=False):
        # Parse the info from msg.
        map_origin, map_size, map_voxel_num, resolution, map_data = msg_to_map(relevancy_map_msg)
        self.relevancy_map_ = map_data
        frontiers = msg_to_frontier(frontiers_msg)
        self.frontiers_ = frontiers
        self.reset_region_graph()
        # If no frontiers, and not retrieval, return False
        if len(frontiers) == 0 and not retrieval:
            return False
        # update the region_graph with frontier info
        self.update_frontier_info()
        # filter the map
        map_filtered = np.where(self.relevancy_map_ > self.global_relevancy_thres_, 
                                self.relevancy_map_, 0)

        # If retrieval
        # We will flip the flag for all regions worth visiting
        # 1. all regions without frontiers, will be using max cell relevancy as utility
        # 2. all regions with frontiers, will be using mean relevancy of the region as utility
        if retrieval:
            self.logger.info("Retrieval mode: Setting flags for all regions.")
            for node in self.region_graph_.nodes:
                seg_idx = self.region_graph_.nodes[node]['seg_idx']
                seg_map = map_filtered[seg_idx[0][0]:seg_idx[0][1], seg_idx[1][0]:seg_idx[1][1]]
                # Check if there are frontiers in this segment
                if self.region_graph_.nodes[node]['has_frontier']:
                    # Use mean relevancy as utility
                    utility = np.mean(seg_map) + 1e-6
                else:
                    # Use max relevancy as utility
                    utility = np.max(seg_map) + 1e-6
                    self.logger.info(f"Considering retrival for region {node}, max relevancy: {utility:.6f}")
                    if utility < self.retrieval_thres_:
                        utility = 1e-6
                    else:
                        self.region_graph_.nodes[node]['retrieval'] = True

                self.region_graph_.nodes[node]['utility'] = utility

        else:
        # if not retrieval
        # 1. only consider regions with frontiers, average relevancy of the region as utility
            for node in self.region_graph_.nodes:
                seg_idx = self.region_graph_.nodes[node]['seg_idx']
                seg_map = map_filtered[seg_idx[0][0]:seg_idx[0][1], seg_idx[1][0]:seg_idx[1][1]]
                # Check if there are frontiers in this segment
                if self.region_graph_.nodes[node]['has_frontier']:
                    # Use mean relevancy as utility
                    utility = np.mean(seg_map) + 1e-6
                elif self.region_graph_.nodes[node]['retrieval']:
                    utility = np.max(seg_map) + 1e-6
                    if utility < self.retrieval_thres_:
                        utility = 1e-6
                else:
                    utility = 1e-6
                self.region_graph_.nodes[node]['utility'] = utility
        return True

    def update_frontier_info(self):
        """
        Update the region graph with frontier information.
        This will set the 'has_frontier' flag for each segment based on the presence of frontiers.
        """
        for ftr in self.frontiers_:
            # Convert frontier position to segment index
            region_x, region_y = self.pos_to_region(ftr.average)
            # self.logger.info(f"Frontier at {ftr.average} is in region ({region_x}, {region_y})")
            if region_x == -1 or region_y == -1:
                print(f"Warning: Frontier is outside the mapped area, skipping.")
                continue
            if (region_x, region_y) in self.segments_:
                # Set the has_frontier flag to True for this segment
                self.region_graph_.nodes[(region_x, region_y)]['has_frontier'] = True


    def global_planner(self, start_pos, relevancy_map, frontiers, retrieval=False):
        """
        Plan exploration region based on the relevancy map and frontiers.
        Args:
            relevancy_map (np.ndarray): A 2D numpy array representing the relevancy map.
            frontiers (list): A list of frontier points to explore.
            retrieval (bool): If True, consider retrieval in planning.   
        """
        # Update the global planning graph with the relevancy map and frontiers
        if not self.update_global_graph(relevancy_map, frontiers, retrieval):
            return None, False

        # Plan to next best region based on the region graph.
        # using cost benefit.
        # The start_pos should be in the same frame as the map.
        best_region_id = None
        best_cost_benefit = -float('inf')
        for node_id in self.region_graph_.nodes:
            # if not retrieval, only consider regions with frontiers
            if not self.region_graph_.nodes[node_id]['retrieval'] and not self.region_graph_.nodes[node_id]['has_frontier']:
                continue
            # Calculate cost as distance from start_pos to region center
            region_center = self.region_graph_.nodes[node_id]['region_center']
            cost = np.linalg.norm(np.array(start_pos) - np.array(region_center))
            utility = self.region_graph_.nodes[node_id]['utility']
            self.logger.info(f"Region {node_id}: Cost = {cost:.2f}, Utility = {utility:.6f}")
            # compute cost-benefit ratio
            cost_benefit = utility / cost
            if cost_benefit > best_cost_benefit:
                best_cost_benefit = cost_benefit
                best_region_id = node_id

        if best_region_id is None:
            self.logger.info("No suitable region found for exploration.")
            return None, False

        # check if the best region is retrival region or frontier region
        if self.region_graph_.nodes[best_region_id]['has_frontier']:
            print(f"Best region for exploration: {best_region_id}")
            return best_region_id, False
        else:
            print(f"Best region for retrieval: {best_region_id}")
            return best_region_id, True


    def local_planner_ortools(self, start_pos, relevancy_map_msg, frontiers_msg, region_id, region_retrieval=False, min_distance=2.0):
        """
        Plan exploration waypoints based on the region map.
        """
        # if regions is retrieval, we don't need to consider frontiers, just go to the region center
        if region_retrieval:
            if region_id is None:
                self.logger.info("No valid region found for retrieval.")
                return []
            # Return waypoint with start_pos and at the center of the region
            region_center = self.region_graph_.nodes[region_id]['region_center']
            return [start_pos, region_center]

        frontiers = msg_to_frontier(frontiers_msg)
        self.frontiers_ = frontiers
        if len(frontiers) == 0:
            self.logger.info("No frontiers available for local planning.")
            return []
        region_ftrs = [f for f in frontiers if self.pos_to_region(f.average) == region_id]
        # If region has no more ftr, switch back to global planning
        if len(region_ftrs) == 0:
            self.logger.info(f"No frontiers found in region {region_id}. Switching to global planning.")
            return []
        self.logger.info(f"Region frontiers size: {len(region_ftrs)}")

        # Apply another filter, compute average frontier utility, only keep frontiers with utility above average
        map_origin, map_size, map_voxel_num, resolution, map_data = msg_to_map(relevancy_map_msg)
        map_filterd = np.where(map_data > self.ftr_relevancy_thres_,
                            map_data, 0)
        # Frontier needs to have average relevancy above the relevancy threshold
        valid_region_ftrs = []
        for ftr in region_ftrs:
            dist = np.linalg.norm(start_pos - ftr.viewpoint)
            if dist < min_distance:
                continue
            cells = ftr.cells
            # convert cells positions to index
            cell_index = np.floor((cells - map_origin) / resolution).astype(np.int32)
            # Clamp to map bounds for each dimension
            for d in range(2):
                cell_index[..., d] = np.clip(cell_index[..., d], 0, map_voxel_num[d] - 1)
            relevancy = map_filterd[cell_index[:, 0], cell_index[:, 1]]
            utility = np.mean(relevancy)
            if utility > self.ftr_relevancy_thres_:
                valid_region_ftrs.append(ftr)
        if len(valid_region_ftrs) == 0:
            self.logger.info(f"No valid frontiers found in region {region_id} after filtering. Switching to local traversal.")
            valid_region_ftrs = region_ftrs
        
        
        self.reset_viewpoint_graph()    
        # Add depot node
        DEPOT_NODE = "f_-1"
        self.viewpoint_graph_.add_node(DEPOT_NODE, pos=start_pos, is_depot=True)

        # Add all frontier nodes
        for i, ftr in enumerate(valid_region_ftrs):
            node_id = f"f_{i}"
            self.viewpoint_graph_.add_node(node_id, pos=ftr.viewpoint, is_depot=False)
            self.viewpoint_graph_.add_edge(DEPOT_NODE, node_id,
                            weight=np.linalg.norm(start_pos - ftr.viewpoint))

        self.logger.info("Computing distance matrix for TSP...")
        for i, ftr in enumerate(valid_region_ftrs):
            self.logger.info(f"f_{i} viewpoint: {ftr.viewpoint}, average: {ftr.average}, box_min: {ftr.box_min}, box_max: {ftr.box_max}")

        if len(valid_region_ftrs) > 1:
            positions = np.array([ftr.viewpoint for ftr in valid_region_ftrs])
            for i in range(len(valid_region_ftrs)):
                for j in range(i + 1, len(valid_region_ftrs)):  # Only compute j > i
                    dist = np.linalg.norm(positions[i] - positions[j])
                    # self.logger.info(f"distance between f_{i} and f_{j}: {dist:.2f}")
                    # self.logger.info(f"view point of i: {positions[i]}, view point of j: {positions[j]}")
                    self.viewpoint_graph_.add_edge(f"f_{i}", f"f_{j}", weight=dist)
                    self.viewpoint_graph_.add_edge(f"f_{j}", f"f_{i}", weight=dist)

            # Solve TSP directly on our complete graph
            try:
                self.logger.info("Solving TSP with ORTools for local path...")
                tour, _ = solve_tsp_ortools(self.viewpoint_graph_)
                self.logger.info(f"Local ATSP tour: {tour}")
                # Extract waypoints (excluding return legs)
                waypoints = [self.viewpoint_graph_.nodes[n]['pos'] for n in tour[:-1]]  # Exclude last node
                return waypoints

            except:
                self.logger.warn(f"ATSP solving failed")
                return []
        else:
            # If only one frontier, return its viewpoint as the waypoint
            ftr = valid_region_ftrs[0]
            self.logger.info(f"Only one frontier in region {region_id}, returning its viewpoint.")
            return [start_pos, ftr.viewpoint]
        

    def pos_to_region(self, pos):
        """
        Convert a position in world coordinates to region (i,j) indices.
        
        Args:
            pos (np.ndarray): 2D position in world coordinates (x,y)
        
        Returns:
            tuple: (i,j) region indices, or None if position is outside mapped area
        
        Raises:
            ValueError: If position is outside valid map bounds
        """
        # Convert to relative coordinates
        rel_x = pos[0] - self.map_origin_[0]
        rel_y = pos[1] - self.map_origin_[1]
        
        # Check bounds (now handles negative values)
        if (rel_x < 0 or rel_y < 0 or 
            rel_x >= self.map_size_[0] or rel_y >= self.map_size_[1]):
            return -1, -1  # Invalid position
        
        # Calculate region indices
        region_x = int(rel_x // self.region_size_)
        region_y = int(rel_y // self.region_size_)
        
        # Clamp to valid regions (shouldn't be needed due to bounds check)
        region_x = min(region_x, self.num_segments_x_ - 1)
        region_y = min(region_y, self.num_segments_y_ - 1)
        
        return (region_x, region_y)
    

    def check_region_flag(self, pos):
        """
        Check if we are inside a retrieval region, and update the flag accordingly.
        """
        region = self.pos_to_region(pos)
        if region != (-1, -1):
            # if flag = true, flip it
            if self.region_graph_.nodes[region]['retrieval'] == True:
                self.region_graph_.nodes[region]['retrieval'] = False
                self.logger.info(f"Inside retrieval region {region}, setting retrieval flag to false.")
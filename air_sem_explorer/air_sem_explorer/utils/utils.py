import numpy as np
from scipy.spatial.transform import Rotation as R
from air_sem_gridmap_interfaces.msg import RelevancyMap
from air_sem_explorer_interfaces.msg import Frontiers, Frontier
from air_sem_explorer.mapper.frontier_detector import FrontierInfo
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def quat_msg_to_rot_mat(orientation) -> np.ndarray:
    # Extract quaternion [x, y, z, w]
    quat = [
        orientation.x,
        orientation.y,
        orientation.z,
        orientation.w
    ]
    
    # Convert to 3x3 rotation matrix
    return R.from_quat(quat).as_matrix()


def map_to_msg(origin: np.ndarray,
               size: np.ndarray,
               voxel_num: np.ndarray,
               resolution: float,
               data: np.ndarray) -> dict:
    """
    Convert map origin and resolution to a message format.
    """
    map = RelevancyMap()
    map.origin_x = float(origin[0])
    map.origin_y = float(origin[1])
    map.size_x = float(size[0])
    map.size_y = float(size[1])
    map.resolution = float(resolution)
    map.voxel_num_x = int(voxel_num[0])
    map.voxel_num_y = int(voxel_num[1])
    map.data = data.flatten().tolist()  # Flatten to 1D list
    return map


def msg_to_map(msg: RelevancyMap) -> tuple:
    """
    Convert a message to map parameters.
    """
    origin = np.array([msg.origin_x, msg.origin_y])
    size = np.array([msg.size_x, msg.size_y])
    voxel_num = np.array([msg.voxel_num_x, msg.voxel_num_y], dtype=int)
    resolution = msg.resolution
    data = np.array(msg.data).reshape(voxel_num[0], voxel_num[1])
    # Ensure data is a 2D numpy array with the correct shape
    assert data.shape == (voxel_num[0], voxel_num[1]), "Data shape does not match voxel numbers"
    return origin, size, voxel_num, resolution, data


def msg_to_frontier(msg: Frontiers) -> list:
    """
    Convert frontier msg to Frontier objects
    """
    frontiers = []
    for ftr in msg.frontiers:
        x_pos = np.array(ftr.cells_x, dtype=np.float32)
        y_pos = np.array(ftr.cells_y, dtype=np.float32)
        cell_pos = np.column_stack((x_pos, y_pos))
        avg = np.array(ftr.average, dtype=np.float32)
        box_min = np.array(ftr.box_min, dtype=np.float32)
        box_max = np.array(ftr.box_max, dtype=np.float32)
        vp = np.array(ftr.viewpoint, dtype=np.float32)
        ftr = FrontierInfo(
            cells=cell_pos,
            average=avg,
            box_min=box_min,
            box_max=box_max,
            normal=np.zeros(2),  # Placeholder, not used in this context
            viewpoint=vp
        )
        frontiers.append(ftr)
    return frontiers


def frontier_to_msg(frontiers: list) -> Frontiers:
    msg = Frontiers()
    for ftr in frontiers:
        ftr_msg = Frontier()
        ftr_msg.cells_x = ftr.cells[:, 0].tolist()
        ftr_msg.cells_y = ftr.cells[:, 1].tolist()
        ftr_msg.average = ftr.average.tolist()
        ftr_msg.box_min = ftr.box_min.tolist()
        ftr_msg.box_max = ftr.box_max.tolist()
        ftr_msg.viewpoint = ftr.viewpoint.tolist()
        msg.frontiers.append(ftr_msg)
    return msg


def get_area_discovered(last_map):
    """
    Get the area discovered in the occupancy map.
    
    Returns:
        float: Area discovered in square meters
    """
    # covert last map 
    map_origin, map_size, map_voxel_num, resolution, map_data = msg_to_map(last_map)
    mapped_voxels = np.sum(map_data != -1)
    area_discovered = mapped_voxels * (resolution ** 2)
    return area_discovered
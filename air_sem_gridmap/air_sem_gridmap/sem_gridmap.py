import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from sklearn.cluster import KMeans
import warnings
from sklearn.exceptions import ConvergenceWarning
from air_sem_gridmap.naradio.naradio import NARadioEncoder

class SemGridMap:
    def __init__(self, config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_resolution = tuple(config.get('input_resolution', (640, 480)))
        self.feature_dim = config.get('feature_dim', 1152)
        self.grid_resolution = config.get('resolution', 1.0)
        self.pointcloud_downsample_voxel_size = config.get('pointcloud_downsample_voxel_size', 0.5)
        self.ema_alpha      = config.get('ema_alpha', 0.1)
        self._cached_prompt = None
        self._cached_prompt_embedding = None
        
        # fixed grid bounds
        self.map_min_x = config.get('map_min_x', -100.0)
        self.map_max_x = config.get('map_max_x', 100.0)
        self.map_min_y = config.get('map_min_y', -50.0)
        self.map_max_y = config.get('map_max_y', 50.0)

        # update box
        self.update_box = None
        self.relevancy_map = None
        self.highest_point = None
        
        # grid origin and size
        self.grid_origin = torch.tensor([self.map_min_x, self.map_min_y], device=self.device)
        self.map_size = torch.tensor([self.map_max_x - self.map_min_x, self.map_max_y - self.map_min_y], device=self.device)
        self.map_voxel_num = torch.ceil(self.map_size / self.grid_resolution).long()
        
        # initialize fixed-size grids
        H, W = self.map_voxel_num.tolist()
        self.grid_features = torch.zeros((H, W, self.feature_dim), dtype=torch.float32, device=self.device)
        self.grid_counts   = torch.zeros((H, W), dtype=torch.long, device=self.device)
        
        # coordinate grids
        self.x_coords = self.grid_origin[0] + (torch.arange(H, device=self.device) + 0.5) * self.grid_resolution
        self.y_coords = self.grid_origin[1] + (torch.arange(W, device=self.device) + 0.5) * self.grid_resolution
        
        # persistent global voxelized point cloud
        self.global_points = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        self.global_colors = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        self.publish_pointcloud = config.get('publish_pointcloud', False)
        
        # camera extrinsics: cam2body for camera pitched 90 deg down
        self.cam2body_R = torch.tensor([
                            [0, -1, 0],
                            [-1, 0, 0],
                            [0, 0, -1]
                        ],
                        dtype=torch.float32,
                        device=self.device)
        self.fx = 415.69219771027923
        self.fy = 415.69219771027923
        self.cx = 320.0
        self.cy = 240.0
        
        # encoder initialization
        self.encoder = NARadioEncoder(model_version="radio_v2.5-b", lang_model="siglip", input_resolution=self.input_resolution)
        self.encoder.model = self.encoder.model.to(self.device)
        # Use default prompt from config
        self.set_text_prompt(config.get('default_text_prompt', ['blue car on road', 'person', 'building']))
        self.num_positive_classes = config.get('num_positive_classes', 1)
        
    def set_text_prompt(self, text_prompt: list[str]):
        """Set the text prompt and compute its embedding. If the prompt is the same as the cached one, do nothing."""
        if self._cached_prompt == text_prompt and self._cached_prompt_embedding is not None:
            return
        with torch.no_grad():
            text_feat = self.encoder.encode_labels(text_prompt)  # shape [N, C], torch
            self._cached_prompt_embedding = text_feat.to(self.device)
            self._cached_prompt = text_prompt

    def voxel_downsample(self, points, colors, voxel_size):
        """Downsample points and colors to a voxel grid with the given voxel size."""
        # Quantize to voxel grid
        coords = torch.floor(points / voxel_size).int()
        unique_coords, inverse, counts = torch.unique(coords, dim=0, return_inverse=True, return_counts=True)
        last_indices = torch.zeros(unique_coords.shape[0], dtype=torch.long, device=points.device)
        last_indices.scatter_(0, inverse, torch.arange(points.shape[0], device=points.device))
        voxel_centers = (unique_coords.float() + 0.5) * voxel_size
        voxel_colors = colors[last_indices]
        return voxel_centers, voxel_colors

    def update_global_pointcloud(self, new_points, new_colors):
        """Update the global point cloud with new points and colors, applying voxel downsampling."""
        if new_points.shape[0] == 0:
            return
        all_points = torch.cat([self.global_points, new_points], dim=0)
        all_colors = torch.cat([self.global_colors, new_colors], dim=0)
        voxel_points, voxel_colors = self.voxel_downsample(all_points, all_colors, self.pointcloud_downsample_voxel_size)
        self.global_points = voxel_points
        self.global_colors = voxel_colors

    def process_measurement(self, rgb, depth, pose, intrinsics):
        """
        Process a measurement given numpy arrays for rgb (H, W, 3), depth (H, W), and a pose array [x, y, z, qx, qy, qz, qw]
        """
        rgb_tensor = torch.from_numpy(rgb).float().permute(2,0,1).unsqueeze(0).to(self.device) / 255.0
        depth = torch.from_numpy(depth).float().to(self.device) / 1000.0  # meters

        # resize using nvidia get nearest supported resolution
        h_orig, w_orig = rgb_tensor.shape[2], rgb_tensor.shape[3]
        h_resized, w_resized = self.encoder.get_nearest_size(h_orig, w_orig)
        
        # resize
        rgb_tensor = F.interpolate(rgb_tensor, size=(h_resized, w_resized), mode='nearest')

        # scale intrinsics accordingly
        intrinsics = torch.tensor(intrinsics, dtype=torch.float32, device=self.device)
        intrinsics[0, 0] *= w_resized / w_orig  # fx
        intrinsics[1, 1] *= h_resized / h_orig  # fy
        intrinsics[0, 2] *= w_resized / w_orig  # cx
        intrinsics[1, 2] *= h_resized / h_orig  # cy

        # update encoder
        self.encoder.input_resolution = (h_resized, w_resized)

        # get feature map
        with torch.no_grad():
            feat_map = self.encoder.encode_image_to_feat_map(rgb_tensor)
            feat_map = self.encoder.align_spatial_features_with_language(feat_map)
        feat_map = feat_map.squeeze(0)  # [C, H_feat, W_feat], torch
        H, W = depth.shape
        feat_map_upsampled = F.interpolate(feat_map.unsqueeze(0), size=(H, W), mode='nearest').squeeze(0)  # [C, H, W]
        self.last_rgb_down = torch.from_numpy(rgb).to(self.device)  # Store for visualization
        points_3d = self.project_to_3d(depth, pose, intrinsics)
        # update highest point
        max_height = points_3d[:,2].max().item()
        # if max_height > self.highest_point:
        if self.highest_point is None:
            self.highest_point = max_height
        # compute min and max grid cells that are updated
        min_x = (points_3d[:,0].min() - self.grid_origin[0]) / self.grid_resolution
        max_x = (points_3d[:,0].max() - self.grid_origin[0]) / self.grid_resolution
        min_y = (points_3d[:,1].min() - self.grid_origin[1]) / self.grid_resolution
        max_y = (points_3d[:,1].max() - self.grid_origin[1]) / self.grid_resolution
        buffer_size = 0
        min_x = int(torch.clamp(min_x.floor() - buffer_size, 0, self.map_voxel_num[0]-1).item())
        max_x = int(torch.clamp(max_x.ceil() + buffer_size, 0, self.map_voxel_num[0]-1).item())
        min_y = int(torch.clamp(min_y.floor() - buffer_size, 0, self.map_voxel_num[1]-1).item())
        max_y = int(torch.clamp(max_y.ceil() + buffer_size, 0, self.map_voxel_num[1]-1).item())
        self.update_box = (min_x, max_x, min_y, max_y)
        rgb_flat = torch.from_numpy(rgb.reshape(-1, 3)).float().to(self.device) / 255.0
        if self.publish_pointcloud:
            self.update_global_pointcloud(points_3d, rgb_flat)
        self.aggregate_to_grid(points_3d, feat_map_upsampled)

    def project_to_3d(self, depth, pose, intrinsics):
        """Project depth image to 3D points in world frame."""
        H, W = depth.shape
        fx = intrinsics[0, 0]  # fy
        fy = intrinsics[1, 1]  # fx
        cx = intrinsics[0, 2]  # cx
        cy = intrinsics[1, 2]  # cy
        pos = pose[:3] # xyz
        quat = pose[3:] # xyzw
        R = torch.from_numpy(self.quaternion_to_rotmat(quat)).float().to(self.device)
        t = torch.tensor(pos, dtype=torch.float32, device=self.device)
        xs, ys = torch.meshgrid(torch.arange(W, device=self.device), torch.arange(H, device=self.device), indexing='xy')
        zs = depth
        xs = (xs - cx) * zs / fx
        ys = (ys - cy) * zs / fy
        pts_cam = torch.stack([xs, ys, zs], dim=-1).reshape(-1, 3)
        pts_body = (self.cam2body_R @ pts_cam.T).T
        pts_world = (R @ pts_body.T).T + t
        return pts_world

    def quaternion_to_rotmat(self, q):
        # q should be [w, x, y, z]
        # scipy expects [x, y, z, w]
        r = Rotation.from_quat([q[1], q[2], q[3], q[0]])
        return r.as_matrix()

    def pos_to_index(self, pos):
        """
        Convert world position(s) to grid index/indices (same as occ_map.py).
        
        Args:
            pos: shape (..., 2) tensor (x, y) or (x, y, z) position(s) in world frame
            
        Returns:
            index: (..., 2) tensor of grid indices, clamped to map bounds
        """
        index = torch.floor((pos - self.grid_origin) / self.grid_resolution).long()
        # Clamp to map bounds for each dimension
        for d in range(2):
            index[..., d] = torch.clamp(index[..., d], 0, self.map_voxel_num[d] - 1)
        return index

    def index_to_pos(self, index):
        """
        Convert grid index to world position (center of voxel) (same as occ_map.py).
        
        Args:
            index: tensor of grid indices
            
        Returns:
            pos: tensor of world position (x,y) or (x,y,z)
        """
        return (index + 0.5) * self.grid_resolution + self.grid_origin

    def is_in_map(self, index):
        """
        Check if index is within map bounds (same as occ_map.py).
        
        Args:
            index: tensor of grid indices
            
        Returns:
            bool: True if index is within map bounds
        """
        # Handle both single index and batch of indices
        if index.dim() == 1:
            # Single index
            return torch.all((index >= 0) & (index < self.map_voxel_num))
        else:
            # Batch of indices - check each one
            return torch.all((index >= 0) & (index < self.map_voxel_num), dim=1)

    def aggregate_to_grid(self, pts_world, feat_map, ema_alpha=0.1):
        """Aggregate point features into grid cells using mean and EMA."""
        xy = pts_world[:, :2]
            
        # Reshape feature map to match point cloud
        # feat_map shape: [C, H, W] -> [H*W, C]
        C, H, W = feat_map.shape
        feats = feat_map.permute(1, 2, 0).reshape(H * W, C)  # [H*W, C]
        
        # Convert to grid indices (same as occ_map.py approach)
        indices = self.pos_to_index(xy)
        
        # Filter out points outside the map bounds
        in_bounds_mask = self.is_in_map(indices)
        valid_indices = indices[in_bounds_mask]
        valid_feats = feats[in_bounds_mask]
        
        if valid_indices.size(0) == 0:
            return
        
        # find unique grid cells and mapping
        keys, inverse, counts = torch.unique(valid_indices, dim=0, return_inverse=True, return_counts=True)
        
        # sum features per cell
        feat_sums = torch.zeros((keys.shape[0], self.feature_dim), device=valid_feats.device)
        feat_sums = feat_sums.index_add(0, inverse, valid_feats)        
        # mean features per cell
        feat_means = feat_sums / counts.unsqueeze(1)        
        # indices for grid update
        i, j = keys[:, 0], keys[:, 1]
        
        # get previous counts for EMA mask
        prev_counts = self.grid_counts[i, j]
        ema_mask = prev_counts > 0
        
        # for cells with previous data, do EMA
        if ema_mask.any():
            self.grid_features[i[ema_mask], j[ema_mask]] = (
                ema_alpha * feat_means[ema_mask] + (1 - ema_alpha) * self.grid_features[i[ema_mask], j[ema_mask]]
            )        
        # for new cells, just assign
        if (~ema_mask).any():
            self.grid_features[i[~ema_mask], j[~ema_mask]] = feat_means[~ema_mask]
        
        # update counts
        self.grid_counts[i, j] += counts
        return

    def compute_relevancy(self, local_update=True):
        """Compute relevancy per cell."""
        if self._cached_prompt_embedding is None:
            return None
        text_feat = self._cached_prompt_embedding
        # convert text to tensor
        if isinstance(text_feat, np.ndarray):
            text_feat = torch.from_numpy(text_feat).to(self.device)
        if text_feat.ndim == 1:
            text_feat = text_feat.unsqueeze(0)
        
        if self.grid_features is None or self.grid_counts is None or self.grid_counts.sum() == 0:
            return None
        mask = self.grid_counts > 0
        if mask.sum() == 0:
            return None
        if local_update:
            # only update cells within update box
            min_x, max_x, min_y, max_y = self.update_box
            box_mask = torch.zeros_like(mask, dtype=torch.bool)
            box_mask[min_x:max_x, min_y:max_y] = True
            mask = box_mask & mask
        feats = self.grid_features[mask]
        norm_feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-8)
        norm_texts = text_feat / (text_feat.norm(dim=1, keepdim=True) + 1e-8)
        sims = torch.matmul(norm_feats, norm_texts.t())
        sims = torch.softmax(sims * 100, dim=1)
        sims = sims[:, :self.num_positive_classes].max(dim=1).values # take max over positive classes
        if self.relevancy_map is None:
            relevancy = torch.full(self.grid_counts.shape, float('nan'), device=self.device)
        else:
            relevancy = self.relevancy_map
        relevancy[mask] = sims
        return relevancy

    def get_relevancy_voxels(self):
        """Get relevancy voxels for visualization."""
        relevancy = self.relevancy_map
        if relevancy is None:
            return None, None
        
        points = []
        values = []
        relevancy_np = relevancy.cpu().numpy()
        ixs, iys = np.where(~np.isnan(relevancy_np))
        rels = relevancy_np[ixs, iys]
        xs = self.x_coords[ixs].cpu().numpy()
        ys = self.y_coords[iys].cpu().numpy()
        points = np.stack([xs, ys, np.zeros_like(xs)], axis=1)
        values = rels.astype(np.float32)

        return np.array(points, dtype=np.float32), np.array(values, dtype=np.float32)

    def get_grid_info(self, local_update=True):
        """
        Get grid information for publishing relevancy maps (compatible with utils.py).
        
        Returns:
            tuple: (origin, size, voxel_num, resolution, relevancy_data)
        """
        self.relevancy_map = self.compute_relevancy(local_update)
        relevancy = torch.clone(self.relevancy_map)
        if relevancy is None:
            return None

        relevancy_np = relevancy.cpu().numpy()
        relevancy_np = np.nan_to_num(relevancy_np, nan=-1.0)
        
        H, W = self.map_voxel_num.tolist()
        
        return (
            self.grid_origin.cpu().numpy(),
            self.map_size.cpu().numpy(),
            self.map_voxel_num.cpu().numpy(),
            self.grid_resolution,
            relevancy_np
        )

    def reset_map(self):
        """Reset the semantic grid map to initial state."""
        H, W = self.map_voxel_num.tolist()
        self.grid_features = torch.zeros((H, W, self.feature_dim), dtype=torch.float32, device=self.device)
        self.grid_counts   = torch.zeros((H, W), dtype=torch.long, device=self.device)
        self.relevancy_map = None
        self.global_points = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        self.global_colors = torch.empty((0, 3), dtype=torch.float32, device=self.device)
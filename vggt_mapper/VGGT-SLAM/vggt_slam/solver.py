import numpy as np
import cv2
import gtsam
import matplotlib.pyplot as plt
import torch
import open3d as o3d
import viser
import viser.transforms as viser_tf
from termcolor import colored

from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from vggt_slam.loop_closure import ImageRetrieval
from vggt_slam.frame_overlap import FrameTracker
from vggt_slam.map import GraphMap
from vggt_slam.submap import Submap
from vggt_slam.h_solve import ransac_projective
from vggt_slam.gradio_viewer import TrimeshViewer

def color_point_cloud_by_confidence(pcd, confidence, cmap='viridis'):
    """
    Color a point cloud based on per-point confidence values.
    
    Parameters:
        pcd (o3d.geometry.PointCloud): The point cloud.
        confidence (np.ndarray): Confidence values, shape (N,).
        cmap (str): Matplotlib colormap name.
    """
    assert len(confidence) == len(pcd.points), "Confidence length must match number of points"

    # Normalize confidence to [0, 1]
    confidence_normalized = (confidence - np.min(confidence)) / (np.ptp(confidence) + 1e-8)
    
    # Map to colors using matplotlib colormap
    colormap = plt.get_cmap(cmap)
    colors = colormap(confidence_normalized)[:, :3]  # Drop alpha channel

    # Assign to point cloud
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

class Viewer:
    def __init__(self, port: int = 8080):
        # print(f"Starting viser server on port {port}")

        # self.server = viser.ViserServer(host="0.0.0.0", port=port)
        # self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

        # # Global toggle for all frames and frustums
        # self.gui_show_frames = self.server.gui.add_checkbox(
        #     "Show Cameras",
        #     initial_value=True,
        # )
        # self.gui_show_frames.on_update(self._on_update_show_frames)

        # Store frames and frustums by submap
        self.submap_frames: Dict[int, List[viser.FrameHandle]] = {}
        self.submap_frustums: Dict[int, List[viser.CameraFrustumHandle]] = {}

        num_rand_colors = 250
        self.random_colors = np.random.randint(0, 256, size=(num_rand_colors, 3), dtype=np.uint8)

    def visualize_frames(self, extrinsics: np.ndarray, images_: np.ndarray, submap_id: int) -> None:
        """
        Add camera frames and frustums to the scene for a specific submap.
        extrinsics: (S, 3, 4)
        images_:    (S, 3, H, W)
        """

        if isinstance(images_, torch.Tensor):
            images_ = images_.cpu().numpy()

        if submap_id not in self.submap_frames:
            self.submap_frames[submap_id] = []
            self.submap_frustums[submap_id] = []

        S = extrinsics.shape[0]
        for img_id in range(S):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            frame_name = f"submap_{submap_id}/frame_{img_id}"
            frustum_name = f"{frame_name}/frustum"

            # Add the coordinate frame
            frame_axis = self.server.scene.add_frame(
                frame_name,
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frame_axis.visible = self.gui_show_frames.value
            self.submap_frames[submap_id].append(frame_axis)

            # Convert image and add frustum
            img = images_[img_id]
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            h, w = img.shape[:2]
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            frustum = self.server.scene.add_camera_frustum(
                frustum_name,
                fov=fov,
                aspect=w / h,
                scale=0.05,
                image=img,
                line_width=3.0,
                color=self.random_colors[submap_id]
            )
            frustum.visible = self.gui_show_frames.value
            self.submap_frustums[submap_id].append(frustum)

    def _on_update_show_frames(self, _) -> None:
        """Toggle visibility of all camera frames and frustums across all submaps."""
        visible = self.gui_show_frames.value
        for frames in self.submap_frames.values():
            for f in frames:
                f.visible = visible
        for frustums in self.submap_frustums.values():
            for fr in frustums:
                fr.visible = visible



class Solver:
    def __init__(self,
        init_conf_threshold: float,  # represents percentage (e.g., 50 means filter lowest 50%)
        use_point_map: bool = False,
        visualize_global_map: bool = False,
        use_sim3: bool = False,
        gradio_mode: bool = False,
        enable_loop_closure: bool = True,
        enable_icp: bool = False,
        submap_size=5,
        overlap_window_size=1):
        
        self.init_conf_threshold = init_conf_threshold
        self.use_point_map = use_point_map
        self.gradio_mode = gradio_mode
        self.enable_loop_closure = enable_loop_closure
        self.enable_icp = enable_icp

        if self.gradio_mode:
            self.viewer = TrimeshViewer()
        else:
            self.viewer = Viewer()

        self.flow_tracker = FrameTracker()
        self.map = GraphMap()
        self.use_sim3 = use_sim3
        if self.use_sim3:
            from vggt_slam.graph_se3 import PoseGraph
        else:
            from vggt_slam.graph import PoseGraph
        self.graph = PoseGraph()

        if self.enable_loop_closure:
            self.image_retrieval = ImageRetrieval()
        self.current_working_submap = None

        self.first_edge = True

        self.T_w_kf_minus = None

        self.prior_pcd = None
        self.prior_conf = None
        self.icp_prior_pts = None

        self.submap_size = submap_size
        self.overlap_window_size = overlap_window_size

        print("Starting viser server...")

    def set_point_cloud(self, points_in_world_frame, points_colors, name, point_size):
        if self.gradio_mode:
            self.viewer.add_point_cloud(points_in_world_frame, points_colors)
        else:
            self.viewer.server.scene.add_point_cloud(
                name="pcd_"+name,
                points=points_in_world_frame,
                colors=points_colors,
                point_size=point_size,
                point_shape="circle",
            )

    def set_submap_point_cloud(self, submap):
        # Add the point cloud to the visualization.
        points_in_world_frame = submap.get_points_in_world_frame()
        points_colors = submap.get_points_colors()
        name = str(submap.get_id())
        self.set_point_cloud(points_in_world_frame, points_colors, name, 0.001)

    def set_submap_poses(self, submap):
        # Add the camera poses to the visualization.
        extrinsics = submap.get_all_poses_world()
        if self.gradio_mode:
            for i in range(extrinsics.shape[0]):
                self.viewer.add_camera_pose(extrinsics[i])
        else:
            images = submap.get_all_frames()
            self.viewer.visualize_frames(extrinsics, images, submap.get_id())

    def export_3d_scene(self, output_path="output.glb"):
        return self.viewer.export(output_path)

    def update_all_submap_vis(self):
        for submap in self.map.get_submaps():
            self.set_submap_point_cloud(submap)
            self.set_submap_poses(submap)

    def update_latest_submap_vis(self):
        submap = self.map.get_latest_submap()
        self.set_submap_point_cloud(submap)
        self.set_submap_poses(submap)

    def add_points(self, pred_dict):
        """
        Args:
            pred_dict (dict):
            {
                "images": (S, 3, H, W)   - Input images,
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
            }
        """
        # Unpack prediction dict
        images = pred_dict["images"]  # (S, 3, H, W)

        extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
        intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)
        # print(intrinsics_cam)

        detected_loops = pred_dict["detected_loops"]

        if self.use_point_map:
            world_points_map = pred_dict["world_points"]  # (S, H, W, 3)
            conf = pred_dict["world_points_conf"]  # (S, H, W)
            world_points = world_points_map
        else:
            depth_map = pred_dict["depth"]  # (S, H, W, 1)
            conf = pred_dict["depth_conf"]  # (S, H, W)
            if len(depth_map.shape) == 3:
                depth_map = np.expand_dims(depth_map, axis=-1)
            world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
        
        # scale the world points
        scale = pred_dict["scale"]
        world_points *= scale

        # now override the extrinsics
        # extrinsics_cam = pred_dict["fused_extrinsics"]
        extrinsics_cam = pred_dict["scaled_extrinsics"] # w2c
        
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(world_points.reshape(-1, 3))
        # pcd.colors = o3d.utility.Vector3dVector((images.transpose(0, 2, 3, 1) * 255).reshape(-1, 3).astype(np.uint8) / 255.0)
        # # pcd = color_point_cloud_by_confidence(pcd, conf.reshape(-1), cmap='viridis')
        # o3d.visualization.draw_geometries([pcd], window_name="Point Cloud")

        # Convert images from (S, 3, H, W) to (S, H, W, 3)
        # Then flatten everything for the point cloud
        colors = (images.transpose(0, 2, 3, 1) * 255).astype(np.uint8)  # now (S, H, W, 3)

        # Flatten
        cam_to_world = closed_form_inverse_se3(extrinsics_cam)  # shape (S, 4, 4) c2w

        # estimate focal length from points
        points_in_first_cam = world_points[0,...]
        h, w = points_in_first_cam.shape[0:2]

        new_pcd_num = self.current_working_submap.get_id()
        if self.first_edge:
            self.first_edge = False
            self.prior_pcd = world_points[-1,...].reshape(-1, 3)
            self.prior_conf = conf[-1,...].reshape(-1)
            self.prior_colors = colors[-1,...].reshape(-1, 3)

            # Add node to graph.
            H_w_submap = np.eye(4)
            self.graph.add_homography(new_pcd_num, H_w_submap)
            self.graph.add_prior_factor(new_pcd_num, H_w_submap, self.graph.anchor_noise)
        else:
            prior_pcd_num = self.map.get_largest_key()
            prior_submap = self.map.get_submap(prior_pcd_num)

            current_pts = world_points[0,...].reshape(-1, 3)
        
            # TODO conf should be using the threshold in its own submap
            good_mask = self.prior_conf > prior_submap.get_conf_threshold() * (conf[0,...,:].reshape(-1) > prior_submap.get_conf_threshold())
            
            if self.use_sim3:
                # Note we still use H and not T in variable names so we can share code with the Sim3 case, 
                # and SIM3 and SE3 are also subsets of the SL4 group
                R_temp = prior_submap.poses[-self.overlap_window_size][0:3,0:3] # c2w
                t_temp = prior_submap.poses[-self.overlap_window_size][0:3,3]
                T_temp = np.eye(4)
                T_temp[0:3,0:3] = R_temp
                T_temp[0:3,3] = t_temp
                T_temp = np.linalg.inv(T_temp)
                
                H_relative = np.eye(4)
                H_relative[0:3,0:3] = R_temp
                H_relative[0:3,3] = t_temp

                if new_pcd_num >= 2:
                    prior2_pcd_num = self.map.get_largest_key() - 1
                    prior2_submap = self.map.get_submap(prior2_pcd_num)
                    window2_size = self.submap_size - self.overlap_window_size - 1
                    R_temp2 = prior2_submap.poses[-window2_size][:3,:3]
                    t_temp2 = prior2_submap.poses[-window2_size][:3,3]
                    H2_relative = np.eye(4)
                    H2_relative[0:3,0:3] = R_temp2
                    H2_relative[0:3,3] = t_temp2

            else:
                H_relative = ransac_projective(current_pts[good_mask], self.prior_pcd[good_mask])
            
            H_w_submap = prior_submap.get_reference_homography() @ H_relative

            self.icp_prior_pts = self.prior_pcd
            self.icp_prior_colors = self.prior_colors
            self.icp_current_pts = current_pts
            self.icp_current_colors = colors[0,...].reshape(-1, 3)

            non_lc_frame = self.current_working_submap.get_last_non_loop_frame_index()
            pts_cam0_camn = world_points[non_lc_frame,...].reshape(-1, 3)
            self.prior_pcd = pts_cam0_camn
            self.prior_colors = colors[non_lc_frame,...].reshape(-1, 3)
            self.prior_conf = conf[non_lc_frame,...].reshape(-1)

            # Add node to graph.
            self.graph.add_homography(new_pcd_num, H_w_submap)

            # Add between factor.
            self.graph.add_between_factor(prior_pcd_num, new_pcd_num, H_relative, self.graph.relative_noise)

        # Create and add submap.
        self.current_working_submap.set_reference_homography(H_w_submap)
        self.current_working_submap.add_all_poses(cam_to_world)
        self.current_working_submap.add_all_points(world_points, colors, conf, self.init_conf_threshold, intrinsics_cam)
        self.current_working_submap.set_conf_masks(conf) # TODO should make this work for point cloud conf as well

        # Add Colored ICP-based constraint between submaps, if enabled
        if not self.first_edge and self.enable_icp and self.icp_prior_pts is not None:
            try:
                # Build Open3D point clouds in WORLD frame (already transformed by submap H_w_map)
                pcd_prior = o3d.geometry.PointCloud()
                pcd_prior.points = o3d.utility.Vector3dVector(self.icp_prior_pts)
                pcd_prior.colors = o3d.utility.Vector3dVector(self.icp_prior_colors / 255.0)

                pcd_current = o3d.geometry.PointCloud()
                pcd_current.points = o3d.utility.Vector3dVector(self.icp_current_pts)
                pcd_current.colors = o3d.utility.Vector3dVector(self.icp_current_colors / 255.0)

                # Multi-scale colored ICP with small motion clamps
                # voxel_radius = [0.4, 0.3, 0.2]
                voxel_radius = [0.2]
                max_iter = [30, 20, 10]
                rot_limits_deg = [10.0, 5.0, 2.0]   # per-level max rotation update
                trans_limits = [5.0, 2.0, 1.0]   # per-level max translation update (meters)

                current_transformation = H_relative.copy() # T_c_p
                best_fitness = -1.0
                best_rmse = np.inf
                
                for level in range(len(voxel_radius)):
                    radius = voxel_radius[level]
                    iterations = max_iter[level]

                    # maps source -> target: T_t_s == T_p_c
                    # so target is prior, source is current
                    target_down = pcd_prior.voxel_down_sample(radius)
                    source_down = pcd_current.voxel_down_sample(radius)

                    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
                    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

                    result = o3d.pipelines.registration.registration_colored_icp(
                        source_down,
                        target_down,
                        radius,
                        current_transformation,
                        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                        o3d.pipelines.registration.ICPConvergenceCriteria(
                            relative_fitness=1e-1,
                            relative_rmse=1,
                            max_iteration=iterations,
                        ),
                    )

                    T_refined = result.transformation

                    # Accept only if metrics do not degrade badly
                    fitness_ok = (result.fitness >= best_fitness - 1e-6)
                    rmse_ok = (result.inlier_rmse <= best_rmse * 1.25)

                    if fitness_ok and rmse_ok:
                        current_transformation = T_refined
                        best_fitness = max(best_fitness, result.fitness)
                        best_rmse = min(best_rmse, result.inlier_rmse)
                    else:
                        # Do not accept this level's update; keep previous
                        pass

                # current_transformation is prior -> current
                self.graph.add_between_factor(prior_pcd_num, new_pcd_num, current_transformation, self.graph.relative_noise)
            except Exception as e:
                print(colored(f"Colored ICP error: {e}", "red"))

        # Add in loop closures if any were detected.
        for index, loop in enumerate(detected_loops):
            assert loop.query_submap_id == self.current_working_submap.get_id()

            loop_index = self.current_working_submap.get_last_non_loop_frame_index() + index + 1

            if self.use_sim3:
                pose_world_detected = self.map.get_submap(loop.detected_submap_id).get_pose_subframe(loop.detected_submap_frame)
                pose_world_query = self.current_working_submap.get_pose_subframe(loop_index)
                pose_world_detected = gtsam.Pose3(pose_world_detected)
                pose_world_query = gtsam.Pose3(pose_world_query)
                H_relative_lc = pose_world_detected.between(pose_world_query).matrix()
            else:
                points_world_detected = self.map.get_submap(loop.detected_submap_id).get_frame_pointcloud(loop.detected_submap_frame).reshape(-1, 3)
                points_world_query = self.current_working_submap.get_frame_pointcloud(loop_index).reshape(-1, 3)
                H_relative_lc = ransac_projective(points_world_query, points_world_detected)


            self.graph.add_between_factor(loop.detected_submap_id, loop.query_submap_id, H_relative_lc, self.graph.relative_noise)
            self.graph.increment_loop_closure() # Just for debugging and analysis, keep track of total number of loop closures

            print("added loop closure factor", loop.detected_submap_id, loop.query_submap_id, H_relative_lc)
            print("homography between nodes estimated to be", np.linalg.inv(self.map.get_submap(loop.detected_submap_id).get_reference_homography()) @ H_w_submap)

        self.map.add_submap(self.current_working_submap)


    def sample_pixel_coordinates(self, H, W, n):
        # Sample n random row indices (y-coordinates)
        y_coords = torch.randint(0, H, (n,), dtype=torch.float32)
        # Sample n random column indices (x-coordinates)
        x_coords = torch.randint(0, W, (n,), dtype=torch.float32)
        # Stack to create an (n,2) tensor
        pixel_coords = torch.stack((y_coords, x_coords), dim=1)
        return pixel_coords

    def run_predictions(self, image_names, model, max_loops, ext_cands=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        images = load_and_preprocess_images(image_names).to(device) # (N, 3, H, W)

        # print("Running inference...")
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        # Prepare new submap
        new_pcd_num = self.map.get_largest_key() + 1
        new_submap = Submap(new_pcd_num)
        new_submap.add_all_frames(images.detach().cpu())
        new_submap.set_frame_ids(image_names)

        # On-the-fly loop closure via external pose candidates, if provided        
        detected_loops = []
        if self.enable_loop_closure and len(ext_cands) > 0:
            # Build retrieval vectors on-the-fly
            # for new submap
            new_submap.set_all_retrieval_vectors(
                self.image_retrieval.get_all_submap_embeddings(new_submap)
            )
            # for each external candidate submap only
            candidate_ids = []
            if ext_cands:
                for cid in ext_cands:
                    candidate_ids.append(cid)
                    print('cid', cid)
                    sub = self.map.get_submap(cid)
                    print('sub', sub)
                    sub.set_all_retrieval_vectors(
                        self.image_retrieval.get_all_submap_embeddings(sub)
                    )
                    print('set retrieval vectors for submap', cid)
            # Run original loop closure detection
            detected_loops = self.image_retrieval.find_loop_closures(
                self.map, new_submap, candidate_ids, max_loop_closures=max_loops
            )
            print(f"Detected {len(detected_loops)} loop closures with external candidates: {ext_cands}")
        if len(detected_loops) > 0:
            print(colored("detected_loops", "yellow"), detected_loops)
        retrieved_frames = self.map.get_frames_from_loops(detected_loops)

        num_loop_frames = len(retrieved_frames)
        new_submap.set_last_non_loop_frame_index(images.shape[0] - 1)
        if num_loop_frames > 0:
            image_tensor = torch.stack(retrieved_frames).cuda()  # Shape (n, 3, w, h)
            images = torch.cat([images, image_tensor], dim=0) # Shape (s+n, 3, w, h)

            # TODO we don't really need to store the loop closure frame again, but this makes lookup easier for the visualizer.
            # We added the frame to the submap once before to get the retrieval vectors,
            new_submap.add_all_frames(images.detach().cpu())

            print("Images shape after adding loop closure frames:", images.shape)

        self.current_working_submap = new_submap

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic        
        
        predictions["detected_loops"] = detected_loops

        for key in predictions.keys():
            # remove batch dimension and convert to numpy
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze()

        # Clear GPU cache to prevent memory accumulation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return predictions
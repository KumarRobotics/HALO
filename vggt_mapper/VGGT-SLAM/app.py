import os
import zipfile
import glob
import shutil
import numpy as np
import torch
import gradio as gr
import cv2
from tqdm import tqdm

import vggt_slam.slam_utils as utils
from vggt_slam.solver import Solver
from vggt.models.vggt import VGGT


def run_slam(
    image_zip,
    use_sim3=False,
    submap_size=16,
    max_loops=1,
    min_disparity=50.0,
    conf_threshold=25.0
):
    # Handle zip from Gradio
    zip_path = image_zip.name

    # Clean and extract
    tmp_dir = "temp_images"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(tmp_dir)
        print("Extracted files:", zip_ref.namelist())

    # Recursive glob
    image_paths = [
        f for f in glob.glob(os.path.join(tmp_dir, "**", "*"), recursive=True)
        if "depth" not in f.lower() and f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    image_paths = utils.sort_images_by_number(image_paths)

    use_optical_flow_downsample = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    solver = Solver(
        init_conf_threshold=conf_threshold,
        use_point_map=False,
        use_sim3=use_sim3,
        gradio_mode=True
    )

    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)

    image_names_subset = []
    for image_name in tqdm(image_paths):
        if use_optical_flow_downsample:
            img = cv2.imread(image_name)
            enough_disparity = solver.flow_tracker.compute_disparity(img, min_disparity, False)
            if enough_disparity:
                image_names_subset.append(image_name)
        else:
            image_names_subset.append(image_name)

        # Run submap
        if len(image_names_subset) == submap_size + 1 or image_name == image_paths[-1]:
            print(image_names_subset)
            predictions = solver.run_predictions(image_names_subset, model, max_loops)
            solver.add_points(predictions)
            solver.graph.optimize()
            solver.map.update_submap_homographies(solver.graph)

            image_names_subset = image_names_subset[-1:]

    solver.update_all_submap_vis()

    num_submaps = solver.map.get_num_submaps()
    num_loops = solver.graph.get_num_loops()
    glb_path = solver.export_3d_scene()
    message = f"VGGT-SLAM completed with {num_submaps} submaps and {num_loops} loop closures."
    return glb_path, message


# Setup Gradio outputs
model_output = gr.Model3D(label="🗺️ Reconstructed 3D Map", height=600)
status_output = gr.Textbox(label="", show_label=False) 

demo = gr.Interface(
    fn=run_slam,
    inputs=[
        gr.File(label="Upload .zip of images", file_types=[".zip"]),
        gr.Checkbox(label="Use Sim3", value=False),
        gr.Slider(4, 32, value=16, step=1, label="Submap Size"),
        gr.Slider(0, 5, value=1, step=1, label="Max potential loop closures to add for each new submap"),
        gr.Slider(0.0, 100.0, value=50.0, step=1.0, label="Minimum disparity between keyframes"),
        gr.Slider(0.0, 100.0, value=25.0, step=1.0, label="Confidence Threshold (increasing will decrease number of points)"),
    ],
    outputs=[model_output, status_output],
    examples=[["office_loop.zip"]],
    title="VGGT-SLAM Demo",
    description=(
        "We've prepared a simple demo of VGGT-SLAM on Hugging Face with an example scene. Our github repo contains a more powerful visualization which shows incremental scene reconstruction. \n\n"
        "To try your own scene, upload a ZIP of RGB images and run VGGT-SLAM to reconstruct the scene.\n\n"
        "Outputs a 3D point cloud and estimated camera poses.\n\n"
        "⏱️ **Estimated runtime for provided example scene: <1 minute**"
    ),
    allow_flagging="never",
    theme="default"
)

if __name__ == "__main__":
    demo.launch()

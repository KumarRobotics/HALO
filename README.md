<!-- PROJECT HEADER -->

<div align="center">

# HALO: Language-Conditioned Overhead Monocular Aerial Exploration and Navigation

### IEEE Robotics and Automation Letters (RA-L), 2026

<p>
  <a href="https://tyuezhan.github.io/"><strong>Yuezhan Tao*</strong></a> ·
  <a href="https://dexterong.com/"><strong>Dexter Ong*</strong></a> ·
  <a href="https://fcladera.com/"><strong>Fernando Cladera</strong></a> ·
  <a href="https://jhughes50.github.io/index.html"><strong>Jason Hughes</strong></a> ·
  <a href="https://www.cis.upenn.edu/~cjtaylor/home.html"><strong>Camillo J. Taylor</strong></a> ·
  <a href="https://pratikac.github.io/"><strong>Pratik Chaudhari</strong></a> ·
  <a href="https://www.kumarrobotics.org/"><strong>Vijay Kumar</strong></a>
</p>

<p><i>* Equal contribution</i></p>

### [📄 Paper](https://arxiv.org/abs/2511.17497) | [🎥 Video](https://www.youtube.com/watch?v=MB65Bz4iBXI) | [🌐 Project Page](https://tyuezhan.github.io/halo/)

</div>

---

<p align="center">
  <img src="./assets/platform.gif" width="47%">
  <img src="./assets/realworld.gif" width="47%">
</p>

---

## Overview

HALO is a language-conditioned autonomous aerial exploration and navigation system designed for overhead monocular UAV operation. The system enables a robot to explore unknown environments, build semantic maps, and search for user-specified targets using natural language prompts.

This repository provides the official release, including simulation tools, ROS integration, configuration files, and deployment utilities.

---

## Repository Contents

This release includes:

- Source code
- ROS launch files
- Configuration files
- Utility scripts
- tmux session scripts for simulation and robot deployment
- Dockerfiles
- Unity simulation binaries

---

## System Requirements

This repository has been tested with:

- Ubuntu 22.04
- ROS 2 Humble

---

## Quick Start for Simulation

We provide Docker scripts for a quick simulation setup.

### 1. Build the Docker Image

```bash
./build_sim.sh
```

### 2. Start the Docker Container

```bash
./run_sim.sh
```

---

## Simulation Binary

Please download the Unity simulation binaries from the following link:

[HALO Binary](https://drive.google.com/file/d/1_nw0bDuZw8NvNTspxNpvoEgkRWhX6SRF/view?usp=sharing)

After downloading, unzip the binaries and copy them into the Docker container:

```
docker cp aerial_sim_binaries/ air_sem_exp:/root/ros2_ws/src/
```

---

## Running the Simulation with ROS

The provided tmux script launches the full simulation stack using `tmuxp`.

```bash
cd ros2_ws/src/air_sem_explorer/tmux
tmuxp load sim.yaml
```

This launches:

- Zenoh router
- Unity simulator
- Planner node
- Geometric mapper node
- Semantic mapper node
- Simulation tracker node

To trigger exploration, go to the `Trigger` tmux tab and press `Enter`.

To shut down all processes, go to the final `kill` tmux tab and press `Enter`.

---

## Configuration

### Change the Map Boundary

Edit:

```bash
air_sem_explorer/config/map_common.yaml
```

### Change the Default Task Prompt or Retask the Robot

Edit:

```bash
air_sem_gridmap/config/config.yaml
```

Use:

```bash
ros2 service call /set_prompt air_sem_gridmap_interfaces/srv/SetPrompt "{prompt: '<prompt1,prompt2>'}"
```

When using the tmux script, a GUI window will also appear to allow you to set the task prompt.

---

## Acknowledgements

We thank the authors of the following repositories for their open-source code:

- [RayFronts](https://github.com/RayFronts/RayFronts)
- [RADIO](https://github.com/nvlabs/radio)
- [VGGT](https://github.com/facebookresearch/vggt)
- [VGGT-SLAM](https://github.com/MIT-SPARK/VGGT-SLAM)

---

## Citation

If you find our paper or code useful, please consider citing:

```bibtex
@article{tao2025halo,
  title={HALO: High-Altitude Language-Conditioned Monocular Aerial Exploration and Navigation},
  author={Tao, Yuezhan and Ong, Dexter and Cladera, Fernando and Hughes, Jason and Taylor, Camillo J and Chaudhari, Pratik and Kumar, Vijay},
  journal={arXiv preprint arXiv:2511.17497},
  year={2025}
}
```
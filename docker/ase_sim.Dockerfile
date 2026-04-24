# Use the official NVIDIA CUDA 12.5 devel image (Ubuntu 22.04)
FROM nvidia/cuda:12.5.0-devel-ubuntu22.04

# set non-interactive mode for apt-get
ARG DEBIAN_FRONTEND=noninteractive

# Set environment variables for CUDA and GPU support
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all

# revert to official pip index
ENV PIP_INDEX_URL=https://pypi.org/simple

# ros
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt upgrade -y
RUN apt install software-properties-common -y
RUN add-apt-repository universe -y
RUN apt update
RUN apt install -y \
    build-essential \
    cmake \
    tmux \
    git \
    gdb \
    wget \
    curl \
    vim \
    ca-certificates 

RUN apt-get update && apt-get install -y --no-install-recommends curl gnupg2 \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" > /etc/apt/sources.list.d/ros2.list \
    && rm -rf /var/lib/apt/lists/*

RUN apt update && apt install -y ros-humble-ros-base
RUN apt install -qy python3-rosdep \
    python3-colcon-common-extensions \
    libgflags-dev \
    python3-vcstool \
    libeigen3-dev \
    bash-completion \
    openssh-client \
    python3-argcomplete \
    ros-dev-tools \
    ros-humble-ament-* \
    libasio-dev \
    ros-humble-rtcm-msgs \
    ros-humble-rmw-zenoh-cpp \
    ros-humble-rosbag2-storage-mcap \
    ros-humble-gtsam \
    libgoogle-glog-dev \
    ros-humble-point-cloud-transport \
    ros-humble-rviz2 \
    zstd
RUN apt install python3-pip -yq
RUN pip uninstall cmake -y
RUN pip install cmake==3.27.0
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# git
RUN apt -yqq install ssh
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts
# workspace
WORKDIR /root/ros2_ws
RUN rosdep init
RUN rosdep update
RUN mkdir -p /root/ros2_ws/src

# repos
RUN apt update
WORKDIR /root/ros2_ws/src
RUN --mount=type=ssh git clone git@github.com:KumarRobotics/HALO.git
RUN --mount=type=ssh git clone https://github.com/berndpfrommer/rosbag2_composable_recorder

WORKDIR /root/ros2_ws/src/HALO
RUN pip install -r requirements.txt
RUN git submodule update --init --recursive

WORKDIR /root/ros2_ws/src/HALO/vggt_mapper/VGGT-SLAM
RUN sudo apt remove python3-blinker -qy
RUN python3 -m pip install --upgrade pip
RUN sh setup.sh

# build
WORKDIR /root/ros2_ws
RUN . /opt/ros/humble/setup.sh && rosdep install --from-paths src --ignore-src -r -y
RUN . /opt/ros/humble/setup.sh && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release --parallel-workers $(nproc)

WORKDIR /root/

RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "export AIR_SEM_EXPLORER_WS=/root/ros2_ws/" >> ~/.bashrc
# RUN echo "export RMW_IMPLEMENTATION=rmw_zenoh_cpp" >> ~/.bashrc
# RUN echo "source /root/ros2_ws/install/local_setup.bash" >> ~/.bashrc
SHELL ["/bin/bash", "-c"]

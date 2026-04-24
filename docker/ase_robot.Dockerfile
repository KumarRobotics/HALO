FROM dustynv/pytorch:2.7-r36.4.0 as base
# FROM dustynv/pytorch:2.1-r36.2.0

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
    libgoogle-glog-dev \
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
    ros-humble-point-cloud-transport \
    zstd

# install ZED
RUN wget https://download.stereolabs.com/zedsdk/5.0/l4t36.4/jetsons && \
         chmod +x jetsons && \
         ./jetsons -- silent \
         && rm jetsons

RUN pip uninstall cmake -y
RUN pip install cmake==3.27.0

# git
RUN apt -yqq install ssh
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts
# workspace
WORKDIR /root/ros2_ws
RUN rosdep init
RUN rosdep update
RUN mkdir -p /root/ros2_ws/src

# separate this so we can rebuild deps without cache
FROM base as rebuild_robot
# repos
RUN apt update
WORKDIR /root/ros2_ws/src
RUN --mount=type=ssh git clone git@github.com:KumarRobotics/HALO.git
RUN --mount=type=ssh git clone git@github.com:KumarRobotics/glider.git
RUN --mount=type=ssh git clone -b ros2 git@github.com:KumarRobotics/ublox.git
RUN --mount=type=ssh git clone -b ros2 git@github.com:KumarRobotics/vectornav.git
RUN --mount=type=ssh git clone https://github.com/stereolabs/zed-ros2-wrapper.git
RUN --mount=type=ssh git clone -b humble https://github.com/ros2/rosbag2.git
RUN --mount=type=ssh git clone https://github.com/berndpfrommer/rosbag2_composable_recorder

# build
WORKDIR /root/ros2_ws
RUN . /opt/ros/humble/setup.sh && rosdep install --from-paths src --ignore-src -r -y
RUN . /opt/ros/humble/setup.sh && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release --parallel-workers $(nproc)

WORKDIR /root/ros2_ws/src/HALO
RUN pip install -r requirements.txt

WORKDIR /root/ros2_ws/src/HALO/vggt_mapper/VGGT-SLAM
RUN sudo apt remove python3-blinker -qy
RUN sh setup.sh

WORKDIR /root/

RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "export ROS_DOMAIN_ID=21" >> ~/.bashrc
RUN echo "export RMW_IMPLEMENTATION=rmw_zenoh_cpp" >> ~/.bashrc
# RUN echo "source /root/ros2_ws/install/local_setup.bash" >> ~/.bashrc
# RUN echo "export AIR_SEM_EXPLORER_WS=/root/ros2_ws/" >> ~/.bashrc
SHELL ["/bin/bash", "-c"]

# separate this to pull repos and rebuild workspace without cache
FROM rebuild_robot as update_robot
RUN --mount=type=ssh cd /root/ros2_ws/src/HALO && git pull && git log
RUN . /opt/ros/humble/setup.sh && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release --parallel-workers $(nproc)
#!/bin/bash

docker run -it --runtime=nvidia \
    --gpus=all \
    --network=host \
    --privileged \
    -v "/dev:/dev" \
    -v /mnt/d:/mnt/d \
    -v /mnt/e:/mnt/e \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e XAUTHORITY=$XAUTH \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /home/odexter/.Xauthority:/root/.Xauthority:rw \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    --name air_sem_exp \
    air-sem-exp-sim \
    bash

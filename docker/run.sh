#!/bin/bash

docker run -it --runtime=nvidia \
    --gpus=all \
    --network=host \
    --privileged \
    -v "/dev:/dev" \
    -v /tmp/.X11-unix/:/tmp/.X11-unix/ \
    -v /home/${USER}/.Xauthority:/root/.Xauthority \
    -v /home/dcist/ase/bags:/bags \
    -v /home/dcist/ase/zed_resources:/usr/local/zed/resources \
    -v /home/dcist/ase/torchhub_data:/data \
    -v /home/dcist/ase/huggingface_cache:/root/.cache/huggingface \
    -v /home/dcist/ase/ase_ws:/root/ase_ws \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e XAUTHORITY=$XAUTH \
    --name air_sem_exp \
    air-sem-exp \
    bash

#!/bin/bash

REBUILD_ROBOT=false
UPDATE_ROBOT=false

# parse arg
if [ $# -eq 1 ]; then
  if [ "$1" = "--rebuild" ]; then
    REBUILD_ROBOT=true
  elif [ "$1" = "--update" ]; then
    UPDATE_ROBOT=true
  else
    echo "Usage: $0 [--rebuild|--update]"
    exit 1
  fi
fi

if $REBUILD_ROBOT; then
  echo "Rebuilding robot..."
  docker build --no-cache-filter rebuild_robot --ssh default --target rebuild_robot -t air-sem-exp -f ase_robot.Dockerfile .
elif $UPDATE_ROBOT; then
  echo "Updating robot..."
  docker build --no-cache-filter update_robot --ssh default --target update_robot -t air-sem-exp -f ase_robot.Dockerfile .
else
  docker build --ssh default --target base -t air-sem-exp-base -f ase_robot.Dockerfile .
  docker build --ssh default --target rebuild_robot -t air-sem-exp -f ase_robot.Dockerfile .
fi
echo "Build complete."
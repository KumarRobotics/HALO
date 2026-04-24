#!/bin/bash
docker build --ssh default -t air-sem-exp-sim -f ase_sim.Dockerfile .
echo "Build complete."
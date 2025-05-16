#!/bin/bash

set -x
set -e

NUM_CORES=$(awk '/^Core\(s\) per socket:/ {cores=$4} /^Socket\(s\):/ {sockets=$2} END {print cores * sockets}' < <(lscpu))
NUM_CORES=$((NUM_CORES - 1))

taskset -pc "$NUM_CORES" $$

which opt

RAY_custom_unit_instance_resources=physical_core \
    PYTHONPATH=$PYTHONPATH:. \
    python3 -m compiler_opt.sl.unrolling.generate_unroll_training_samples \
    --dataset ~/datasets/ComPileLoopInputs.sqlite3 \
    --output-dataset ~/datasets/UnrollTrainingSamples.sqlite3

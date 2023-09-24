#!/bin/bash

# Define your commands
commands=("blenderproc run renderingParallel.py -i renderingGroupData/000000.pkl" "blenderproc run renderingParallel.py -i renderingGroupData/000001.pkl" "blenderproc run renderingParallel.py -i renderingGroupData/000002.pkl" "blenderproc run renderingParallel.py -i renderingGroupData/000003.pkl")

# Iterate over the commands and run them in the background
for cmd in "${commands[@]}"; do
    $cmd &
done

# Wait for all background processes to finish
wait

echo -e "finished all processes!"

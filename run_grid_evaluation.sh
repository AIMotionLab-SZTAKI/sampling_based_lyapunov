#!/bin/bash

for i in {0..23}
do
 gnome-terminal --tab -- bash -c "cd /home/aimotion-i9/Projects/sampling-based-lyapunov && source venv/bin/activate && python3 sampling_based_lyapunov/system_models.py '$i'; exec bash"
done
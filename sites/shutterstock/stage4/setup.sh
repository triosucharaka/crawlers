#!/bin/bash

# No input
export DEBIAN_FRONTEND=noninteractive

# Package updates and installations
sudo apt update && sudo apt install ffmpeg libsm6 libxext6 screen wget nano -y

# Install specific Python packages
pip3 install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip3 install -r requirements.txt

echo 'Script completed.'

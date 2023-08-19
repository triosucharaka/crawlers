#!/bin/bash

# Package updates and installations
sudo apt update && sudo apt install ffmpeg libsm6 libxext6 screen wget nano git-lfs -y

# Install specific Python packages
python3 -m pip install torch~=2.0.0 https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-2.0-cp38-cp38-linux_x86_64.whl
python3 -m pip install pillow opencv-python

# Set up git Large File Storage (LFS) and clone the repos
git lfs install
git clone https://github.com/chavinlo/im2im
git clone https://huggingface.co/lopho/im2im-sswm

# Set environment variables
echo 'export XRT_TPU_CONFIG="localservice;0;localhost:51011"' >> ~/.bashrc
echo 'export TPU_NUM_DEVICES=4' >> ~/.bashrc

# Source bashrc to make new variables available immediately
source ~/.bashrc

echo 'Environment variables set and script completed.'

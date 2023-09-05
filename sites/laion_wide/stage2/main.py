# Based on shutterstock/stage4

import multiprocessing as mp
import time
import json
import wandb
import os
import numpy as np
import io
import tempfile
import gc
import logging
import psutil
import cv2
import threading
import traceback
import jax.numpy as jnp
import numpy as np
import jax
import sqlite3
import argparse
from diffusers import FlaxAutoencoderKL
from wrapt_timeout_decorator import *

mp.set_start_method("spawn", force=True)

# get instance id from args
parser = argparse.ArgumentParser()
parser.add_argument("--instance", type=int, required=True)
args = parser.parse_args()
INSTANCE = args.instance

### Configuration ###

## Paths
SQL_DB_PATH = "database.db"
OUT_DISK_PATH = "/home/windowsuser/mount-folder/tempofunkds/shutterstock/stage4/"

NPY_EXTENSION = "frames" # ex.: 00001.frames (not .npy, to differentiate with the text embeds on stage 5)

## Wandb
global USE_WANDB
USE_WANDB = True
WANDB_ENTITY = "peruano"  # none if not using wandb
WANDB_PROJ = "laion_wide" 
WANDB_NAME = f"stage2_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
LOG_MEMORY = True

## Multiprocessing
ASSIGN_WORKER_COUNT = 4
ASSIGN_MTPC = 10
TAR_WORKER_COUNT = 4
TAR_MTPC = 40

## TPU Workers
TPU_CORE_COUNT = 4
TPU_BATCH_SIZE = 64

## Model Parameters
VAE_MODEL_PATH = "CompVis/stable-diffusion-v1-4"
VAE_MODEL_REVISION = "flax"
VAE_MODEL_SUBFOLDER = "vae"
C_C = 3
C_H = 384  # (divisible by 64)
C_W = 640  # (divisible by 64)

## Pipes
FILE_PIPE_MAX = 100
IN_DATA_PIPE_MAX = 400
OUT_DATA_PIPE_MAX = 400
TAR_PIPE_MAX = 200
WANDB_PIPE_MAX = 1000

### End Configuration ###

assert C_C == 3, "C_C must be 3, for RGB"
assert C_H % 64 == 0, "C_H must be divisible by 64"
assert C_W % 64 == 0, "C_W must be divisible by 64"


def get_memory():
    mem_info = psutil.virtual_memory()
    free_memory = f"SMU: {round(mem_info.used / 1024 ** 2, 2)}MB; {mem_info.percent}"
    return free_memory


class CustomLogger(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return f"{get_memory()} - {msg}", kwargs

logger = logging.getLogger(__name__)
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('file.log')
c_format = logging.Formatter(f'%(asctime)s - %(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter(f'%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)
logger.addHandler(c_handler)
logger.addHandler(f_handler)
logger.setLevel(logging.DEBUG)
logger = CustomLogger(logger, {})

if LOG_MEMORY:
    logger = CustomLogger(logger, {})

def sql_reader_func(sql_pipe: mp.Queue):
    logger.info("SQL-R: Started")

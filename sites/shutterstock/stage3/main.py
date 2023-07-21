import multiprocessing as mp
import time
import json
import os
import math
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch
from im2im.main import load_model
import numpy as np
import torchvision
import torchvision.transforms.functional as F
import wandb
import copy

"""
v1 - wonderhoy
"""
CODENAME = "wonderhoy"

# config

## General
DISK_PATH = "/mnt/disks/hhd/tango/videos"
SAVE_PATH = "/home/windowsuser/test"
FILE_PIPE_MAX = 20
IN_DATA_PIPE_MAX = 250
OUT_DATA_PIPE_MAX = 250
ASSIGN_WORKERS = 4

## Wandb

global USE_WANDB

USE_WANDB = True
WANDB_ENTITY = "tempofunk" # none if not using wandb
WANDB_PROJ = "shutterstock_stage3"
WANDB_ONLY_ONE_CORE = True

## TPU Workers
TPU_CORE_COUNT = 8
TPU_BATCH_SIZE = 16
IM2IM_MODEL_PATH = "im2im-sswm"

C_C = 3
C_H = 384 # (divisible by 64)
C_W = 640 # (divisible by 64)

## Debug
DEBUG = False
DR_DELAY = 0.1

def disk_reader(file_pipe: mp.Queue):
    print("dr: started")
    dir_list = os.listdir(DISK_PATH)
    print(f"dr: dir list loaded, {len(dir_list)} files found")
    for file in dir_list:
        filename, fileextension = os.path.splitext(file)
        if fileextension == ".json":
            continue
        if not os.path.exists(f'{DISK_PATH}/{filename}.json'):
            print(f"dr: {filename}.json not found, skipping")
            continue
        input_obj = (f'{DISK_PATH}/{file}', f'{DISK_PATH}/{filename}.json')
        file_pipe.put(input_obj)
        print(f"dr: {filename} sent to assign worker")
        if DEBUG:
            time.sleep(DR_DELAY)
    print("dr: all files loaded, sending termination signal")
    for i in range(TPU_CORE_COUNT):
        file_pipe.put((None, None))

class wandb_logger:
    def __init__(self, wandb_pipe: mp.Queue, index: int, prefix: str = None):
        self.wandb_pipe = wandb_pipe
        self.index = index
        self.prefix = prefix

    def dlog(self, name, data):
        self.wandb_pipe.put(
            {
                'at': 'direct', 
                'c': {
                    f"{self.prefix}_{self.index}/{name}": data
                }
            }
        )

    def g_dlog(self, data, name):
        self.wandb_pipe.put(
            {
                'at': 'direct', 
                'c': {
                    f"{self.prefix}_g/{name}": data
                }
            }
        )

def tpu_worker(index, in_data_pipe: mp.Queue, out_data_pipe: mp.Queue, wandb_pipe: mp.Queue = None):
    print(f'tw-{index}: started, grabbing device {index}')
    device = xm.xla_device()
    print(f'tw-{index}: device {index} successfully grabbed')
    model = load_model(IM2IM_MODEL_PATH, device)
    print(f'tw-{index}: model loaded')

    global USE_WANDB
    use_wandb = copy.copy(USE_WANDB)

    if WANDB_ONLY_ONE_CORE:
        if index != 0:
            use_wandb = False

    if use_wandb:
        wl = wandb_logger(wandb_pipe, index, "tw")

    def prep_batch(batch):
        # batch should be a torch tensor, uint8, 0-255, (B, H, W, C)
        batch = batch.permute(0,3,1,2) # (B, H, W, C) -> (B, C, H, W)
        batch = batch.to(device).to(torch.float32).div(255).to(memory_format=torch.contiguous_format) # batch is now a tensor, float32, 0-1
        _bs, _c, _h, _w = batch.shape
        # batch size must be managed by the encoder process manager (ENC-PM)
        right = math.ceil(_w / model.ksize) * model.ksize - _w
        bottom = math.ceil(_h / model.ksize) * model.ksize - _h
        batch = torch.nn.functional.pad(batch, [0, right, 0, bottom], mode = 'reflect')
        return batch, _h, _w
    
    def post_batch(batch, _h, _w):
        batch = batch[:,:,0:_h,0:_w]
        batch = batch.mul(255).round().clamp(0,255).permute(0,2,3,1).to(device = 'cpu', dtype = torch.uint8).numpy() # (B, C, H, W) -> (B, H, W, C) # batch is now a numpy array, uint8, 0-255
        return batch

    print(f'tw-{index}: init signal received')
    first_run = True # first run is always compilation 
    while True:
        data = in_data_pipe.get() # (B, H, W, C)
        if data is None:
            break
        if first_run:
            print(f'tw-{index}: first run, is always compilation')
        print(f'tw-{index}: data received')
        init_time = time.time()
        value = torch.from_numpy(data['value']) # pipe is always numpy to avoid problems
        value, _h, _w = prep_batch(value) # prep_batch expects torch
        print(f'tw-{index}: data preprocessed, end shape {value.shape}, dtype {value.dtype}, device {value.device}, range {value.min()} - {value.max()}')
        batch = model(value)
        print(f'tw-{index}: data modeled, out shape {batch.shape}, dtype {batch.dtype}, device {batch.device}, range {batch.min()} - {batch.max()}')
        batch = post_batch(batch, _h, _w)
        #batch = out.numpy(force=True)
        finish_time = time.time()
        if first_run:
            first_run = False
            print(f'tw-{index}: compilation done in {finish_time - init_time} seconds, out shape {batch.shape}')
        else:
            print(f'tw-{index}: data processed in {finish_time - init_time} seconds, out shape {batch.shape}')
        out_data_pipe.put({'value': batch, 'meta': {'batch_id': data['meta']['batch_id'], 'aw_worker_index': data['meta']['aw_worker_index']}})
        
        if use_wandb:
            # bps = batch per second (1 core)
            wl.dlog("bps", 1 / (finish_time - init_time))
            # fps = frame per second (1 core)
            wl.dlog("fps", (1 / (finish_time - init_time)) * TPU_BATCH_SIZE)
            if WANDB_ONLY_ONE_CORE:
                # global bps = batch per second (all cores)
                wl.g_dlog((1 / (finish_time - init_time)) * TPU_CORE_COUNT, "bps")
                # global fps = frame per second (all cores)
                wl.g_dlog(((1 / (finish_time - init_time)) * TPU_BATCH_SIZE) * TPU_CORE_COUNT, "fps")

def assign_worker(index: int, file_pipe: mp.Queue, in_data_pipe: mp.Queue, out_data_pipe: mp.Queue, wandb_pipe = mp.Queue):
    print("aw: started")

    global USE_WANDB
    use_wandb = copy.copy(USE_WANDB)

    while True:
        video_path, json_path = file_pipe.get()
        print(f"aw: {video_path} received")
        if video_path is None and json_path is None:
            print("aw: termination signal received")
            for i in range(TPU_CORE_COUNT):
                in_data_pipe.put(None)
            break
        frames, _, v_metadata = torchvision.io.read_video(video_path, output_format="TCHW")
        frames = F.resize(frames, (C_H, C_W))
        frame_count = frames.shape[0]
        print(f"aw: {video_path} video loaded")

        if frame_count < TPU_BATCH_SIZE:
            print(f"aw: {video_path} - {frame_count} < {TPU_BATCH_SIZE}, skipping")
            continue

        j_metadata = json.load(open(json_path, 'r'))

        # batch = (f, f, f, f, .. 16 times) # 16 frames
        # superbatch = (b, b, b, b, ...Y times) # Y splits where Y is the frame_count // TPU_BATCH_SIZE

        # Divide video into batches
        superbatch_count = frame_count // TPU_BATCH_SIZE
        print(f"aw: {video_path} - using {superbatch_count * TPU_BATCH_SIZE}/{frame_count} frames, {superbatch_count} megabatches")

        batch_ids_to_retrieve = set(range(superbatch_count))

        print(f"aw: {video_path} - sending batches to TPU workers")
        for i in range(superbatch_count):
            batch = frames[range(i * TPU_BATCH_SIZE, (i + 1) * TPU_BATCH_SIZE)]
            batch = batch.permute(0,2,3,1) # (B, C, H, W) -> (B, H, W, C)
            # I don't think permute latency matters that much, yes its done twice but at most its 1ms of extra latency
            # https://oneflow2020.medium.com/how-to-implement-a-permute-transpose-op-6-times-faster-than-pytorch-6280d63d0b4b
            batch = batch.numpy(force=True)
            assert batch.shape[3] == C_C
            assert batch.shape[0] == TPU_BATCH_SIZE
            assert batch.dtype == np.uint8
            assert batch.min() >= 0
            assert batch.max() <= 255
            batch = {'value': batch, 'meta': {'batch_id': i, 'aw_worker_index': index}}
            in_data_pipe.put(batch) 
            # I EXPECT (B, H, W, C) AKA (16, 256, 256, 3)!!!!
            # batch should be a numpy array, uint8, 0-255, (B, H, W, C)
            print(f"aw: {video_path} - batch {i} sent with sahpe {batch['value'].shape}")
        print(f"aw: {video_path} - batches sent to TPU workers")

        output_superbatch = list()

        while len(batch_ids_to_retrieve) > 0:
            batch = out_data_pipe.get()
            if batch['meta']['aw_worker_index'] != index:
                out_data_pipe.put(batch)
                continue
            batch_id = batch['meta']['batch_id']
            batch_ids_to_retrieve.remove(batch_id)
            # add batch at the correct index
            output_superbatch.append({"o": batch_id, "v": batch['value']})
            print(f"aw: {video_path} - batch {batch_id} received")
        # order
        output_superbatch.sort(key=lambda x: x['o'])
        output_superbatch = [x['v'] for x in output_superbatch]
        print(f"aw: {video_path} - all batches received")

        video_id = j_metadata['id']

        final_out = np.concatenate(output_superbatch, axis=0) # batch should be a numpy array, uint8, 0-255, (B, H, W, C)
        torchvision.io.write_video(f'{SAVE_PATH}/{video_id}.mp4', final_out, v_metadata['video_fps'])
        print(f"aw: {video_path} - {SAVE_PATH}/{video_id}.mp4 saved")

        if use_wandb:
            wandb_pipe.put(
                {
                    'at': 'sum', 
                    'c': {
                        'name': 'aw/videos',
                        'value': 1
                    }
                }
            )

            wandb_pipe.put(
                {
                    'at': 'sum', 
                    'c': {
                        'name': 'aw/frames',
                        'value': frame_count
                    }
                }
            )

            wandb_pipe.put(
                {
                    'at': 'sum', 
                    'c': {
                        'name': 'aw/hours',
                        'value': frame_count / v_metadata['video_fps'] / 60 / 60
                    }
                }
            )

def wandb_worker(wandb_pipe: mp.Queue):
    wandb_run = wandb.init(
        project=WANDB_PROJ, 
        entity=WANDB_ENTITY,
        config={
            "FILE_PIPE_MAX": FILE_PIPE_MAX,
            "IN_DATA_PIPE_MAX": IN_DATA_PIPE_MAX,
            "OUT_DATA_PIPE_MAX": OUT_DATA_PIPE_MAX,
            "ASSIGN_WORKERS": ASSIGN_WORKERS,
            "TPU_CORE_COUNT": TPU_CORE_COUNT,
            "TPU_BATCH_SIZE": TPU_BATCH_SIZE,
            "C_H": C_H,
            "C_W": C_W,
        },
        name=f"{CODENAME}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
        )

    start_time = time.time()
    
    local_vars = {}

    while True:
        data = wandb_pipe.get()
        if data is None:
            break
        access_type = data['at'] # access type
        content = data['c'] # content
        curr_time = time.time()
        time_diff = int(curr_time - start_time)

        if access_type == "direct":
            wandb_run.log(content, step = time_diff)
        elif access_type == "sum":
            if content['name'] in local_vars:
                local_vars[content['name']] += content['value']
            else:
                local_vars[content['name']] = content['value']
            wandb_run.log({content['name']: local_vars[content['name']]}, step = time_diff)

def main():
    print("Initializing...")
    global manager
    manager = mp.Manager()
    file_pipe = manager.Queue(FILE_PIPE_MAX)
    in_data_pipe = manager.Queue(IN_DATA_PIPE_MAX)
    out_data_pipe = manager.Queue(OUT_DATA_PIPE_MAX)
    print("Objects created")
    assign_workers = list()

    if USE_WANDB:
        wandb_pipe = manager.Queue()
        wandb_worker_process = mp.Process(target=wandb_worker, args=(wandb_pipe,))
        wandb_worker_process.start()
        print("Wandb worker started")
    else:
        wandb_pipe = None

    for i in range(ASSIGN_WORKERS):
        assign_worker_process = mp.Process(target=assign_worker, args=(i, file_pipe, in_data_pipe, out_data_pipe, wandb_pipe,))
        assign_worker_process.start()
        assign_workers.append(assign_worker_process)
        print(f"Assign worker {i} started")

    print("Assign workers started")

    disk_reader_worker = mp.Process(target=disk_reader, args=(file_pipe,))
    disk_reader_worker.start()
    print("Disk reader started")

    print("starting TPU workers (forced join, idk why???)")
    xmp_obj = xmp.spawn(tpu_worker, args=(in_data_pipe, out_data_pipe, wandb_pipe,), nprocs=TPU_CORE_COUNT)
    print("TPU workers started")

    for worker in assign_workers:
        worker.join()
    disk_reader_worker.join()
    if USE_WANDB:
        wandb_worker_process.join()
    print("All workers joined, exiting")

if __name__ == "__main__":
    main()
import multiprocessing as mp
import time
import json
import os
import numpy as np
import cv2
import wandb
import copy
import gc
import numpy as np
import psutil
import logging

def get_memory():
    mem_info = psutil.virtual_memory()
    free_memory = f"SMU: {round(mem_info.used / 1024 ** 2, 2)}MB; {mem_info.percent}" 
    return free_memory

class CustomLogger(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return f'{get_memory()} - {msg}', kwargs

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


mp.set_start_method("spawn", force=True)

"""
v1 - ghostfood
https://www.youtube.com/watch?v=fBXMGe5mNCE
v1-test1 - toerina
to see how much memory it eats
v1-1-prod - ghostfood
"""
CODENAME = "notv-threads-toerina"

# config

## General
DISK_PATH = "/mnt/disks/hhd/tango/videos"
SAVE_PATH = "/home/windowsuser/test"
FILE_PIPE_MAX = 2
IN_DATA_PIPE_MAX = 256
OUT_DATA_PIPE_MAX = 256
ASSIGN_WORKERS = 1

## Wandb

global USE_WANDB

USE_WANDB = True
WANDB_ENTITY = "tempofunk" # none if not using wandb
WANDB_PROJ = "shutterstock_stage4"
WANDB_ONLY_ONE_CORE = True

## TPU Workers
TPU_CORE_COUNT = 8
TPU_BATCH_SIZE = 32
MAX_SUPERBATCHES = 30
MIN_SUPERBATCHES = TPU_CORE_COUNT / ASSIGN_WORKERS

QUEUE_TIMEOUT = 0.5 # 500 ms

VAE_MODEL_PATH = "CompVis/stable-diffusion-v1-4"
VAE_MODEL_REVISION = "flax"
VAE_MODEL_SUBFOLDER = "vae"
C_C = 3
C_H = 384 # (divisible by 64)
C_W = 640 # (divisible by 64)

## Debug
DEBUG = False
DR_DELAY = 0.1

## tmp
SNAPSHOT_DIR = "/home/windowsuser/crawlers/sites/shutterstock/stage3/snapshots"

"""
Per Image               |  Total Time Taken (10 rounds)| Devices; Batch Size
0.007594532426446676        19.441503763198853          8; 32
0.007988434843719005        20.449864387512207          8; 32
0.007887406926602124        20.19119167327881           8; 32
0.013347899541258812        17.084990739822388          8; 16
0.013619952462613582        17.433118104934692          8; 16
0.012901734933257103        16.51395297050476           8; 16
0.013376232236623764        8.560500860214233           8; 8
0.013047899678349495        8.350401401519775           8; 8
0.012519768625497817        8.012182235717773           8; 8
"""

def disk_reader(file_pipe: mp.Queue):
    gc_obj = 0
    logger.info("dr: started")
    dir_list = os.listdir(DISK_PATH)
    logger.info(f"dr: dir list loaded, {len(dir_list)} files found")
    for file in dir_list:
        filename, fileextension = os.path.splitext(file)
        if fileextension == ".json":
            continue
        if not os.path.exists(f'{DISK_PATH}/{filename}.json'):
            logger.info(f"dr: {filename}.json not found, skipping")
            continue
        input_obj = (f'{DISK_PATH}/{file}', f'{DISK_PATH}/{filename}.json')
        file_pipe.put(input_obj)
        logger.info(f"dr: {filename} sent to assign worker")
        del input_obj, filename, fileextension
        gc_obj += gc.collect()
        if DEBUG:
            time.sleep(DR_DELAY)
    del dir_list
    gc_obj += gc.collect()
    logger.info(f"dr: all files loaded, sending termination signal, gc: {gc_obj} objects collected")
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
    from diffusers import FlaxAutoencoderKL
    import jax.numpy as jnp
    import jax
    weight_dtype = jnp.float16
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        VAE_MODEL_PATH,
        revision=VAE_MODEL_REVISION,
        subfolder=VAE_MODEL_SUBFOLDER,
        dtype=weight_dtype,
    )
    logger.info(f'tw-{index}: model loaded')

    global USE_WANDB
    use_wandb = copy.copy(USE_WANDB)

    # if WANDB_ONLY_ONE_CORE:
    #     if index != 0:
    #         use_wandb = False
    # only one process since we use jax.pmap to split the batch

    if use_wandb:
        wl = wandb_logger(wandb_pipe, index, "tw")

    expected_input_shape = (TPU_CORE_COUNT, TPU_BATCH_SIZE, C_C, C_H, C_W)

    def prep_batch(batch):
        # (D, B, C, H, W) (uint8) (0-255)
        assert batch.dtype == np.uint8, f"{batch.dtype} != uint8"
        assert batch.max() <= 255, f"{batch.max()} > 255"
        assert batch.min() >= 0, f"{batch.min()} < 0"
        assert batch.shape == expected_input_shape, f"{batch.shape} != {expected_input_shape}"
        batch = batch.astype(np.float32)
        batch = batch / 255.0 # (D, B, C, H, W) (float32) (0-1)
        # added 7/24/2023
        batch = batch * 2.0 - 1.0
        assert batch.dtype == np.float32, f"{batch.dtype} != float32"
        assert batch.max() <= 1.0, f"{batch.max()} > 1.0"
        assert batch.min() >= -1.0, f"{batch.min()} < -1.0"
        gc.collect()
        return batch
    
    def encode(img, rng, vae_params):
        init_latent_dist = vae.apply({"params": vae_params}, img, method=vae.encode).latent_dist
        latent_dist = init_latent_dist.sample(key=rng)
        latent_dist = latent_dist.transpose((0, 3, 1, 2))
        latent_dist = vae.config.scaling_factor * latent_dist
        return latent_dist
    
    def create_key(seed=0):
        return jax.random.PRNGKey(seed)
    rng = create_key(0)
    
    model = jax.pmap(encode, in_axes=(0, None, None))
    first_run = True # first run is always compilation 
    logger.info(f'tw-{index}: starting run...')

    null_meta = {'batch_id': -1, 'aw_worker_index': -1}

    while True:
        gc_obj = 0
        np_batch = list()
        data_batch = list()

        for i in range(TPU_CORE_COUNT):
            try:
                data = in_data_pipe.get(timeout=QUEUE_TIMEOUT)
                # (B, C, H, W) (uint8) (0-255)
                if data is None:
                    break
                np_batch.append(data['value'])
                data_batch.append(data['meta'])
            except Exception:
                logger.info(f'tw-{index}: queue timeout, filling with random data')
                data = np.random.randint(0, 255, size=(TPU_BATCH_SIZE, C_C, C_H, C_W), dtype=np.uint8)
                np_batch.append(data)
                data_batch.append(null_meta)

        np_batch = np.stack(np_batch)

        if first_run:
            logger.info(f'tw-{index}: first run, is always compilation')

        logger.info(f'tw-{index}: data received')
        init_time = time.time()
        value = prep_batch(np_batch) # converts to float32, -1 <-> 1

        logger.info(f'tw-{index}: data preprocessed, end shape {value.shape}, dtype {value.dtype}, range {value.min()} - {value.max()}')

        output = model(value, rng, vae_params)
        output = np.array(output) # (D, B, C, H, W) (float32)

        logger.info(f'tw-{index}: data modeled, out shape {output.shape}, dtype {output.dtype}, range {output.min()} - {output.max()}')

        finish_time = time.time()
        if first_run:
            first_run = False
            logger.info(f'tw-{index}: compilation done in {finish_time - init_time} seconds, out shape {output.shape}')
        else:
            logger.info(f'tw-{index}: data processed in {finish_time - init_time} seconds, out shape {output.shape}')

        for i in range(TPU_CORE_COUNT):
            obt_out = output[i]
            obt_dat = data_batch[i]
            if obt_dat['batch_id'] == -1 and obt_dat['aw_worker_index'] == -1:
                continue
            out_data_pipe.put({'value': obt_out, 'meta': obt_dat})
            # output should be:
            # (B, C, H, W) (float32) (0-1)
            # (32, 4, 64, 64) (float32) (0-1)
        
        if use_wandb:
            # bps = batch per second (1 core)
            wl.dlog("bps", 1 / (finish_time - init_time) * TPU_CORE_COUNT)
            # fps = frame per second (1 core)
            wl.dlog("fps", (1 / (finish_time - init_time) * TPU_CORE_COUNT) * TPU_BATCH_SIZE)

        del value, data, init_time, finish_time
        gc_obj += gc.collect()
        logger.info(f'tw-{index}: data processed, {gc_obj} objects collected')

def assign_worker(index: int, file_pipe: mp.Queue, in_data_pipe: mp.Queue, out_data_pipe: mp.Queue, wandb_pipe = mp.Queue):
    logger.info(f"aw-{index}: started")

    global USE_WANDB
    use_wandb = copy.copy(USE_WANDB)

    while True:
        gc_obj = 0

        logger.info(f"aw-{index}: {index} waiting for file")
        video_path, json_path = file_pipe.get()
        logger.info(f"aw-{index}: {video_path} received")
        if video_path is None and json_path is None:
            logger.info(f"aw-{index}: termination signal received")
            for i in range(TPU_CORE_COUNT):
                in_data_pipe.put(None)
            break

        logger.info(f"aw-{index}: {video_path} loading video")
        video_cap = cv2.VideoCapture(video_path)
        logger.info(f"aw-{index}: {video_path} video loaded")
        resized_frames = []
        while True:
            ret, frame = video_cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, (C_W, C_H))
            numpy_frame = np.array(resized_frame)
            resized_frames.append(numpy_frame)
        frames = np.array(resized_frames)
        logger.info(f"aw-{index}: {video_path} video resized")
        frames = frames.transpose((0, 3, 1, 2)) # (F, C, H, W)
        logger.info(f"aw-{index}: {video_path} video transposed")
        frame_count = frames.shape[0]
        logger.info(f"aw-{index}: {video_path} video ready")

        if frame_count < TPU_BATCH_SIZE:
            logger.info(f"aw-{index}: {video_path} - {frame_count} < {TPU_BATCH_SIZE}, skipping")
            continue

        with open(json_path, 'r') as f:
            j_metadata = json.load(f)

        del json_path

        # batch = (f, f, f, f, .. 16 times) # 16 frames
        # superbatch = (b, b, b, b, ...Y times) # Y splits where Y is the frame_count // TPU_BATCH_SIZE

        # Divide video into batches
        superbatch_count = frame_count // TPU_BATCH_SIZE
        if superbatch_count > MAX_SUPERBATCHES:
            superbatch_count = MAX_SUPERBATCHES
        if superbatch_count < MIN_SUPERBATCHES:
            logger.info(f"aw-{index}: {video_path} - {superbatch_count} < {MIN_SUPERBATCHES} superbatch, skipping")
            continue
        logger.info(f"aw-{index}: {video_path} - using {superbatch_count * TPU_BATCH_SIZE}/{frame_count} frames, {superbatch_count} megabatches")

        batches_sent = 0

        logger.info(f"aw-{index}: {video_path} - sending batches to TPU workers")
        for i in range(superbatch_count):
            batch = frames[range(i * TPU_BATCH_SIZE, (i + 1) * TPU_BATCH_SIZE)]
            # should be a torch tensor, uint8, 0-255, (B, C, H, W)
            #batch = batch.numpy(force=True)
            assert batch.shape[1] == C_C
            assert batch.shape[0] == TPU_BATCH_SIZE
            assert batch.dtype == np.uint8
            assert batch.min() >= 0
            assert batch.max() <= 255
            batch = {'value': batch, 'meta': {'batch_id': i, 'aw_worker_index': index}}
            in_data_pipe.put(batch)
            # input should be:
            # (B, C, H, W) (uint8) (0-255)
            logger.info(f"aw-{index}: {video_path} - batch {i} sent with shape {batch['value'].shape}")
            batches_sent += 1
            del batch
        logger.info(f"aw-{index}: {video_path} - {batches_sent} batches sent to TPU workers")

        del frames
        gc_obj += gc.collect()

        output_superbatch = list()

        while batches_sent != 0:
            batch = out_data_pipe.get()
            if batch['meta']['aw_worker_index'] != index:
                out_data_pipe.put(batch)
                #unlike the dewatermarker, this will return a vae latent
                #which has a shape of (F, C, H, W) aka (16, 4, 64, 64)
                # sleep for (20-100 ms)
                logger.info(f"aw-{index}: {video_path} - obtained a workload meant for {batch['meta']['aw_worker_index']}, sleeping for a bit (50-100 ms")
                rand_time = np.random.randint(50, 100) / 1000
                time.sleep(rand_time)
                continue
            batch_id = batch['meta']['batch_id']
            batches_sent -= 1
            # add batch at the correct index
            output_superbatch.append({"o": batch_id, "v": batch['value']})
            logger.info(f"aw-{index}: {video_path} - batch {batch_id} received")
            del batch, batch_id
        # order
        output_superbatch.sort(key=lambda x: x['o'])
        output_superbatch = [x['v'] for x in output_superbatch]
        logger.info(f"aw-{index}: {video_path} - all batches received")

        video_id = j_metadata['id']

        final_out = np.concatenate(output_superbatch, axis=0) # batch should be a numpy array, (F, C, W, H) containing all frames in latent form
        expected_out_shape = (TPU_BATCH_SIZE * superbatch_count, 4, C_H/8, C_W/8)
        assert final_out.shape == expected_out_shape, f"{final_out.shape} != {expected_out_shape}"
        #np.save(f"{SAVE_PATH}/{video_id}.npy", final_out)
        logger.info(f"aw-{index}: {video_path} - {SAVE_PATH}/{video_id}.npy saved")

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

            # wandb_pipe.put(
            #     {
            #         'at': 'sum', 
            #         'c': {
            #             'name': 'aw/hours',
            #             'value': frame_count / v_metadata['video_fps'] / 60 / 60
            #         }
            #     }
            # )

            wandb_pipe.put(
                {
                    'at': 'direct', 
                    'c': {'aw/file_pipe': file_pipe.qsize(),
                          'aw/in_data_pipe': in_data_pipe.qsize(),
                          'aw/out_data_pipe': out_data_pipe.qsize(),
                          'aw/superbatch_count': superbatch_count,}
                }
            )

        del output_superbatch, video_id, final_out, j_metadata, frame_count, superbatch_count
        gc_obj += gc.collect()
        logger.info(f"aw-{index}: {video_path} - done, {gc_obj} objects collected")

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

        del data
        del access_type
        del content
        del curr_time
        del time_diff
        gc.collect()

def main():
    logger.info("Initializing...")
    logger.info(f"Start method is {mp.get_start_method()}")
    global manager
    manager = mp.Manager()
    file_pipe = manager.Queue(FILE_PIPE_MAX)
    in_data_pipe = manager.Queue(IN_DATA_PIPE_MAX)
    out_data_pipe = manager.Queue(OUT_DATA_PIPE_MAX)
    logger.info("Objects created")
    assign_workers = list()

    if USE_WANDB:
        wandb_pipe = manager.Queue()
        wandb_worker_process = mp.Process(target=wandb_worker, args=(wandb_pipe,))
        wandb_worker_process.start()
        logger.info("Wandb worker started")
    else:
        wandb_pipe = None

    for i in range(ASSIGN_WORKERS):
        assign_worker_process = mp.Process(target=assign_worker, args=(i, file_pipe, in_data_pipe, out_data_pipe, wandb_pipe,))
        assign_worker_process.start()
        assign_workers.append(assign_worker_process)
        logger.info(f"Assign worker {i} started")

    logger.info("Assign workers started")

    disk_reader_worker = mp.Process(target=disk_reader, args=(file_pipe,))
    disk_reader_worker.start()
    logger.info("Disk reader started")

    logger.info("starting TPU worker")
    tpu_worker_process = mp.Process(target=tpu_worker, args=(0, in_data_pipe, out_data_pipe, wandb_pipe,))
    tpu_worker_process.start()
    logger.info("TPU workers started")
    logger.info("All workers started, up and running")
    
    disk_reader_worker.join()
    logger.info("Disk reader joined")
    for i in range(ASSIGN_WORKERS):
        assign_workers[i].join()
        logger.info(f"Assign worker {i} joined")
    tpu_worker_process.join()
    logger.info("TPU worker joined")
    if USE_WANDB:
        wandb_worker_process.join()
    logger.info("All workers joined, exiting")

if __name__ == "__main__":
    main()
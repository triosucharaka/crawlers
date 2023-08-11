import multiprocessing as mp
import time
import json
import wandb
import os
import numpy as np
import io
import tempfile
import tarfile
import gc
import webdataset as wds
import logging
import psutil
import cv2
import threading
import traceback
import jax.numpy as jnp
import numpy as np
import jax
from diffusers import FlaxAutoencoderKL
from wrapt_timeout_decorator import *

mp.set_start_method("spawn", force=True)

### Configuration ###

## Paths
IN_DISK_PATH = "/home/windowsuser/mount-folder/tempofunkds/shutterstock/stage3/"
OUT_DISK_PATH = "/home/windowsuser/mount-folder/tempofunkds/shutterstock/stage4/"
JSON_MAP_PATH = "/home/windowsuser/crawlers/sites/shutterstock/stage4/map.json"
JSON_READ_PATH = "/home/windowsuser/mount-folder/tempofunkds/shutterstock/nano/stage4_read.json"

## Wandb
global USE_WANDB
USE_WANDB = True
WANDB_ENTITY = "peruano"  # none if not using wandb
WANDB_PROJ = "debug_shutterstock_stage4"
WANDB_NAME = f"mass_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
LOG_MEMORY = True
SPAWN_ON_EVERYTHING = False # wandb only grabs the current process logger, not global one, and also logs everything thats printed such as errors

## Multiprocessing
ASSIGN_WORKER_COUNT = 4
ASSIGN_WORKER_MAX_TASKS_PER_CHILD = 10
TAR_WORKER_COUNT = 4
TAR_SIZE = 128 * 1024 * 1024  # 512MB
TAR_INDIV_TIMEOUT = 15 # seconds
TAR_WORKER_MAX_TASKS_PER_CHILD = 5

## TPU Workers
TPU_CORE_COUNT = 4
TPU_BATCH_SIZE = 64
MAX_SUPERBATCHES = 60

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

def probe(prefix: str):
    run = wandb.init(
        project=WANDB_PROJ, 
        entity=WANDB_ENTITY, 
        name=f"{prefix}_{WANDB_NAME}",)

    return run

def wds_reader_func(file_pipe: mp.Queue):
    logger.info("WDS: started")

    if SPAWN_ON_EVERYTHING:
        run = probe("wds_reader")

    json_map = json.load(open(JSON_MAP_PATH, "r"))
    tar_map = list()
    read_tars = list()

    if os.path.exists(JSON_READ_PATH):
        removed_tars = 0
        json_read = json.load(open(JSON_READ_PATH, "r"))
        for entry in json_read:
            read_tars.append(entry)
            if entry in json_map:
                json_map.remove(entry)
                removed_tars += 1
        logger.info(f"WDS: removed {removed_tars} tars from map")

    logger.info(f"WDS: {len(json_map)} tars to read")

    for tar in json_map:
        tar_map.append(IN_DISK_PATH + tar)

    dataset = wds.WebDataset(tar_map)

    for sample in dataset:
        try:
            logger.info(f"WDS: sending {sample['__key__']}")
            file_pipe.put((sample["__key__"], sample["mp4"], sample["json"]))
        except Exception as e:
            logger.error(f"WDS: {sample['__key__']} ERROR - {e}")
            logger.error(traceback.format_exc())
            continue
        finally:
            tar_filename = sample["__url__"].split("/")[-1]
            if tar_filename not in read_tars:
                read_tars.append(tar_filename)
                json.dump(read_tars, open(JSON_READ_PATH, "w"))

    file_pipe.put((None, None, None))
    logger.info("WDS: finished")


def tpu_worker_func(
    index, in_data_pipe: mp.Queue, out_data_pipe: mp.Queue, tmp_c: mp.Value
):
    logger.info(f"TPU-{index}: started")

    if SPAWN_ON_EVERYTHING:
        run = probe(f"tpu_worker_{index}")

    weight_dtype = jnp.float16
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        VAE_MODEL_PATH,
        revision=VAE_MODEL_REVISION,
        subfolder=VAE_MODEL_SUBFOLDER,
        dtype=weight_dtype,
    )
    logger.info(f"TPU-{index}: model loaded")

    expected_input_shape = (TPU_CORE_COUNT, TPU_BATCH_SIZE, C_C, C_H, C_W)

    with tmp_c.get_lock():
        tmp_c.value += 1

    def prep_batch(batch):
        # batch is (D, B, H, W, C) (uint8) (0-255)
        batch = np.transpose(batch, (0, 1, 4, 2, 3)) # (D, B, H, W, C) -> (D, B, C, H, W)
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

    tasks = list()

    def put_batch_func(tensor, data_batch):
        try:
            logger.info(f"TPU-{index}/ASYNC: placing batch")
            
            output = np.array(tensor)

            for i in range(TPU_CORE_COUNT):
                out_data_pipe.put(
                    {
                        "value": output[i],
                        "meta": data_batch[i],
                    }
                )

            logger.info(f"TPU-{index}/ASYNC: batch on out_data_pipe")
        except Exception as e:
            logger.error(f"TPU-{index}/ASYNC: batch place failed: {e}")
            logger.error(traceback.format_exc())

    def put_batch(tensor, data_batch):
        task_thead = threading.Thread(target=put_batch_func, args=(tensor, data_batch))
        task_thead.start()
        tasks.append(task_thead)

    logger.info(f"TPU-{index}: funcs defined")
    first_run = True  # first run is always compilation
    work_done = False
    while True:
        try:
            np_batch = list()
            data_batch = list()

            logger.info(f"TPU-{index}: waiting for data")
            for i in range(TPU_CORE_COUNT):
                data = in_data_pipe.get()  # (B, H, W, C), numpy array, uint8, 0-255
                if data is None:
                    work_done = True
                    break
                np_batch.append(data['value'])
                data_batch.append(data['meta'])

            if work_done:
                break

            np_batch = np.stack(np_batch)

            logger.info(f"TPU-{index}: data received")

            if first_run:
                logger.info(f"TPU-{index}: first run, is always compilation")
            
            init_time = time.time()

            value = prep_batch(np_batch) # (D, B, C, H, W) (float32) (-1, 1)
            preparation_time = time.time() - init_time
            logger.info(f"TPU-{index}: batch prepped in {preparation_time} seconds, value shape: {value.shape}")

            total_frames = value.shape[0] * value.shape[1]

            output = model(value, rng, vae_params).block_until_ready()
            modeling_time = time.time() - init_time - preparation_time

            if first_run:
                logger.info(f"TPU-{index}: compiled in {modeling_time} seconds")
                first_run = False
            else:
                logger.info(f"TPU-{index}: modeled in {modeling_time} seconds - fps: {total_frames / modeling_time}")

            put_batch(output, data_batch)
            
            logger.info(f"TPU-{index}: called put_batch")
            
        except Exception as e:
            logger.error(f"TPU-{index}: ERROR - {e}")
            logger.error(traceback.format_exc())
            continue

    for task in tasks:
        task.join()

    logger.info(f"TPU-{index}: finished")


def assign_worker_func(
    index: int,
    file_pipe: mp.Queue,
    in_data_pipe: mp.Queue,
    out_data_pipe: mp.Queue,
    tar_pipe: mp.Queue,
    keep_restarting: mp.Value,
):
    logger.info(f"ASSIGN/PROC-{index}: started")

    processed_tasks = 0
    tasks = []

    vid_id = None

    def save_video_func(fps, final_out: np.array, metadata, vid_id):
        logger.info(f"ASSIGN/ASYNC-{index}: {vid_id} - I have been summoned")

        swap_meta = {
            "np_frame_count": final_out.shape[0],
            "np_height": final_out.shape[2], # axis 1 is latent channels (4)
            "np_width": final_out.shape[3],
        }

        numpy_bytes = io.BytesIO()
        np.save(numpy_bytes, final_out, allow_pickle=False)
        numpy_bytes = numpy_bytes.getvalue()

        metadata = {**metadata, **swap_meta}

        logger.info(
            f"ASSIGN/ASYNC-{index}: {vid_id} - npy encoded, {len(numpy_bytes)/1024/1024} MB with {final_out.shape[0]} frames"
        )

        metadata_bytes = json.dumps(metadata).encode("utf-8")  # json metadata bytes
        tar_pipe.put((vid_id, numpy_bytes, metadata_bytes, metadata))  # str, bytes, bytes
        logger.info(f"ASSIGN/ASYNC-{index}: {vid_id} - sent to tar worker")

    def save_video(fps, final_out, metadata, vid_id):
        task_thead = threading.Thread(
            target=save_video_func, args=(fps, final_out, metadata, vid_id)
        )
        task_thead.start()
        tasks.append(task_thead)

    while True:
        try:
            if processed_tasks >= ASSIGN_WORKER_MAX_TASKS_PER_CHILD:
                logger.info(
                    f"ASSIGN/PROC-{index}: processed {processed_tasks} tasks (max tasks per child), restarting"
                )
                break

            gc_obj = 0

            logger.info(f"ASSIGN/PROC-{index}: {index} waiting for file")
            obtained_obj = file_pipe.get()

            if obtained_obj == (None, None, None):
                logger.info(f"ASSIGN/PROC-{index}: termination signal received")
                for i in range(TPU_CORE_COUNT):
                    in_data_pipe.put(None, timeout=1)
                    tar_pipe.put((None, None, None, None), timeout=1)
                    keep_restarting.value = 0
                break

            vid_id, mp4_bytes, metadata = obtained_obj

            logger.info(f"ASSIGN/PROC-{index}: {vid_id} received")

            metadata = json.loads(metadata.decode("utf-8"))

            ## Try predicting to avoid unnecessary processing
            if float(metadata["duration"]) > 10 * 60:
                # NOTE: not that we cannot split it, but rather could use too much memory when resizing.
                logger.info(
                    f"ASSIGN/PROC-{index}: {vid_id} - {metadata['duration']} > 10*60, during prediction, skipping"
                )
                continue

            predicted_frame_count = int(
                float(metadata["fps"]) * float(metadata["duration"])
            )
            predicted_superbatch_count = predicted_frame_count // TPU_BATCH_SIZE
            if predicted_superbatch_count > MAX_SUPERBATCHES:
                predicted_superbatch_count = MAX_SUPERBATCHES
            predicted_superbatch_count = predicted_superbatch_count // TPU_CORE_COUNT
            if predicted_superbatch_count < 1:
                logger.info(
                    f"ASSIGN/PROC-{index}: {vid_id} - {predicted_superbatch_count} < 1, during prediction, skipping"
                )
                continue

            logger.info(f"ASSIGN/PROC-{index}: {vid_id} loading video")
            with tempfile.NamedTemporaryFile() as temp:
                temp.write(mp4_bytes)
                temp.flush()
                video_cap = cv2.VideoCapture(temp.name)

                resized_frames = []

                logger.info(f"ASSIGN/PROC-{index}: {vid_id} resizing video")
                while True:
                    ret, frame = video_cap.read()
                    if not ret:
                        break
                    resized_frame = cv2.resize(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (C_W, C_H)
                    )
                    numpy_frame = np.array(resized_frame)
                    resized_frames.append(numpy_frame)
                logger.info(f"ASSIGN/PROC-{index}: {vid_id} video resized")

                frames = np.array(resized_frames)  # (T, H, W, C)
                frame_count = frames.shape[0]
                fps = video_cap.get(cv2.CAP_PROP_FPS)

                del video_cap, resized_frames, resized_frame, numpy_frame, frame, mp4_bytes

            logger.info(f"ASSIGN/PROC-{index}: {vid_id} video loaded")

            if frame_count < TPU_BATCH_SIZE:
                logger.info(
                    f"ASSIGN/PROC-{index}: {vid_id} - {frame_count} < {TPU_BATCH_SIZE}, skipping"
                )
                continue

            # batch = (f, f, f, f, .. 16 times) # 16 frames
            # superbatch = (b, b, b, b, ...Y times) # Y splits where Y is the frame_count // TPU_BATCH_SIZE

            # Divide video into batches
            superbatch_count = frame_count // TPU_BATCH_SIZE
            if superbatch_count > MAX_SUPERBATCHES:
                superbatch_count = MAX_SUPERBATCHES
            superbatch_count = superbatch_count // TPU_CORE_COUNT  # to fit exactly
            if superbatch_count < 1:
                logger.info(f"ASSIGN/PROC-{index}: {vid_id} - {superbatch_count} < 1, skipping (no superbatches)")
                continue
            superbatch_count = superbatch_count * TPU_CORE_COUNT
            logger.info(
                f"ASSIGN/PROC-{index}: {vid_id} - using {superbatch_count * TPU_BATCH_SIZE}/{frame_count} frames, {superbatch_count} megabatches"
            )

            batches_sent = 0

            logger.info(f"ASSIGN/PROC-{index}: {vid_id} - sending batches to TPU workers")
            for i in range(superbatch_count):
                batch = frames[range(i * TPU_BATCH_SIZE, (i + 1) * TPU_BATCH_SIZE)]
                assert batch.shape[3] == C_C
                assert batch.shape[0] == TPU_BATCH_SIZE
                assert batch.dtype == np.uint8
                assert batch.min() >= 0
                assert batch.max() <= 255
                batch = {
                    "value": batch,
                    "meta": {"batch_id": i, "aw_worker_index": index, "vid_id": vid_id},
                }
                in_data_pipe.put(batch)
                # I EXPECT (B, H, W, C) AKA (16, 256, 256, 3)!!!!
                # batch should be a numpy array, uint8, 0-255, (B, H, W, C)
                logger.info(
                    f"ASSIGN/PROC-{index}: {vid_id} - batch {i} sent with shape {batch['value'].shape}"
                )
                batches_sent += 1
                del batch
            logger.info(f"ASSIGN/PROC-{index}: {vid_id} - bastches sent to TPU workers")

            del frames
            gc_obj += gc.collect()

            ### RETRIVAL ###

            output_superbatch = list()

            while batches_sent != 0:
                batch = out_data_pipe.get()
                if batch["meta"]["aw_worker_index"] != index:
                    # NOTE: previously this was a delay, but now that we have the pipe manager, this should never happen
                    raise Exception(
                        f"ASSIGN/PROC-{index}: {vid_id} - batch {batch['meta']['batch_id']} received from wrong worker, got {batch['meta']['aw_worker_index']}, expected {index}"
                    )
                batch_id = batch["meta"]["batch_id"]
                batches_sent -= 1
                # add batch at the correct index
                output_superbatch.append({"o": batch_id, "v": batch["value"]})
                logger.info(f"ASSIGN/PROC-{index}: {vid_id} - batch {batch_id} received")
                del batch, batch_id

            ### ORGANIZING ###

            output_superbatch.sort(key=lambda x: x["o"])
            output_superbatch = [x["v"] for x in output_superbatch]
            logger.info(f"ASSIGN/PROC-{index}: {vid_id} - all batches received")
            final_out = np.concatenate(output_superbatch, axis=0)

            expected_out_shape = (TPU_BATCH_SIZE * superbatch_count, 4, C_H/8, C_W/8)
            assert final_out.shape == expected_out_shape

            logger.info(
                f"ASSIGN/PROC-{index}: {vid_id} - final output shape {final_out.shape}, writing numpy..."
            )

            save_video(fps, final_out, metadata, vid_id)

            logger.info(f"ASSIGN/PROC-{index}: {vid_id} - called save_video (Async)")

            gc_obj += gc.collect()
            logger.info(f"ASSIGN/PROC-{index}: {vid_id} - done, {gc_obj} objects collected")

            del vid_id

            processed_tasks += 1
            logger.info(f"ASSIGN/PROC-{index}: {processed_tasks} tasks processed")
        except Exception as e:
            logger.error(f"ASSIGN/PROC-{index}: {vid_id} - ERROR - {e}")
            logger.error(traceback.format_exc())
            continue

    logger.info(f"ASSIGN/PROC-{index}: waiting for async tasks to finish")
    for task in tasks:
        task.join()
    logger.info(f"ASSIGN/PROC-{index}: exiting")


def assign_worker_manager(
    index: int,
    file_pipe: mp.Queue,
    in_data_pipe: mp.Queue,
    out_data_pipe: mp.Queue,
    tar_pipe: mp.Queue,
):
    logger.info(f"ASSIGN/MANAGER-{index}: started")

    if SPAWN_ON_EVERYTHING:
        run = probe(f"assign_worker_manager_{index}")

    keep_restarting = mp.Value("i", 1)

    while keep_restarting.value == 1:
        try:
            logger.info(f"ASSIGN/MANAGER-{index}: starting worker")
            child_process = mp.Process(
                target=assign_worker_func,
                args=(
                    index,
                    file_pipe,
                    in_data_pipe,
                    out_data_pipe,
                    tar_pipe,
                    keep_restarting,
                ),
            )
            logger.info(f"ASSIGN/MANAGER-{index}: worker created")
            child_process.start()
            logger.info(f"ASSIGN/MANAGER-{index}: worker started")
            child_process.join()
            gc.collect()
            logger.info(f"ASSIGN/MANAGER-{index}: worker exited/joined")
        except Exception as e:
            logger.error(f"ASSIGN/MANAGER-{index}: ERROR - {e}")
            logger.error(traceback.format_exc())
            continue

    logger.info(f"ASSIGN/MANAGER-{index}: exiting")


def tar_worker_func(
    index: int, tar_pipe: mp.Queue, wandb_pipe: mp.Queue, tar_id: mp.Value, keep_restarting: mp.Value
):
    processed_tasks = 0

    @timeout(TAR_INDIV_TIMEOUT)
    def add2tar(files_tuple, tar):
        vid_id, npy_bytes, meta_bytes = files_tuple
        logger.info(f"TAR-{index}: {vid_id} - writing to tar")
        tarinfo = tarfile.TarInfo(name=f"{vid_id}.npy")
        tarinfo.size = len(npy_bytes)
        tar.addfile(tarinfo, io.BytesIO(npy_bytes))
        logger.info(f"TAR-{index}: {vid_id} - numpy written to tar")
        tarinfo = tarfile.TarInfo(name=f"{vid_id}.json")
        tarinfo.size = len(meta_bytes)
        tar.addfile(tarinfo, io.BytesIO(meta_bytes))
        logger.info(f"TAR-{index}: {vid_id} - meta written to tar")
    
    if SPAWN_ON_EVERYTHING:
        run = probe(f"tar_worker_{index}")

    files_tar = list()
    files_size = 0
    vid_id = None # for error handling without pipe.get

    logger.info(f"TAR-{index}: started")

    while True:
        try:
            if processed_tasks >= TAR_WORKER_MAX_TASKS_PER_CHILD:
                logger.info(
                    f"TAR/PROC-{index}: processed {processed_tasks} tasks (max tasks per child), restarting"
                )
                break

            logger.info(f"TAR/PROC-{index}: waiting for data")
            vid_id, npy_bytes, meta_bytes, meta = tar_pipe.get()
            logger.info(f"TAR/PROC-{index}: got data")

            if all(x is None for x in [vid_id, npy_bytes, meta_bytes, meta]):
                logger.info(f"TAR/PROC-{index}: got None, exiting")
                keep_restarting.value = 0
                if USE_WANDB:
                    wandb_pipe.put(None)
                break

            files_tar.append((vid_id, npy_bytes, meta_bytes))
            files_size += len(npy_bytes) + len(meta_bytes)
            if USE_WANDB:
                wandb_pipe.put(
                    {
                        "type": "video_entry",
                        "data": (
                            vid_id,
                            meta["cv_fps"],
                            meta["np_frame_count"] / meta["cv_fps"],
                            meta["np_frame_count"],
                        ),
                    }
                )

            logger.info(
                f"TAR/PROC-{index}: {vid_id} - added to tar, size {files_size/1024/1024} MB"
            )

            if files_size > TAR_SIZE:
                with tar_id.get_lock():
                    tar_id.value += 1
                    tar_id_val = tar_id.value
                logger.info(
                    f"TAR/PROC-{index}: {tar_id_val} - tar size exceeded, writing to disk"
                )

                tar_id_val = str(tar_id_val).zfill(6)

                # make sure we dont use the ones from current itteration
                del vid_id, npy_bytes, meta_bytes, meta

                success_files = 0
                fail_files = 0

                with tarfile.open(
                    name=f"{OUT_DISK_PATH}/{tar_id_val}.tar", mode="w"
                ) as tar:
                    for files_tuple in files_tar:
                        vid_id = files_tuple[0]
                        try:
                            add2tar(files_tuple, tar)
                            success_files += 1
                            logger.info(
                                f"TAR/PROC-{index}: {vid_id} - successfully written to tar"
                            )
                        except TimeoutError:
                            fail_files += 1
                            logger.error(
                                f"TAR/PROC-{index}: {vid_id} - ERROR - timeout while writing to tar"
                            )
                            logger.error(traceback.format_exc())
                            continue
                        except Exception as e:
                            fail_files += 1
                            logger.error(
                                f"TAR/PROC-{index}: {vid_id} - ERROR - {e}"
                            )
                            logger.error(traceback.format_exc())
                            continue
                files_tar = list()
                files_size = 0

                processed_tasks += 1

                logger.info(f"TAR/PROC-{index}: {tar_id_val} - tar written to disk, with {success_files} files, and {fail_files} failutes, total processed tasks {processed_tasks}")

                if USE_WANDB:
                    wandb_pipe.put({"type": "tar_entry", "data": None})
        except Exception as e:
            logger.error(f"TAR/PROC-{index}: {vid_id} - ERROR - {e}")
            logger.error(traceback.format_exc())
            continue

def tar_worker_manager(
    index: int, tar_pipe: mp.Queue, wandb_pipe: mp.Queue, tar_id: mp.Value
):
    logger.info(f"TAR/MANAGER-{index}: started")

    if SPAWN_ON_EVERYTHING:
        run = probe(f"tar_manager_{index}")

    keep_restarting = mp.Value("i", 1)

    while keep_restarting.value == 1:
        try:
            logger.info(f"TAR/MANAGER-{index}: starting worker")
            tar_worker = mp.Process(
                target=tar_worker_func,
                args=(index, tar_pipe, wandb_pipe, tar_id, keep_restarting,),
            )
            logger.info(f"TAR/MANAGER-{index}: worker created")
            tar_worker.start()
            logger.info(f"TAR/MANAGER-{index}: worker started")
            tar_worker.join()
            gc.collect()
            logger.info(f"TAR/MANAGER-{index}: worker exited/joined")
        except Exception as e:
            logger.error(f"TAR/MANAGER-{index}: ERROR - {e}")
            logger.error(traceback.format_exc())
            continue

    logger.info(f"TAR/MANAGER-{index}: exiting")

def wandb_worker_func(
    wandb_pipe: mp.Queue,
    tmp_c: mp.Value,
    file_pipe: mp.Queue,
    i_dp: mp.Queue,
    o_dp: mp.Queue,
    t_p: mp.Queue,
):
    
    from datetime import datetime

    def convert_unix_timestamp(unix_timestamp):
        time = datetime.fromtimestamp(unix_timestamp)
        return time.strftime("%H:%M:%S")
    
    run_config = {
        "IN_DISK_PATH": IN_DISK_PATH,
        "OUT_DISK_PATH": OUT_DISK_PATH,
        "JSON_MAP_PATH": JSON_MAP_PATH,
        "LOG_MEMORY": LOG_MEMORY,
        "ASSIGN_WORKER_COUNT": ASSIGN_WORKER_COUNT,
        "ASSIGN_WORKER_MAX_TASKS_PER_CHILD": ASSIGN_WORKER_MAX_TASKS_PER_CHILD,
        "TAR_WORKER_COUNT": TAR_WORKER_COUNT,
        "TAR_SIZE": TAR_SIZE,
        "TPU_CORE_COUNT": TPU_CORE_COUNT,
        "TPU_BATCH_SIZE": TPU_BATCH_SIZE,
        "MAX_SUPERBATCHES": MAX_SUPERBATCHES,
        "VAE_MODEL_PATH": VAE_MODEL_PATH,
        "VAE_MODEL_REVISION": VAE_MODEL_REVISION,
        "VAE_MODEL_SUBFOLDER": VAE_MODEL_SUBFOLDER,
        "C_C": C_C,
        "C_H": C_H,
        "C_W": C_W,
        "FILE_PIPE_MAX": FILE_PIPE_MAX,
        "IN_DATA_PIPE_MAX": IN_DATA_PIPE_MAX,
        "OUT_DATA_PIPE_MAX": OUT_DATA_PIPE_MAX,
        "TAR_PIPE_MAX": TAR_PIPE_MAX,
        "WANDB_PIPE_MAX": WANDB_PIPE_MAX,
    }

    run = wandb.init(
        project=WANDB_PROJ, 
        entity=WANDB_ENTITY, 
        name=WANDB_NAME,
        config=run_config,)
    
    init_time = time.time()

    frames = 0
    hours = 0
    videos = 0
    tars = 0

    video_id = None

    while True:
        try:
            logger.info(f"WANDB: total tpus: {int(tmp_c.value)}")

            data = wandb_pipe.get()

            elapsed_time = int(time.time() - init_time)

            run.log(
                {
                    "f_p": file_pipe.qsize(),
                    "i_dp": i_dp.qsize(),
                    "o_dp": o_dp.qsize(),
                    "t_p": t_p.qsize(),
                }
            )

            if data is None:
                break

            if data["type"] == "video_entry":
                video_id, fps, duration, frame_count = data["data"]
                logger.info(
                    f"WANDB: video entry - vid_id: {video_id}, fps: {round(fps, 2)}, duration: {round(duration, 2)}, frame_count: {frame_count}"
                )
                videos += 1
                frames += frame_count
                hours += duration / 3600
                run.log(
                    {"videos": videos, "frames": frames, "hours": hours}, step=elapsed_time
                )

            elif data["type"] == "tar_entry":
                tars += 1
                run.log({"tars": tars}, step=elapsed_time)

            # speed report
            if elapsed_time > 0:
                run.log(
                    {
                        "fps": frames / elapsed_time,
                        "vps": videos / elapsed_time,
                        "tps": tars / elapsed_time,
                    },
                    step=elapsed_time,
                )

            logger.info(
                f"WANDB: total stats - runtime: {convert_unix_timestamp(elapsed_time)} - {videos} videos, {frames} frames, {round(hours, 3)} hours, {tars} tars, {round(frames / elapsed_time, 2)} fps, {round(videos / elapsed_time, 3)} vps, {round(tars / elapsed_time, 3)} tps"
                )
        except Exception as e:
            logger.error(f"WANDB: ERROR - {video_id} - {e}")
            logger.error(traceback.format_exc())
            continue

    run.finish()


def out_pipe_manager(main_out_pipe: mp.Queue, secondary_out_pipes: list):

    if SPAWN_ON_EVERYTHING:
        run = probe("out_pipe_manager")

    while True:
        try:
            data = main_out_pipe.get()
            if data is None:
                for pipe in secondary_out_pipes:
                    pipe.put(None)
                break
            worker_id = data["meta"]["aw_worker_index"]
            secondary_out_pipes[worker_id].put(data)
        except Exception as e:
            logger.error(f"OUT_PIPE_MANAGER: ERROR - {e}")
            logger.error(traceback.format_exc())
            continue


if __name__ == "__main__":
    logger.info("MAIN: started")

    manager = mp.Manager()

    ## pipes
    file_pipe = manager.Queue(maxsize=FILE_PIPE_MAX)
    in_data_pipe = manager.Queue(maxsize=IN_DATA_PIPE_MAX)
    tpu_out_data_pipe = manager.Queue(maxsize=OUT_DATA_PIPE_MAX * ASSIGN_WORKER_COUNT)
    tar_pipe = manager.Queue(maxsize=TAR_PIPE_MAX)

    ## worker pipes
    out_data_pipes = list()
    for i in range(ASSIGN_WORKER_COUNT):
        _queue = manager.Queue(maxsize=OUT_DATA_PIPE_MAX)
        out_data_pipes.append(_queue)

    tmp_c = mp.Value("i", 0)
    tar_id = mp.Value("i", 0)

    ## processes
    if USE_WANDB:
        wandb_pipe = manager.Queue(maxsize=WANDB_PIPE_MAX)
        wandb_worker_process = mp.Process(
            target=wandb_worker_func,
            args=(
                wandb_pipe,
                tmp_c,
                file_pipe,
                in_data_pipe,
                tpu_out_data_pipe,
                tar_pipe,
            ),
        )
        wandb_worker_process.start()
    else:
        wandb_pipe = None

    wds_worker_process = mp.Process(target=wds_reader_func, args=(file_pipe,))
    pipe_manager_process = mp.Process(
        target=out_pipe_manager,
        args=(
            tpu_out_data_pipe,
            out_data_pipes,
        ),
    )

    assign_worker_processes = list()
    for i in range(ASSIGN_WORKER_COUNT):
        assign_worker_process = mp.Process(
            target=assign_worker_manager,
            args=(
                i,
                file_pipe,
                in_data_pipe,
                out_data_pipes[i],
                tar_pipe,
            ),
        )
        assign_worker_processes.append(assign_worker_process)

    tar_worker_processes = list()
    for i in range(TAR_WORKER_COUNT):
        tar_worker_process = mp.Process(
            target=tar_worker_manager,
            args=(
                i,
                tar_pipe,
                wandb_pipe,
                tar_id,
            ),
        )
        tar_worker_processes.append(tar_worker_process)

    ## start them
    wds_worker_process.start()
    pipe_manager_process.start()
    for assign_worker_process in assign_worker_processes:
        assign_worker_process.start()
    for tar_worker_process in tar_worker_processes:
        tar_worker_process.start()

    # Since we are now using jnp, theres no need for multiple tpu workers.
    tpu_worker_process = mp.Process(
        target=tpu_worker_func,
        args=(
            0,
            in_data_pipe,
            tpu_out_data_pipe,
            tmp_c,
        ),
    )
    tpu_worker_process.start()

    ## wait for them to finish
    logger.info("MAIN: waiting for workers to finish")
    wds_worker_process.join()
    logger.info("MAIN: wds_worker_process finished")
    pipe_manager_process.join()
    logger.info("MAIN: pipe_manager_process finished")
    for assign_worker_process in assign_worker_processes:
        assign_worker_process.join()
        logger.info("MAIN: SOME assign_worker_process finished")
    logger.info("MAIN: assign_worker_processes finished")
    for tar_worker_process in tar_worker_processes:
        tar_worker_process.join()
        logger.info("MAIN: SOME tar_worker_process finished")
    logger.info("MAIN: tar_worker_processes finished")
    tpu_worker_process.join()
    logger.info("MAIN: tpu_worker_process finished")

    if USE_WANDB:
        wandb_pipe.put(None)
        wandb_worker_process.join()

    logger.info("MAIN: finished")

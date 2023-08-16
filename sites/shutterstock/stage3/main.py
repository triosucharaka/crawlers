import multiprocessing as mp
import time
import json
import wandb
import math
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch
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
import os
from im2im.main import load_model
from wrapt_timeout_decorator import *

mp.set_start_method("spawn", force=True)

### Configuration ###

## Paths
IN_DISK_PATH = "/home/windowsuser/mount-folder/tempofunkds/shutterstock/stage2/videos/"
OUT_DISK_PATH = "/home/windowsuser/mount-folder/tempofunkds/shutterstock/stage3/0/"
JSON_MAP_PATH = "/home/windowsuser/mount-folder/tempofunkds/shutterstock/nano/output1.json"
JSON_READ_PATH = "/home/windowsuser/mount-folder/tempofunkds/shutterstock/nano/stage2_instance1.json"

## Wandb
global USE_WANDB
USE_WANDB = True
WANDB_ENTITY = "peruano"  # none if not using wandb
WANDB_PROJ = "shutterstock_stage3"
WANDB_NAME = f"stage3_instace1_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
LOG_MEMORY = True

## Multiprocessing
ASSIGN_WORKER_COUNT = 4
ASSIGN_WORKER_MAX_TASKS_PER_CHILD = 10
TAR_WORKER_COUNT = 8
TAR_SIZE = 512 * 1024 * 1024  # 512MB
TAR_INDIV_TIMEOUT = 15 # seconds
TAR_WORKER_MAX_TASKS_PER_CHILD = 5

## TPU Workers
TPU_CORE_COUNT = 4
TPU_BATCH_SIZE = 64
MAX_SUPERBATCHES = 60

## Model Parameters
IM2IM_MODEL_PATH = "im2im-sswm"
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
f_handler = logging.FileHandler("main.log")
f_format = logging.Formatter(f"%(asctime)s - %(name)s - %(levelname)s - %(message)s")
f_handler.setFormatter(f_format)
logger.addHandler(f_handler)
logger.setLevel(logging.DEBUG)

if LOG_MEMORY:
    logger = CustomLogger(logger, {})


def wds_reader_func(file_pipe: mp.Queue):
    logger.info("WDS: started")
    json_map = json.load(open(JSON_MAP_PATH, "r"))
    if os.path.exists(JSON_READ_PATH):
        json_read = json.load(open(JSON_READ_PATH, "r"))
    else:
        json_read = list()
    tar_map = list()
    read_tars = list()

    for entry in json_read:
        read_tars.append(entry["tar"])
        if entry in json_map:
            json_map.remove(entry)

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
    logger.info(f"TPU-{index}: started, grabbing device {index}")
    device = xm.xla_device()
    logger.info(f"TPU-{index}: device {index} successfully grabbed")
    model = load_model(IM2IM_MODEL_PATH, device)
    logger.info(f"TPU-{index}: model loaded")

    with tmp_c.get_lock():
        tmp_c.value += 1

    def prep_batch(batch):
        # batch should be a torch tensor, uint8, 0-255, (B, H, W, C)
        batch = batch.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        batch = (
            batch.to(device)
            .to(torch.float32)
            .div(255)
            .to(memory_format=torch.contiguous_format)
        )  # batch is now a tensor, float32, 0-1
        _bs, _c, _h, _w = batch.shape
        right = math.ceil(_w / model.ksize) * model.ksize - _w
        bottom = math.ceil(_h / model.ksize) * model.ksize - _h
        batch = torch.nn.functional.pad(batch, [0, right, 0, bottom], mode="reflect")
        del right
        del bottom
        gc.collect()
        return batch, _h, _w

    def post_batch(batch, _h, _w):
        batch = batch[:, :, 0:_h, 0:_w]
        batch = (
            batch.mul(255).round().clamp(0, 255).permute(0, 2, 3, 1)
        )  # (B, C, H, W) -> (B, H, W, C) still float32, numpy is done on async put_batch_func thread
        del _h, _w
        gc.collect()
        return batch

    tasks = list()

    def put_batch_func(batch, data):
        try:
            logger.info(f"TPU-{index}/ASYNC: placing batch")
            cpu_batch = batch.to(
                device="cpu", dtype=torch.uint8
            ).numpy()  # numpy, uint8, 0-255
            out_data_pipe.put(
                {
                    "value": cpu_batch,
                    "meta": {
                        "batch_id": data["meta"]["batch_id"],
                        "aw_worker_index": data["meta"]["aw_worker_index"],
                    },
                }
            )
            logger.info(f"TPU-{index}/ASYNC: batch on out_data_pipe")
        except Exception as e:
            logger.error(f"TPU-{index}/ASYNC: batch place failed: {e}")
            logger.error(traceback.format_exc())

    def put_batch(batch, data):
        task_thead = threading.Thread(target=put_batch_func, args=(batch, data))
        task_thead.start()
        tasks.append(task_thead)

    logger.info(f"TPU-{index}: funcs defined")
    first_run = True  # first run is always compilation
    while True:
        try:
            gc_obj = 0
            logger.info(f"TPU-{index}: waiting for data")
            data = in_data_pipe.get()  # (B, H, W, C), numpy array, uint8, 0-255
            if data is None:
                break
            if first_run:
                logger.info(f"TPU-{index}: first run, is always compilation")
            logger.info(f"TPU-{index}: data received")

            init_time = time.time()
            value = torch.from_numpy(data["value"]) # (B, H, W, C), torch tensor, uint8, 0-255
            value, _h, _w = prep_batch(value) # (B, C, H, W), torch tensor, float32, 0-1
            logger.info(f"TPU-{index}: data prep in {round(time.time() - init_time, 4)} seconds")
            batch = model(value)
            logger.info(f"TPU-{index}: model run in {round(time.time() - init_time, 4)} seconds")
            batch = post_batch(batch, _h, _w)
            logger.info(f"TPU-{index}: data post in {round(time.time() - init_time, 4)} seconds")
            finish_time = time.time()

            if first_run:
                first_run = False
                logger.info(
                    f"TPU-{index}: compilation done in {round(finish_time - init_time, 4)} seconds, out shape {batch.shape}"
                )
            else:
                logger.info(
                    f"TPU-{index}: data processed in {round(finish_time - init_time, 4)} seconds, out shape {batch.shape}"
                )

            logger.info(f"TPU-{index}: starting async put_batch")
            put_batch(batch, data)
            logger.info(f"TPU-{index}: async put_batch started")

            del value, batch, data, init_time, finish_time, _h, _w
            gc_obj += gc.collect()
            logger.info(
                f"TPU-{index}: in queue size: {in_data_pipe.qsize()}, out queue size: {out_data_pipe.qsize()}"
            )
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

    def save_video_func(fps, final_out, metadata, vid_id):
        logger.info(f"ASSIGN/ASYNC-{index}: {vid_id} - I have been summoned")
        # final out is the tensor
        with tempfile.NamedTemporaryFile(suffix=".mp4") as temp:
            out = cv2.VideoWriter(
                temp.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (C_W, C_H)
            )
            written_frames = 0
            for frame in final_out:
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(bgr_frame)
                written_frames += 1
            logger.info(f"ASSIGN/ASYNC-{index}: {vid_id} - {written_frames} frames written")

            out_frame_count = int(written_frames)
            out_height = int(final_out.shape[1])
            out_width = int(final_out.shape[2])
            out_fps = fps
            out_duration = out_frame_count / out_fps

            swap_meta = {
                "cv_frame_count": out_frame_count,
                "cv_height": out_height,
                "cv_width": out_width,
                "cv_fps": out_fps,
                "cv_duration": out_duration,
            }

            metadata = {**metadata, **swap_meta}
            out.release()
            temp.flush()

            with open(temp.name, "rb") as f:
                final_out = f.read()  # mp4 out bytes
        logger.info(
            f"ASSIGN/ASYNC-{index}: {vid_id} - video written, {len(final_out)/1024/1024} MB"
        )

        metadata_bytes = json.dumps(metadata).encode("utf-8")  # json metadata bytes
        tar_pipe.put((vid_id, final_out, metadata_bytes, metadata))  # str, bytes, bytes
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

            # order
            output_superbatch.sort(key=lambda x: x["o"])
            output_superbatch = [x["v"] for x in output_superbatch]
            logger.info(f"ASSIGN/PROC-{index}: {vid_id} - all batches received")
            final_out = np.concatenate(output_superbatch, axis=0)
            logger.info(
                f"ASSIGN/PROC-{index}: {vid_id} - final output shape {final_out.shape}, writing video..."
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
            logger.info(f"ASSIGN/MANAGER-{index}: worker started")
            child_process.start()
            logger.info(f"ASSIGN/MANAGER-{index}: worker joined")
            child_process.join()
            gc.collect()
            logger.info(f"ASSIGN/MANAGER-{index}: worker exited")
        except Exception as e:
            logger.error(f"ASSIGN/MANAGER-{index}: ERROR - {e}")
            logger.error(traceback.format_exc())
            continue

    logger.info(f"ASSIGN/MANAGER-{index}: exiting")


def tar_worker_func(
    index: int, tar_pipe: mp.Queue, wandb_pipe: mp.Queue, tar_id: mp.Value
):
    processed_tasks = 0

    @timeout(TAR_INDIV_TIMEOUT)
    def add2tar(files_tuple, tar):
        vid_id, vid_bytes, meta_bytes = files_tuple
        logger.info(f"TAR-{index}: {vid_id} - writing to tar")
        tarinfo = tarfile.TarInfo(name=f"{vid_id}.mp4")
        tarinfo.size = len(vid_bytes)
        tar.addfile(tarinfo, io.BytesIO(vid_bytes))
        logger.info(f"TAR-{index}: {vid_id} - mp4 written to tar")
        tarinfo = tarfile.TarInfo(name=f"{vid_id}.json")
        tarinfo.size = len(meta_bytes)
        tar.addfile(tarinfo, io.BytesIO(meta_bytes))
        logger.info(f"TAR-{index}: {vid_id} - meta written to tar")

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

            logger.info(f"TAR-{index}: waiting for data")
            vid_id, vid_bytes, meta_bytes, meta = tar_pipe.get()
            logger.info(f"TAR-{index}: got data")

            if all(x is None for x in [vid_id, vid_bytes, meta_bytes, meta]):
                logger.info(f"TAR-{index}: got None, exiting")
                if USE_WANDB:
                    wandb_pipe.put(None)
                break

            files_tar.append((vid_id, vid_bytes, meta_bytes))
            files_size += len(vid_bytes) + len(meta_bytes)
            if USE_WANDB:
                wandb_pipe.put(
                    {
                        "type": "video_entry",
                        "data": (
                            vid_id,
                            meta["cv_fps"],
                            meta["cv_duration"],
                            meta["cv_frame_count"],
                        ),
                    }
                )

            logger.info(
                f"TAR-{index}: {vid_id} - added to tar, size {files_size/1024/1024} MB"
            )

            if files_size > TAR_SIZE:
                with tar_id.get_lock():
                    tar_id.value += 1
                    tar_id_val = tar_id.value
                logger.info(
                    f"TAR-{index}: {tar_id_val} - tar size exceeded, writing to disk"
                )

                tar_id_val = str(tar_id_val).zfill(6)

                # make sure we dont use the ones from current itteration
                del vid_id, vid_bytes, meta_bytes, meta

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
            logger.error(f"TAR-{index}: {vid_id} - ERROR - {e}")
            logger.error(traceback.format_exc())
            continue

def tar_worker_manager(
    index: int, tar_pipe: mp.Queue, wandb_pipe: mp.Queue, tar_id: mp.Value
):
    logger.info(f"TAR/MANAGER-{index}: started")

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
        "IM2IM_MODEL_PATH": IM2IM_MODEL_PATH,
        "C_C": C_C,
        "C_H": C_H,
        "C_W": C_W,
        "FILE_PIPE_MAX": FILE_PIPE_MAX,
        "IN_DATA_PIPE_MAX": IN_DATA_PIPE_MAX,
        "OUT_DATA_PIPE_MAX": OUT_DATA_PIPE_MAX,
        "TAR_PIPE_MAX": TAR_PIPE_MAX,
        "WANDB_PROJ": WANDB_PROJ,
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
            target=tar_worker_func,
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

    xmp.spawn(
        tpu_worker_func,
        args=(
            in_data_pipe,
            tpu_out_data_pipe,
            tmp_c,
        ),
        nprocs=TPU_CORE_COUNT,
    )

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

    if USE_WANDB:
        wandb_pipe.put(None)
        wandb_worker_process.join()

    logger.info("MAIN: finished")

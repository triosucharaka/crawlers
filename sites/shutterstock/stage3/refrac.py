import multiprocessing as mp
import time
import json
import wandb
import math
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch
from im2im.main import load_model
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

mp.set_start_method("spawn", force=True)

IN_DISK_PATH = "/home/windowsuser/mount-folder/tempofunkds/shutterstock/stage2/"
OUT_DISK_PATH = "/home/windowsuser/mount-folder/tempofunkds/shutterstock/stage3/"

JSON_MAP_PATH = "/home/windowsuser/crawlers/sites/shutterstock/stage2/map.json"

TMP_PTH = "/home/windowsuser/crawlers/sites/shutterstock/stage3/tmp"

## Wandb

global USE_WANDB

USE_WANDB = True
WANDB_ENTITY = "tempofunk" # none if not using wandb
WANDB_PROJ = "shutterstock_stage3"
WANDB_NAME = f"mondaynight_{int(time.time())}"
WANDB_ONLY_ONE_CORE = True
TAR_SIZE = 128 * 1024 * 1024 # 512MB

ASSIGN_WORKER_COUNT = 2
TAR_WORKER_COUNT = 2

## TPU Workers
TPU_CORE_COUNT = 8
TPU_BATCH_SIZE = 16
MAX_SUPERBATCHES = 30
IM2IM_MODEL_PATH = "im2im-sswm"

## Pipes
FILE_PIPE_MAX = 200
IN_DATA_PIPE_MAX = 400
OUT_DATA_PIPE_MAX = 400
TAR_PIPE_MAX = 200
WANDB_PIPE_MAX = 1000

C_C = 3
C_H = 384 # (divisible by 64)
C_W = 640 # (divisible by 64)

LOG_MEMORY = True

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

if LOG_MEMORY:
    logger = CustomLogger(logger, {})

def wds_reader_func(file_pipe: mp.Queue):
    logger.info("wds-r: started")

    json_map = json.load(open(JSON_MAP_PATH, "r"))
    tar_map = list()

    for tar in json_map:
        tar_map.append(IN_DISK_PATH + tar)

    dataset = wds.WebDataset(tar_map)

    for sample in dataset:
        logger.info(f"wds-r: sending {sample['__key__']}")
        file_pipe.put((
            sample["__key__"],
            sample["mp4"],
            sample["json"]
        ))

    file_pipe.put((None, None, None))

    logger.info("wds-r: finished")

def tpu_worker_func(index, in_data_pipe: mp.Queue, out_data_pipe: mp.Queue, tmp_c: mp.Value):
    logger.info(f'tw-{index}: started, grabbing device {index}')
    device = xm.xla_device()
    logger.info(f'tw-{index}: device {index} successfully grabbed')
    model = load_model(IM2IM_MODEL_PATH, device)
    logger.info(f'tw-{index}: model loaded')

    with tmp_c.get_lock():
        tmp_c.value += 1

    def prep_batch(batch):
        # batch should be a torch tensor, uint8, 0-255, (B, H, W, C)
        batch = batch.permute(0,3,1,2) # (B, H, W, C) -> (B, C, H, W)
        batch = batch.to(device).to(torch.float32).div(255).to(memory_format=torch.contiguous_format) # batch is now a tensor, float32, 0-1
        _bs, _c, _h, _w = batch.shape
        right = math.ceil(_w / model.ksize) * model.ksize - _w
        bottom = math.ceil(_h / model.ksize) * model.ksize - _h
        batch = torch.nn.functional.pad(batch, [0, right, 0, bottom], mode = 'reflect')
        del right
        del bottom
        gc.collect()
        return batch, _h, _w
    
    def post_batch(batch, _h, _w):
        batch = batch[:,:,0:_h,0:_w]
        batch = batch.mul(255).round().clamp(0,255).permute(0,2,3,1) # (B, C, H, W) -> (B, H, W, C) still float32
        del _h, _w
        gc.collect()
        return batch

    tasks = list()

    def put_batch_func(batch, data):
        try:
            print(f'tw-{index}: putting batch')
            cpu_batch = batch.to(device = 'cpu', dtype = torch.uint8).numpy() # numpy, uint8, 0-255
            out_data_pipe.put({
                'value': cpu_batch, 
                'meta': {
                    'batch_id': data['meta']['batch_id'], 
                    'aw_worker_index': data['meta']['aw_worker_index']
                    }
                }
            )
            print(f'tw-{index}: batch put')
        except Exception as e:
            print(f'tw-{index}: batch put failed: {e}')
            traceback.print_exc()
            print(e)

    def put_batch(batch, data):
        task_thead = threading.Thread(target = put_batch_func, args = (batch, data))
        task_thead.start()
        tasks.append(task_thead)

    logger.info(f'tw-{index}: init signal received')
    first_run = True # first run is always compilation 
    while True:
        gc_obj = 0
        logger.info(f'tw-{index}: waiting for data')
        data = in_data_pipe.get() # (B, H, W, C)
        if data is None:
            break
        if first_run:
            logger.info(f'tw-{index}: first run, is always compilation')
        logger.info(f'tw-{index}: data received')

        init_time = time.time()
        value = torch.from_numpy(data['value'])
        value, _h, _w = prep_batch(value)
        logger.info(f'tw-{index}: data prep in {time.time() - init_time} seconds')
        batch = model(value)
        logger.info(f'tw-{index}: model run in {time.time() - init_time} seconds')
        batch = post_batch(batch, _h, _w)
        logger.info(f'tw-{index}: data post in {time.time() - init_time} seconds')
        finish_time = time.time()

        if first_run:
            first_run = False
            logger.info(f'tw-{index}: compilation done in {finish_time - init_time} seconds, out shape {batch.shape}')
        else:
            logger.info(f'tw-{index}: data processed in {finish_time - init_time} seconds, out shape {batch.shape}')

        logger.info(f'tw-{index}: triggering put_batch')
        put_batch(batch, data)
        logger.info(f'tw-{index}: put_batch triggered')

        del value, batch, data, init_time, finish_time, _h, _w
        gc_obj += gc.collect()
        logger.info(f'tw-{index}: data processed, {gc_obj} objects collected, {len(tasks)} tasks in queue')
        logger.info(f'tw-{index}: in queue size: {in_data_pipe.qsize()}, out queue size: {out_data_pipe.qsize()}')

    for task in tasks:
        task.join()

    logger.info(f'tw-{index}: finished')

def assign_worker_func(index: int, file_pipe: mp.Queue, in_data_pipe: mp.Queue, out_data_pipe: mp.Queue, tar_pipe: mp.Queue):
    logger.info(f"aw-{index}: started")

    while True:
        gc_obj = 0

        logger.info(f"aw-{index}: {index} waiting for file")
        vid_id, mp4_bytes, metadata = file_pipe.get()
        logger.info(f"aw-{index}: {vid_id} received")

        if vid_id is None and mp4_bytes is None and metadata is None:
            logger.info(f"aw-{index}: termination signal received")
            for i in range(TPU_CORE_COUNT):
                in_data_pipe.put(None, timeout=1)
                tar_pipe.put((None, None, None, None), timeout=1)
            break

        metadata = json.loads(metadata.decode("utf-8"))

        logger.info(f"aw-{index}: {vid_id} loading video")
        with tempfile.NamedTemporaryFile() as temp:
            temp.write(mp4_bytes)
            temp.flush()
            video_cap = cv2.VideoCapture(temp.name)

            resized_frames = []

            logger.info(f"aw-{index}: {vid_id} resizing video")
            while True:
                ret, frame = video_cap.read()
                if not ret:
                    break
                resized_frame = cv2.resize(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                    (C_W, C_H)
                )
                numpy_frame = np.array(resized_frame)
                resized_frames.append(numpy_frame)
            logger.info(f"aw-{index}: {vid_id} video resized")

            frames = np.array(resized_frames) # (T, H, W, C)
            frame_count = frames.shape[0]
            fps = video_cap.get(cv2.CAP_PROP_FPS)

            del video_cap, resized_frames, resized_frame, numpy_frame, frame, mp4_bytes

        logger.info(f"aw-{index}: {vid_id} video loaded")

        if frame_count < TPU_BATCH_SIZE:
            logger.info(f"aw-{index}: {vid_id} - {frame_count} < {TPU_BATCH_SIZE}, skipping")
            continue

        # batch = (f, f, f, f, .. 16 times) # 16 frames
        # superbatch = (b, b, b, b, ...Y times) # Y splits where Y is the frame_count // TPU_BATCH_SIZE

        # Divide video into batches
        superbatch_count = frame_count // TPU_BATCH_SIZE
        #superbatch_count = superbatch_count // TPU_CORE_COUNT
        if superbatch_count > MAX_SUPERBATCHES:
            superbatch_count = MAX_SUPERBATCHES
        logger.info(f"aw-{index}: {vid_id} - using {superbatch_count * TPU_BATCH_SIZE}/{frame_count} frames, {superbatch_count} megabatches")

        batches_sent = 0

        logger.info(f"aw-{index}: {vid_id} - sending batches to TPU workers")
        for i in range(superbatch_count):
            batch = frames[range(i * TPU_BATCH_SIZE, (i + 1) * TPU_BATCH_SIZE)]
            #batch = batch.permute(0,2,3,1) # (B, C, H, W) -> (B, H, W, C)
            #batch = batch.numpy(force=True)
            assert batch.shape[3] == C_C
            assert batch.shape[0] == TPU_BATCH_SIZE
            assert batch.dtype == np.uint8
            assert batch.min() >= 0
            assert batch.max() <= 255
            batch = {'value': batch, 'meta': {'batch_id': i, 'aw_worker_index': index, 'vid_id': vid_id}}
            in_data_pipe.put(batch) 
            # I EXPECT (B, H, W, C) AKA (16, 256, 256, 3)!!!!
            # batch should be a numpy array, uint8, 0-255, (B, H, W, C)
            logger.info(f"aw-{index}: {vid_id} - batch {i} sent with sahpe {batch['value'].shape}")
            batches_sent += 1
            del batch
        logger.info(f"aw-{index}: {vid_id} - bastches sent to TPU workers")

        del frames
        gc_obj += gc.collect()

        output_superbatch = list()

        while batches_sent != 0:
            batch = out_data_pipe.get()
            if batch['meta']['aw_worker_index'] != index:
                out_data_pipe.put(batch)
                # sleep for (20-100 ms)
                logger.info(f"aw-{index}: {vid_id} - obtained a workload meant for {batch['meta']['aw_worker_index']}, sleeping for a bit (20-100 ms")
                rand_time = np.random.randint(20, 100) / 1000
                time.sleep(rand_time)
                continue
            batch_id = batch['meta']['batch_id']
            batches_sent -= 1
            # add batch at the correct index
            output_superbatch.append({"o": batch_id, "v": batch['value']})
            logger.info(f"aw-{index}: {vid_id} - batch {batch_id} received")
            del batch, batch_id
        # order
        output_superbatch.sort(key=lambda x: x['o'])
        output_superbatch = [x['v'] for x in output_superbatch]
        logger.info(f"aw-{index}: {vid_id} - all batches received")
        final_out = np.concatenate(output_superbatch, axis=0)
        logger.info(f"aw-{index}: {vid_id} - final output shape {final_out.shape}, writing video...")

        with tempfile.NamedTemporaryFile(suffix='.mp4') as temp:
            out = cv2.VideoWriter(temp.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (C_W, C_H))
            written_frames = 0
            for frame in final_out:
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(bgr_frame)
                written_frames += 1
            logger.info(f"aw-{index}: {vid_id} - {written_frames} frames written")
            out_frame_count = int(written_frames)
            out_height = int(final_out.shape[1])
            out_width = int(final_out.shape[2])
            out_fps = fps
            out_duration = out_frame_count / out_fps

            swap_meta = {
                'cv_frame_count': out_frame_count,
                'cv_height': out_height,
                'cv_width': out_width,
                'cv_fps': out_fps,
                'cv_duration': out_duration,
            }

            metadata = {**metadata, **swap_meta}
            out.release()
            temp.flush()

            with open(temp.name, 'rb') as f:
                final_out = f.read() # mp4 out bytes
        logger.info(f"aw-{index}: {vid_id} - video written, {len(final_out)/1024/1024} MB")

        metadata_bytes = json.dumps(metadata).encode('utf-8') # json metadata bytes

        del output_superbatch, out, out_frame_count, out_height, out_width, out_fps, out_duration, swap_meta

        logger.info(f"avail meta: {metadata} ")
        tar_pipe.put((vid_id, final_out, metadata_bytes, metadata)) # str, bytes, bytes
        logger.info(f"aw-{index}: {vid_id} - sent to tar worker")

        del final_out, metadata, metadata_bytes

        gc_obj += gc.collect()
        print(f"aw-{index}: {vid_id} - done, {gc_obj} objects collected")

        del vid_id

def tar_worker_func(index: int, tar_pipe: mp.Queue, wandb_pipe: mp.Queue, tar_id: mp.Value):
    files_tar = list()
    files_size = 0

    logger.info(f"tar-{index}: started")

    while True:
        logger.info(f"tar-{index}: waiting for data")
        vid_id, vid_bytes, meta_bytes, meta = tar_pipe.get()
        logger.info(f"tar-{index}: got data")

        if all(x is None for x in [vid_id, vid_bytes, meta_bytes, meta]):
            logger.info(f"tar-{index}: got None, exiting")
            if USE_WANDB:
                wandb_pipe.put(None)
            break

        files_tar.append((vid_id, vid_bytes, meta_bytes))
        files_size += len(vid_bytes) + len(meta_bytes)
        if USE_WANDB:
            wandb_pipe.put({"type": "video_entry", "data": (vid_id, meta['cv_fps'], meta['cv_duration'], meta['cv_frame_count'])})

        logger.info(f"tar-{index}: {vid_id} - added to tar, size {files_size/1024/1024} MB")

        if files_size > TAR_SIZE:
            with tar_id.get_lock():
                tar_id.value += 1
                tar_id_val = tar_id.value
            logger.info(f"tar-{index}: {tar_id_val} - tar size exceeded, writing to disk")

            tar_id_val = str(tar_id_val).zfill(6)

            # make sure we dont use the ones from current itteration
            del vid_id, vid_bytes, meta_bytes, meta

            with tarfile.open(name=f"{OUT_DISK_PATH}/{tar_id_val}.tar", mode='w') as tar:
                for vid_id, vid_bytes, meta_bytes in files_tar:
                    tarinfo = tarfile.TarInfo(name=f"{vid_id}.mp4")
                    tarinfo.size = len(vid_bytes)
                    tar.addfile(tarinfo, io.BytesIO(vid_bytes))
                    tarinfo = tarfile.TarInfo(name=f"{vid_id}.json")
                    tarinfo.size = len(meta_bytes)
                    tar.addfile(tarinfo, io.BytesIO(meta_bytes))
            files_tar = list()
            files_size = 0

            logger.info(f"tar-{index}: {tar_id_val} - tar written to disk")

            if USE_WANDB:
                wandb_pipe.put({"type": "tar_entry", "data": None})

def wandb_worker_func(wandb_pipe: mp.Queue, tmp_c: mp.Value, file_pipe: mp.Queue, i_dp: mp.Queue, o_dp: mp.Queue, t_p: mp.Queue):
    run = wandb.init(
        project=WANDB_PROJ, 
        entity=WANDB_ENTITY,
        name=WANDB_NAME
        )
    init_time = time.time()

    frames = 0
    hours = 0
    videos = 0
    tars = 0

    while True:
        logger.info(f"total tpus: {int(tmp_c.value)}")

        data = wandb_pipe.get()

        elapsed_time = int(time.time() - init_time)

        run.log({
            "f_p": file_pipe.qsize(),
            "i_dp": i_dp.qsize(),
            "o_dp": o_dp.qsize(),
            "t_p": t_p.qsize(),
        })

        if data is None:
            break

        if data["type"] == "video_entry":
            video_id, fps, duration, frame_count = data["data"]
            logger.info(f"fabella vid_id: {video_id}, fps: {fps}, duration: {duration}, frame_count: {frame_count}")
            videos += 1
            frames += frame_count
            hours += duration / 3600
            logger.info(f"fabella to send: {videos}, {frames}, {hours}")
            run.log({"videos": videos, "frames": frames, "hours": hours}, step = elapsed_time)

        elif data["type"] == "tar_entry":
            tars += 1
            run.log({"tars": tars}, step = elapsed_time)

        # speed report
        if elapsed_time > 0:
            run.log({"fps": frames / elapsed_time, "vps": videos / elapsed_time, "tps": tars / elapsed_time}, step = elapsed_time)

    run.finish()

if __name__ == "__main__":
    logger.info("main: started")

    manager = mp.Manager()

    file_pipe = manager.Queue(maxsize=FILE_PIPE_MAX)
    in_data_pipe = manager.Queue(maxsize=IN_DATA_PIPE_MAX)
    out_data_pipe = manager.Queue(maxsize=OUT_DATA_PIPE_MAX)
    tar_pipe = manager.Queue(maxsize=TAR_PIPE_MAX)

    tmp_c = mp.Value('i', 0)

    if USE_WANDB:
        wandb_pipe = manager.Queue(maxsize=WANDB_PIPE_MAX)
        wandb_worker_process = mp.Process(target=wandb_worker_func, args=(wandb_pipe, tmp_c, file_pipe, in_data_pipe, out_data_pipe, tar_pipe,))
        wandb_worker_process.start()
    else:
        wandb_pipe = None

    wds_worker_process = mp.Process(target=wds_reader_func, args=(file_pipe,))

    assign_worker_processes = list()
    for i in range(ASSIGN_WORKER_COUNT):
        assign_worker_process = mp.Process(target=assign_worker_func, args=(i, file_pipe, in_data_pipe, out_data_pipe, tar_pipe,))
        assign_worker_processes.append(assign_worker_process)

    tar_id = mp.Value('i', 0)
    tar_worker_processes = list()
    for i in range(TAR_WORKER_COUNT):
        tar_worker_process = mp.Process(target=tar_worker_func, args=(i, tar_pipe, wandb_pipe, tar_id,))
        tar_worker_processes.append(tar_worker_process)

    # tpu_worker_processes = list()
    # for i in range(TPU_CORE_COUNT):
    #     tpu_worker_process = mp.Process(target=tpu_worker_func, args=(i, in_data_pipe, out_data_pipe,))
    #     tpu_worker_processes.append(tpu_worker_process)

    wds_worker_process.start()
    for assign_worker_process in assign_worker_processes:
        assign_worker_process.start()
    for tar_worker_process in tar_worker_processes:
        tar_worker_process.start()    
    # for tpu_worker_process in tpu_worker_processes:
    #     tpu_worker_process.start()

    xmp.spawn(tpu_worker_func, args=(in_data_pipe, out_data_pipe, tmp_c,), nprocs=TPU_CORE_COUNT)

    logger.info("main: waiting for workers to finish")
    wds_worker_process.join()
    logger.info("main: wds_worker_process finished")
    # for tpu_worker_process in tpu_worker_processes:
    #     tpu_worker_process.join()
    for assign_worker_process in assign_worker_processes:
        assign_worker_process.join()
        logger.info("main: SOME assign_worker_process finished")
    logger.info("main: assign_worker_processes finished")
    for tar_worker_process in tar_worker_processes:
        tar_worker_process.join()
        logger.info("main: SOME tar_worker_process finished")
    logger.info("main: tar_worker_processes finished")

    if USE_WANDB:
        wandb_pipe.put(None)
        wandb_worker_process.join()

    logger.info("main: finished")
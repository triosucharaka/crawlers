import multiprocessing as mp
import time
import json
import wandb
import os
import numpy as np
import io
import tarfile
import gc
import logging
import psutil
import argparse
import threading
import traceback
import jax.numpy as jnp
import numpy as np
import jax
from transformers import CLIPTokenizer, FlaxCLIPTextModel
from wrapt_timeout_decorator import *

mp.set_start_method("spawn", force=True)

# get instance id from args
parser = argparse.ArgumentParser()
parser.add_argument("--instance", type=int, required=True)
args = parser.parse_args()
INSTANCE = args.instance

### Configuration ###

## Paths
IN_DISK_PATH = "/home/windowsuser/mount-folder/tempofunkds/shutterstock/stage4/"
OUT_DISK_PATH = "/home/windowsuser/mount-folder/tempofunkds/shutterstock/stage4/"
JSON_MAP_PATH = "/home/windowsuser/mount-folder/tempofunkds/shutterstock/global/latents_map.json"

NPY_EXTENSION = "cliptext" # ex.: 00001.cliptext (not .npy, to differentiate with the text embeds on stage 5)

## Wandb
global USE_WANDB
USE_WANDB = True
WANDB_ENTITY = "peruano"  # none if not using wandb
WANDB_PROJ = "debug_shutterstock_stage5"
WANDB_NAME = f"mass_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
LOG_MEMORY = True
SPAWN_ON_EVERYTHING = False # wandb only grabs the current process logger, not global one, and also logs everything thats printed such as errors

## Multiprocessing
ASSIGN_WORKER_COUNT = 1
ASSIGN_WORKER_MAX_TASKS_PER_CHILD = 10
TAR_WORKER_COUNT = 1
TAR_SIZE = 128 * 1024 * 1024  # 512MB
TAR_INDIV_TIMEOUT = 15 # seconds
TAR_MTPC = 5

## TPU Workers
TPU_CORE_COUNT = 4
TPU_BATCH_SIZE = 64

## Model Parameters
### CLIP Tokenizer
CLIP_TOKENIZER_MODEL_PATH = "CompVis/stable-diffusion-v1-4"
CLIP_TOKENIZER_MODEL_REVISION = "flax"
CLIP_TOKENIZER_MODEL_SUBFOLDER = "tokenizer"
### CLIP Text Model
CLIP_TEXT_MODEL_PATH = "CompVis/stable-diffusion-v1-4"
CLIP_TEXT_MODEL_REVISION = "flax"
CLIP_TEXT_MODEL_SUBFOLDER = "text_encoder"

## Pipes
FILE_PIPE_MAX = 512
IN_DATA_PIPE_MAX = 1024
OUT_DATA_PIPE_MAX = 1024
TAR_PIPE_MAX = 1024
WANDB_PIPE_MAX = 1000

### End Configuration ###


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
    json_map = json.load(open(JSON_MAP_PATH, "r"))[str(INSTANCE)]

    if SPAWN_ON_EVERYTHING:
        run = probe("wds_reader")

    for fileid in json_map:
        try:
            logger.info(f"WDS: sending {fileid}")

            json_bytes = open(f"{IN_DISK_PATH}/{fileid}.json", "rb").read()

            file_pipe.put((fileid, json_bytes))
        except Exception as e:
            logger.error(f"WDS: {fileid} ERROR - {e}")
            logger.error(traceback.format_exc())
            continue

    file_pipe.put((None, None, None))
    logger.info("WDS: finished")


def tpu_worker_func(
    index, in_data_pipe: mp.Queue, out_data_pipe: mp.Queue, tmp_c: mp.Value
):
    logger.info(f"TPU-{index}: started")

    if SPAWN_ON_EVERYTHING:
        run = probe(f"tpu_worker_{index}")

    weight_dtype = jnp.bfloat16
    
    text_encoder = FlaxCLIPTextModel.from_pretrained(
        CLIP_TEXT_MODEL_PATH, 
        revision=CLIP_TEXT_MODEL_REVISION,
        subfolder=CLIP_TEXT_MODEL_SUBFOLDER,
        dtype=weight_dtype)

    logger.info(f"TPU-{index}: model loaded")

    output_expected_shape = (TPU_CORE_COUNT, TPU_BATCH_SIZE, 77, 768)
    input_expected_shape = (TPU_CORE_COUNT, TPU_BATCH_SIZE, 77)
    # assuming we use the fixed 77-token length
    # TODO: check if we should use the max len or some other value

    with tmp_c.get_lock():
        tmp_c.value += 1

    # NOTE: removed prep_batch as it's not needed.
    
    def encode(input_ids: jnp.array, attention_mask: jnp.array):
        # only returns `last_hidden_state`, that's what the diffusers
        # SD pipeline uses too, so I guess it's that
        return text_encoder(input_ids, attention_mask=attention_mask)[0]

    model = jax.pmap(encode, in_axes=(0, 0,))

    tasks = list()

    def put_batch_func(tensor, data_batch):
        try:
            logger.info(f"TPU-{index}/ASYNC: placing batch")
            
            # given tensor is (TPU_CORE_COUNT, TPU_BATCH_SIZE, 77, 768)
            # so tensor placed on outpipe is (TPU_BATCH_SIZE, 77, 768)
            output = np.array(tensor)

            assert output.shape == output_expected_shape

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
            input_ids_batch = list()
            attention_mask_batch = list()
            data_batch = list()

            logger.info(f"TPU-{index}: waiting for data")
            for i in range(TPU_CORE_COUNT):
                data = in_data_pipe.get()  # (B, H, W, C), numpy array, uint8, 0-255
                if data is None:
                    work_done = True
                    break
                """
                {
                    "value": {
                        "input_ids": np.array,
                        "attention_mask": np.array,
                    },
                    "meta": {
                        "key": str,
                    },
                }
                """
                input_ids_batch.append(data['value']['input_ids'])
                attention_mask_batch.append(data['value']['attention_mask'])
                data_batch.append(data['meta'])
                logger.info(f"TPU-{index}: data received, {i+1}/{TPU_CORE_COUNT}")

            if work_done:
                break

            input_ids_batch = np.stack(input_ids_batch)
            attention_mask_batch = np.stack(attention_mask_batch)

            assert input_ids_batch.shape == input_expected_shape
            assert attention_mask_batch.shape == input_expected_shape

            logger.info(f"TPU-{index}: data received")

            if first_run:
                logger.info(f"TPU-{index}: first run, is always compilation")
            
            total_texts = input_ids_batch.shape[0] * input_ids_batch.shape[1]
            init_time = time.time()
            # encode(input_ids: jnp.array, attention_mask: jnp.array):
            output = model(input_ids_batch, attention_mask_batch).block_until_ready()
            modeling_time = time.time() - init_time

            if first_run:
                logger.info(f"TPU-{index}: compiled in {modeling_time} seconds")
                first_run = False
            else:
                logger.info(f"TPU-{index}: modeled in {modeling_time} seconds - fps: {total_texts / modeling_time}")

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

    tokenizer = CLIPTokenizer.from_pretrained(
        CLIP_TOKENIZER_MODEL_PATH, 
        revision=CLIP_TOKENIZER_MODEL_REVISION,
        subfolder=CLIP_TOKENIZER_MODEL_SUBFOLDER)

    processed_tasks = 0
    tasks = []
    vid_id = None

    def save_func(vid_id,vid_meta, vid_embeds):
        logger.info(f"ASSIGN/ASYNC-{index}: {vid_id} - I have been summoned")

        embed_bytes = io.BytesIO()
        np.save(embed_bytes, vid_embeds, allow_pickle=False)
        embed_bytes = embed_bytes.getvalue()

        metadata_bytes = json.dumps(vid_meta).encode("utf-8")  # json metadata bytes
        
        tar_pipe.put((vid_id, metadata_bytes, embed_bytes))
        logger.info(f"ASSIGN/ASYNC-{index}: {vid_id} - sent to tar worker")

    def tar_save(vid_id: int, vid_meta: dict, vid_embeds: jnp.array):
        task_thead = threading.Thread(
            target=save_func, args=(vid_id, vid_meta, vid_embeds,)
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

            superbatch = list()

            ### Tokenizing and batching ###

            for a in range(TPU_CORE_COUNT):
                batch = list()

                for i in range(TPU_BATCH_SIZE):
                    logger.info(f"ASSIGN/PROC-{index}: loading video {i}")
                    obtained_obj = file_pipe.get()

                    if obtained_obj == (None, None):
                        logger.info(f"ASSIGN/PROC-{index}: termination signal received")
                        for i in range(TPU_CORE_COUNT):
                            in_data_pipe.put(None, timeout=1)
                            tar_pipe.put((None, None, None), timeout=1)
                            keep_restarting.value = 0
                        break

                    vid_id, metadata = obtained_obj

                    logger.info(f"ASSIGN/PROC-{index}: {vid_id} received")

                    metadata = json.loads(metadata.decode("utf-8"))
                    description = str(metadata['description'])

                    batch.append((vid_id, metadata, description))

                if keep_restarting.value == 0:
                    break

                tokens = tokenizer(
                    [x[2] for x in batch],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="np",
                )

                superbatch.append((
                    [x[0] for x in batch], # vid_id, list of strings
                    [x[1] for x in batch], # metadata, list of dicts
                    tokens # dict with keys: input_ids, attention_mask
                ))

            ### Send to TPUs ###

            for i in range(TPU_CORE_COUNT):
                in_data_pipe.put({
                    "value": {
                        "input_ids": superbatch[i][2]["input_ids"],
                        "attention_mask": superbatch[i][2]['attention_mask']
                    },
                    "meta": {
                        "superbatch_id": i,
                        "aw_worker_index": index,
                    }
                })

            ### Retrieve Output ###

            output_superbatch = list()

            for i in range(TPU_CORE_COUNT):
                output = out_data_pipe.get()
                assert output["meta"]["aw_worker_index"] == index

                output_superbatch.append({
                    "o": output["meta"]['superbatch_id'],
                    "v": output["value"] # just a jnp array
                })

            ### Organize and send ###

            output_superbatch.sort(key=lambda x: x["o"])
            output_superbatch = [x["v"] for x in output_superbatch]

            for i in range(TPU_CORE_COUNT):
                video_ids = superbatch[i][0]
                video_metas = superbatch[i][1]
                video_embeds = output_superbatch[i]

                assert len(video_ids) == len(video_metas) == video_embeds.shape[0]

                for z in range(len(video_ids)):
                    out_tuple = (
                        video_ids[z],       # int
                        video_metas[z],     # dict
                        video_embeds[z],    # jnp array
                    )

                    tar_save(*out_tuple)

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

#vid_id, vid_meta, vid_embeds = tar_pipe.get()

def tar_worker_func(
    index: int, tar_pipe: mp.Queue, wandb_pipe: mp.Queue, keep_restarting: mp.Value
):
    vid_id = None # for error handling without pipe.get
    processed_tasks = 0
    tasks = list()

    logger.info(f"TAR-{index}: started")

    def save_video_func(vid_id, meta_bytes, embed_bytes):
        logger.info(f"TAR-{index}: saving video {vid_id}")
        with open(f"{OUT_DISK_PATH}/{vid_id}.{NPY_EXTENSION}", "wb") as f:
            f.write(embed_bytes)
        logger.info(f"TAR-{index}: saved video {vid_id}")

    def save_video(vid_id, meta_bytes, embed_bytes):
        task_thead = threading.Thread(
            target=save_video_func, args=(vid_id, meta_bytes, embed_bytes,)
        )
        task_thead.start()
        return task_thead

    while processed_tasks <= TAR_MTPC:
        try:
            logger.info(f"TAR-{index}: waiting for data")
            vid_id, meta_bytes, embed_bytes = tar_pipe.get()
            logger.info(f"TAR-{index}: got data")

            if all(x is None for x in [vid_id, meta_bytes, embed_bytes]):
                logger.info(f"TAR-{index}: got None, exiting")
                keep_restarting.value = 0
                if USE_WANDB:
                    wandb_pipe.put(None)
                break

            logger.info(f"TAR-{index}: {vid_id} - saving video")
            _task = save_video(vid_id, meta_bytes, embed_bytes)
            tasks.append(_task)

            if USE_WANDB:
                wandb_pipe.put(
                    {
                        "type": "video_entry",
                        "data": (
                            vid_id
                        ),
                    }
                )

        except Exception as e:
            logger.error(f"TAR-{index}: {vid_id} - ERROR - {e}")
            logger.error(traceback.format_exc())
            continue

    logger.info(f"TAR-{index}: waiting for async tasks to finish")
    for task in tasks:
        task.join()

def tar_worker_manager(
    index: int, tar_pipe: mp.Queue, wandb_pipe: mp.Queue
):
    logger.info(f"TAR/MANAGER-{index}: started")

    if SPAWN_ON_EVERYTHING:
        run = probe(f"tar_manager_{index}")

    keep_restarting = mp.Value("i", 1)

    while keep_restarting.value == 1:
        try:
            tar_worker = mp.Process(
                target=tar_worker_func,
                args=(index, tar_pipe, wandb_pipe, keep_restarting,),
            )
            tar_worker.start()
            tar_worker.join()
            gc.collect()
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

    videos = 0

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
                video_id = data["data"]
                logger.info(
                    f"WANDB: video entry - vid_id: {video_id}"
                )
                videos += 1
                run.log(
                    {"videos": videos}, step=elapsed_time
                )

            # speed report
            if elapsed_time > 0:
                run.log(
                    {
                        "vps": videos / elapsed_time,
                    },
                    step=elapsed_time,
                )

            logger.info(
                f"WANDB: total stats - runtime: {convert_unix_timestamp(elapsed_time)} - {videos} videos, {round(videos / elapsed_time, 3)} vps"
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

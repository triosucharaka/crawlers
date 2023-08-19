import requests
import multiprocessing as mp
from threading import Thread
import io
import json
import traceback
import wandb
import sqlite3
import tarfile
import cv2
import time
import logging
import tempfile
import psutil
import os
# config

OPERATING_MODE = "process" # process, or thread

SQL_READ_PATH = "/home/windowsuser/crawlers/sites/shutterstock/stage2/shuffled.db"
SQL_MAX = 12500000 # 12.5m
SQL_WRITE_CHECKPOINT = 2

DOWNLOAD_WORKERS = 60
DOWNLOAD_TIMEOUT = 10
DOWNLOAD_RETRY_COUNT = 3
DOWNLOAD_MTPC = 30

DISK_PATH = "/home/windowsuser/mount-folder/tempofunkds/shutterstock/stage2"
TAR_WORKERS = 20

MAX_VIDEO = 100000

# pipes limits
SQL_PIPE_MAX = 4000
FILE_PIPE_MAX = 2000
WANDB_PIPE_MAX = 1000

TAR_BYTES = 512 * 1024 * 1024 # 512MB
TAR_MTPC = 100

USE_WANDB = True
WANDB_ENTITY = "tempofunk" # none if not using wandb
WANDB_PROJ = "shutterstock_stage2"
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

## callable functions

def validate_aspect_ratio(aspect_ratio):
    width, height = map(float, aspect_ratio.split(':'))
    if height > 1.2 * width or width > 2.0 * height:
        return False
    else:
        return True
    
def vid_extras(video_bytes: bytes):
    with tempfile.NamedTemporaryFile() as temp:
        temp.write(video_bytes)
        cap = cv2.VideoCapture(temp.name)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // fps
        cap.release()
        del cap
    return height, width, fps, duration, frame_count


## worker functions

def sql_reader_func(sql_pipe: mp.Queue):
    logger.info("sql-r: started")
    conn = sqlite3.connect(SQL_READ_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM shuffled_videos")
    logger.info("sql-r: query executed")

    row_count = 0
    while True:
        try:
            logger.info(f"sql-r: reading: {row_count}")
            row = cursor.fetchone()
            if row is None or row_count >= MAX_VIDEO:
                sql_pipe.put(None)
                logger.info("sql-r: reached max_video")
                break

            v_id, desc, dur, asr, v_url, auth, catg, fps, r18 = row 

            if not validate_aspect_ratio(asr):
                logger.info(f"sql-r: invalid aspect ratio: {v_id}, {asr}")
                continue

            metadata = {
                "id": v_id,
                "description": desc,
                "duration": dur,
                "aspect_ratio": asr,
                "video_url": v_url,
                "author": json.loads(auth),
                "category": json.loads(catg),
                "fps": fps,
                "r18": r18
            }

            sql_pipe.put((v_url, metadata))

            row_count += 1

            logger.info(f"sql-r: sent: {v_id}")
        except Exception as e:
            traceback.print_exc()
            logger.info(f"sql-r: failed: {e}")
            continue

    logger.info("sql-r: finished")

    sql_pipe.put(None)

def download_worker_func(index: int, sql_pipe: mp.Queue, file_pipe: mp.Queue, keep_restarting: mp.Value):
    logger.info(f"downloader-{index}: started")
    processed_tasks = 0

    while processed_tasks < DOWNLOAD_MTPC:
        try:
            data = sql_pipe.get()
            
            if data is None:
                logger.info(f"downloader-{index}: got None")
                sql_pipe.put(None)
                file_pipe.put(None)
                keep_restarting.value = 0
                break

            video_url, metadata = data

            for i in range(DOWNLOAD_RETRY_COUNT):
                try:
                    response = requests.get(video_url, timeout=DOWNLOAD_TIMEOUT)
                    if response.status_code != 200:
                        logger.info(f"downloader-{index}: failed: status code: {response.status_code}")
                        continue
                    break
                except Exception as e:
                    traceback.print_exc()
                    logger.info(f"downloader-{index}: failed: {e}")
                    continue
            else:
                logger.info(f"downloader-{index}: failed: max retries")
                continue
            
            file_pipe.put((response.content, metadata))
        except Exception as e:
            traceback.print_exc()
            logger.info(f"downloader-{index}: failed: {e}")
            continue
        finally:
            processed_tasks += 1

def tar_worker_func(index: int, file_pipe: mp.Queue, keep_restarting: mp.Value, wandb_pipe: mp.Queue = None):
    logger.info(f"tar-{index}: started")
    processed_tasks = 0
    while processed_tasks < TAR_MTPC:
        try:
            logger.info(f"tar-{index}: waiting for data")
            data = file_pipe.get()
            logger.info(f"tar-{index}: got data")

            if data is None:
                logger.info(f"tar-{index}: got None")
                file_pipe.put(None)
                if USE_WANDB:
                    wandb_pipe.put(None)
                keep_restarting.value = 0
                break

            video_bytes, metadata = data
            video_id = metadata["id"]

            try:
                metadata['cv_height'], metadata['cv_width'], metadata['cv_fps'], metadata['cv_duration'], metadata['cv_frame_count'] = vid_extras(video_bytes)
            except Exception:
                logger.info(f"tar-{index}: failed to open video in cv, {video_id}")
                continue
            
            # video
            with open(os.path.join(DISK_PATH, f"{video_id}.mp4"), "wb") as f:
                f.write(video_bytes)

            # metadata
            with open(os.path.join(DISK_PATH, f"{video_id}.json"), "w") as f:
                json.dump(metadata, f)

            if USE_WANDB:
                wandb_pipe.put({"type": "video_entry", "data": (video_id, metadata['cv_fps'], metadata['cv_duration'], metadata['cv_frame_count'])})

        except Exception as e:
            traceback.print_exc()
            logger.info(f"tar-{index}: failed: {e}")
            continue
        finally:
            processed_tasks += 1

def wandb_worker_func(wandb_pipe: mp.Queue):
    run = wandb.init(project=WANDB_PROJ, entity=WANDB_ENTITY)
    init_time = time.time()

    frames = 0
    hours = 0
    videos = 0

    while True:
        try:
            data = wandb_pipe.get()

            elapsed_time = int(time.time() - init_time)

            if data is None:
                break

            if data["type"] == "video_entry":
                video_id, fps, duration, frame_count = data["data"]
                videos += 1
                frames += frame_count
                hours += duration / 3600
                run.log({"videos": videos, "frames": frames, "hours": hours}, step = elapsed_time)

            # speed report
            if elapsed_time > 0:
                run.log({"fps": frames / elapsed_time, "vps": videos / elapsed_time}, step = elapsed_time)
        except Exception as e:
            traceback.print_exc()
            logger.info(f"wandb: failed: {e}")
            continue

    run.finish()

## manager functions

def download_manager_func(index: int, sql_pipe: mp.Queue, file_pipe: mp.Queue):
    logger.info(f"downloader-manager-{index}: started")

    keep_restarting = mp.Value('i', 1)

    while keep_restarting.value == 1:
        try:
            child_proc = mp.Process(
                target=download_worker_func,
                args=(
                    index,
                    sql_pipe,
                    file_pipe,
                    keep_restarting,
                )
            )
            child_proc.start()
            child_proc.join()
        except Exception as e:
            logger.error(f"downloader-manager-{index}: failed: {e}")
            logger.error(traceback.format_exc())
            continue

def tar_manager_func(index: int, file_pipe: mp.Queue, wandb_pipe: mp.Queue = None):
    logger.info(f"tar-manager-{index}: started")

    keep_restarting = mp.Value('i', 1)

    while keep_restarting.value == 1:
        try:
            child_proc = mp.Process(
                target=tar_worker_func,
                args=(
                    index,
                    file_pipe,
                    keep_restarting,
                    wandb_pipe
                )
            )
            child_proc.start()
            child_proc.join()
        except Exception as e:
            logger.error(f"tar-manager-{index}: failed: {e}")
            logger.error(traceback.format_exc())
            continue

def main():

    if OPERATING_MODE == "process":
        spawner = mp.Process
    elif OPERATING_MODE == "thread":
        spawner = Thread

    manager = mp.Manager()

    sql_pipe = manager.Queue(maxsize=SQL_PIPE_MAX)
    file_pipe = manager.Queue(maxsize=FILE_PIPE_MAX)
    if USE_WANDB:
        wandb_pipe = manager.Queue(maxsize=WANDB_PIPE_MAX)
    else:
        wandb_pipe = None

    logger.info("starting threads")

    sql_reader = spawner(target=sql_reader_func, args=(sql_pipe,))
    sql_reader.start()

    logger.info("started sql_reader")

    download_threads = [spawner(target=download_manager_func, args=(i, sql_pipe, file_pipe,)) for i in range(DOWNLOAD_WORKERS)]
    for thread in download_threads:
        thread.start()

    logger.info("started download_workers")

    tar_threads = [spawner(target=tar_manager_func, args=(i, file_pipe, wandb_pipe,)) for i in range(TAR_WORKERS)]
    for thread in tar_threads:
        thread.start()

    logger.info("started tar_workers")

    if USE_WANDB:
        wandb_thread = spawner(target=wandb_worker_func, args=(wandb_pipe,))
        wandb_thread.start()

    logger.info("started wandb_worker")
    logger.info("main: ready and waiting till full join")

    sql_reader.join()
    for thread in download_threads:
        thread.join()
    for thread in tar_threads:
        thread.join()
    if USE_WANDB:
        wandb_thread.join()

    logger.info("main: finished")

if __name__ == "__main__":
    main()
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
# config

#TODO: apply this: https://stackoverflow.com/questions/21485319/high-memory-usage-using-python-multiprocessing max tasks per child, turn funcs into callables or something, just make it so that it restarts

OPERATING_MODE = "process" # process, or thread

SQL_READ_PATH = "/home/windowsuser/crawlers/sites/shutterstock/database.db"
SQL_WRITE_PATH = "/home/windowsuser/crawlers/sites/shutterstock/dataset_map.db"
SQL_MAX = 12500000 # 12.5m
SQL_WRITE_CHECKPOINT = 2

DOWNLOAD_WORKERS = 60
DOWNLOAD_TIMEOUT = 10
DOWNLOAD_RETRY_COUNT = 3

DISK_PATH = "/home/windowsuser/mount-folder/tempofunkds/shutterstock/stage2"
TAR_WORKERS = 5

MAX_VIDEO = 500000

# pipes limits
SQL_PIPE_MAX = 4000
FILE_PIPE_MAX = 2000
WANDB_PIPE_MAX = 1000
META_PIPE_MAX = 1000

TAR_BYTES = 512 * 1024 * 1024 # 512MB

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

def validate_aspect_ratio(aspect_ratio):
    width, height = map(float, aspect_ratio.split(':'))
    if height > 1.2 * width or width > 2.0 * height:
        return False
    else:
        return True
    
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

def download_worker_func(index: int, sql_pipe: mp.Queue, file_pipe: mp.Queue):
    logger.info(f"downloader-{index}: started")
    while True:
        try:
            logger.info(f"downloader-{index}: waiting for data")
            data = sql_pipe.get()
            logger.info(f"downloader-{index}: got data")
            
            if data is None:
                logger.info(f"downloader-{index}: got None")
                sql_pipe.put(None)
                file_pipe.put(None)
                break

            video_url, metadata = data

            for i in range(DOWNLOAD_RETRY_COUNT):
                try:
                    response = requests.get(video_url, timeout=DOWNLOAD_TIMEOUT)
                    if response.status_code != 200:
                        logger.info(f"downloader-{index}: failed: status code: {response.status_code}")
                        continue
                    logger.info(f"downloader-{index}: downloaded")
                    break
                except Exception as e:
                    traceback.print_exc()
                    logger.info(f"downloader-{index}: failed: {e}")
                    continue
            else:
                logger.info(f"downloader-{index}: failed: max retries")
                continue
            
            file_pipe.put((response.content, metadata))
            logger.info(f"downloader-{index}: sent")
        except Exception as e:
            traceback.print_exc()
            logger.info(f"downloader-{index}: failed: {e}")
            continue

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

def tar_worker_func(index: int, file_pipe: mp.Queue, meta_pipe: mp.Queue, tar_id: mp.Value, wandb_pipe: mp.Queue = None):
    meta_tar = list()
    files_tar = list()
    files_size = 0
    logger.info(f"tar-{index}: started")
    while True:
        try:
            logger.info(f"tar-{index}: waiting for data")
            data = file_pipe.get()
            logger.info(f"tar-{index}: got data")

            if data is None:
                logger.info(f"tar-{index}: got None")
                file_pipe.put(None)
                meta_pipe.put(None)
                if USE_WANDB:
                    wandb_pipe.put(None)
                break

            video_bytes, metadata = data
            video_id = metadata["id"]

            metadata_as_bytes = json.dumps(metadata).encode("utf-8")

            logger.info(f"tar-{index}: added {video_id} to tar, size: {int(files_size/1024/1024)}/{int(TAR_BYTES/1024/1024)} MB")

            try:
                metadata['cv_height'], metadata['cv_width'], metadata['cv_fps'], metadata['cv_duration'], metadata['cv_frame_count'] = vid_extras(video_bytes)
            except Exception:
                logger.info(f"tar-{index}: failed to open video in cv, {video_id}")
                continue

            meta_tar.append(metadata)
            files_tar.append((video_id, video_bytes, metadata_as_bytes))
            files_size += len(video_bytes) + len(metadata_as_bytes)

            if USE_WANDB:
                wandb_pipe.put({"type": "video_entry", "data": (video_id, metadata['cv_fps'], metadata['cv_duration'], metadata['cv_frame_count'])})

            if files_size > TAR_BYTES:
                logger.info(f"tar-{index}: tar is full")
                with tar_id.get_lock():
                    tar_id.value += 1
                    tar_id_copy = tar_id.value
                    
                tar_id_copy = str(tar_id_copy).zfill(6)

                logger.info(f"tar-{index}: writing tar to disk")

                with tarfile.open(name=f"{DISK_PATH}/{tar_id_copy}.tar", mode="w") as tar:
                    for video_id, video_bytes, metadata_as_bytes in files_tar:
                        tarinfo = tarfile.TarInfo(name=f"{video_id}.mp4")
                        tarinfo.size = len(video_bytes)
                        tar.addfile(tarinfo, io.BytesIO(video_bytes))
                        tarinfo = tarfile.TarInfo(name=f"{video_id}.json")
                        tarinfo.size = len(metadata_as_bytes)
                        tar.addfile(tarinfo, io.BytesIO(metadata_as_bytes))

                files_tar = list()
                files_size = 0

                logger.info(f"tar-{index}: wrote tar to disk")

                if USE_WANDB:
                    wandb_pipe.put({"type": "tar_entry", "data": tar_id_copy})

                meta_pipe.put((tar_id_copy, meta_tar))

                logger.info(f"tar-{index}: created new tar with id {tar_id_copy}, and {len(meta_tar)} videos")

                meta_tar = list()
        except Exception as e:
            traceback.print_exc()
            logger.info(f"tar-{index}: failed: {e}")
            continue

def wandb_worker_func(wandb_pipe: mp.Queue):
    run = wandb.init(project=WANDB_PROJ, entity=WANDB_ENTITY)
    init_time = time.time()

    frames = 0
    hours = 0
    videos = 0
    tars = 0

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

            elif data["type"] == "tar_entry":
                tars += 1
                run.log({"tars": tars}, step = elapsed_time)

            # speed report
            if elapsed_time > 0:
                run.log({"fps": frames / elapsed_time, "vps": videos / elapsed_time, "tps": tars / elapsed_time}, step = elapsed_time)
        except Exception as e:
            traceback.print_exc()
            logger.info(f"wandb: failed: {e}")
            continue

    run.finish()

def sql_writer_func(meta_pipe: mp.Queue):
    conn = sqlite3.connect(SQL_WRITE_PATH)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS dataset_map (id TEXT PRIMARY KEY, tar_id INTEGER, cv_height INTEGER, cv_width INTEGER, cv_fps INTEGER, cv_duration INTEGER, cv_frame_count INTEGER, aspect_ratio TEXT, video_url TEXT, description TEXT, author TEXT, category TEXT, r18 INTEGER)")

    itteration = 0

    logger.info("sql_writer: started")

    while True:
        try:
            data = meta_pipe.get()

            if data is None:
                break

            tar_id, meta_batch = data

            for video_entry in meta_batch:
                item = video_entry
                cursor.execute("INSERT OR REPLACE INTO dataset_map (id, tar_id, cv_height, cv_width, cv_fps, cv_duration, cv_frame_count, aspect_ratio, video_url, description, author, category, r18) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                item['id'], 
                                tar_id, 
                                item['cv_height'], 
                                item['cv_width'], 
                                item['cv_fps'],
                                item['cv_duration'],
                                item['cv_frame_count'],
                                item['aspect_ratio'],
                                item['video_url'],
                                item['description'],
                                json.dumps(item['author']),
                                json.dumps(item['category']),
                                item['r18']
                            )
                            )
            itteration += 1
            logger.info(f"sql_writer: wrote {len(meta_batch)} entries")

            if itteration % SQL_WRITE_CHECKPOINT == 0:
                conn.commit()
                logger.info("sql_writer: checkpoint")
        except Exception as e:
            traceback.print_exc()
            logger.info(f"sql_writer: failed: {e}")
            continue

    conn.commit()
    conn.close()

def main():

    if OPERATING_MODE == "process":
        spawner = mp.Process
    elif OPERATING_MODE == "thread":
        spawner = Thread

    manager = mp.Manager()

    sql_pipe = manager.Queue(maxsize=SQL_PIPE_MAX)
    file_pipe = manager.Queue(maxsize=FILE_PIPE_MAX)
    meta_pipe = manager.Queue(maxsize=META_PIPE_MAX)
    if USE_WANDB:
        wandb_pipe = manager.Queue(maxsize=WANDB_PIPE_MAX)
    else:
        wandb_pipe = None
    tar_id = mp.Value("i", 0)

    logger.info("starting threads")

    sql_reader = spawner(target=sql_reader_func, args=(sql_pipe,))
    sql_reader.start()

    logger.info("started sql_reader")

    download_threads = [spawner(target=download_worker_func, args=(i, sql_pipe, file_pipe,)) for i in range(DOWNLOAD_WORKERS)]
    for thread in download_threads:
        thread.start()

    logger.info("started download_workers")

    tar_threads = [spawner(target=tar_worker_func, args=(i, file_pipe, meta_pipe, tar_id, wandb_pipe,)) for i in range(TAR_WORKERS)]
    for thread in tar_threads:
        thread.start()

    logger.info("started tar_workers")

    if USE_WANDB:
        wandb_thread = spawner(target=wandb_worker_func, args=(wandb_pipe,))
        wandb_thread.start()

    logger.info("started wandb_worker")

    sql_writer = spawner(target=sql_writer_func, args=(meta_pipe,))
    sql_writer.start()

    logger.info("started sql_writer")
    logger.info("main: ready and waiting till full join")

    sql_reader.join()
    for thread in download_threads:
        thread.join()
    for thread in tar_threads:
        thread.join()
    if USE_WANDB:
        wandb_thread.join()
    sql_writer.join()

    logger.info("main: finished")

if __name__ == "__main__":
    main()
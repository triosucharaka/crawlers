import requests
import multiprocessing as mp
import io
import json
import traceback
import wandb
import sqlite3
import tarfile
import cv2
import numpy as np
import time

# config

SQL_READ_PATH = "/home/windowsuser/crawlers/sites/shutterstock/database.db"
SQL_WRITE_PATH = "/home/windowsuser/crawlers/sites/shutterstock/dataset_map.db"
SQL_MAX = 12500000 # 12.5m
SQL_WRITE_CHECKPOINT = 100

DOWNLOADER_PROCESS_COUNT = 40
DOWNLOADER_TIMEOUT = 15
DOWNLOADER_RETRY_COUNT = 3

DISK_PATH = "/home/windowsuser/mount-folder/tempofunkds/shutterstock1/"

MAX_VIDEO = 500000

# pipes limits
SQL_PIPE_MAX = 1000

TAR_BYTES = 128 * 1024 * 1024 # 128MB

USE_WANDB = True

def validate_aspect_ratio(aspect_ratio):
    width, height = map(float, aspect_ratio.split(':'))
    if height > 1.2 * width or width > 2.0 * height:
        return False
    else:
        return True
    
def sql_reader(sql_pipe: mp.Queue):
    conn = sqlite3.connect(SQL_READ_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM shuffled_videos")

    row_count = 0
    while True:        
        row = cursor.fetchone()
        if row is None or row_count >= MAX_VIDEO:
            break

        v_id, desc, dur, asr, v_url, auth, catg, fps, r18 = row 

        if not validate_aspect_ratio(asr):
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

    sql_pipe.put(None)

def downloader_worker(sql_pipe: mp.Queue, file_pipe: mp.Queue):
    while True:
        data = sql_pipe.get()
        
        if data is None:
            sql_pipe.put(None)
            file_pipe.put(None)
            break

        video_url, metadata = data

        for i in range(DOWNLOADER_RETRY_COUNT):
            try:
                response = requests.get(video_url, timeout=DOWNLOADER_TIMEOUT)
                break
            except:
                traceback.print_exc()
                continue
        else:
            raise Exception("downloader failed")
        
        file_pipe.put((response.content, metadata))

def vid_extras(video_bytes: bytes):
    video_array = np.frombuffer(video_bytes, dtype=np.uint8)
    cap = cv2.VideoCapture()
    cap.open(video_array)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // fps
    cap.release()
    del cap, video_array
    return height, width, fps, duration, frame_count

def tar_worker(file_pipe: mp.Queue, meta_pipe: mp.Queue, tar_id: mp.Value, wandb_pipe: mp.Queue = None):
    mem_tar = io.BytesIO()
    meta_tar = list()
    while True:
        data = file_pipe.get()

        if data is None:
            file_pipe.put(None)
            meta_pipe.put(None)
            break

        video_bytes, metadata = data
        video_id = metadata["id"]

        metadata_as_bytes = json.dumps(metadata).encode("utf-8")

        with tarfile.open(fileobj=mem_tar, mode="w") as tar:
            tarinfo = tarfile.TarInfo(name=f"{video_id}.mp4")
            tarinfo.size = len(video_bytes)
            tar.addfile(tarinfo, io.BytesIO(video_bytes))
            
            tarinfo = tarfile.TarInfo(name=f"{video_id}.json")
            tarinfo.size = len(metadata_as_bytes)
            tar.addfile(tarinfo, io.BytesIO(metadata_as_bytes))

        metadata['cv_height'], metadata['cv_width'], metadata['cv_fps'], metadata['cv_duration'], metadata['cv_frame_count'] = vid_extras(video_bytes)
        meta_tar.append(metadata)

        if USE_WANDB:
            wandb_pipe.put({"type": "video_entry", "data": (video_id, metadata['cv_fps'], metadata['cv_duration'], metadata['cv_frame_count'])})
            # talvez crear tablas?

        if mem_tar.tell() > TAR_BYTES:
            with tar_id.get_lock():
                tar_id_copy = tar_id.value
                tar_id.value += 1
                
            tar_id_copy = str(tar_id_copy).zfill(6)

            mem_tar.seek(0)
            with open(f"{DISK_PATH}/{tar_id_copy}.tar", "wb") as f:
                f.write(mem_tar.read())

            if USE_WANDB:
                wandb_pipe.put({"type": "tar_entry", "data": tar_id_copy})

            meta_pipe.put((tar_id_copy, meta_tar))

            mem_tar = io.BytesIO()
            meta_tar = list()

def wandb_worker(wandb_pipe: mp.Queue):
    run = wandb.init(project="tempofunkds", entity="tempofunk")
    init_time = time.time()

    frames = 0
    hours = 0
    videos = 0
    tars = 0

    while True:
        elapsed_time = time.time() - init_time

        data = wandb_pipe.get()

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

    run.finish()

def sql_writer(meta_pipe: mp.Queue):
    conn = sqlite3.connect(SQL_WRITE_PATH)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS dataset_map (id TEXT PRIMARY KEY, tar_id INTEGER, cv_height INTEGER, cv_width INTEGER, cv_fps INTEGER, cv_duration INTEGER, cv_frame_count INTEGER, aspect_ratio TEXT, video_url TEXT, description TEXT, author TEXT, category TEXT, r18 INTEGER)")

    itteration = 0

    while True:
        data = meta_pipe.get()

        if data is None:
            meta_pipe.put(None)
            break

        tar_id, meta_batch = data

        for video_entry in meta_batch:
            item = video_entry
            cursor.execute("INSERT OR REPLACE INTO dataset_map (id, tar_id, cv_height, cv_width, cv_fps, cv_duration, cv_frame_count, aspect_ratio, video_url, description, author, category, r18) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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

        if itteration % SQL_WRITE_CHECKPOINT == 0:
            conn.commit()
            print(f"checkpoint: {itteration}")

    conn.commit()
    conn.close()
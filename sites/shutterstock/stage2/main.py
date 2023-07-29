import requests
import multiprocessing as mp
import io
import zipfile
import time
import json
import traceback
import re
import tqdm
import os
import random
import sqlite3

# config

SQL_PATH = "/home/windowsuser/crawlers/sites/shutterstock/stage2/database.db"
SQL_MAX = 12500000 # 12.5m
DOWNLOADER_PROCESS_COUNT = int(os.cpu_count() * 2)
DOWNLOADER_TIMEOUT = 15
DOWNLOADER_RETRY_COUNT = 3
DISK_PATH = "/mnt/disks/hhd/tango/videos"
MAX_VIDEO = 500000 # 500k
SQL_PIPE_MAX = 1000

MAX_BYTES = 1024 * 1024 * 1024 * 1024 # 1TB

def validate_aspect_ratio(aspect_ratio):
    width, height = map(float, aspect_ratio.split(':'))
    if height > 1.2 * width or width > 2.0 * height:
        return False
    else:
        return True

# downloader function

def downloader_worker(sql_pipe: mp.Queue, download_count: mp.Value, download_byte_count: mp.Value):
    global pb
    while True:
        try:
            if download_byte_count.value >= MAX_BYTES:
                break
            # get data from sql
            data = sql_pipe.get()
            if data is None:
                break
            # download
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
            # save
            video_id = metadata["id"]
            video_path = os.path.join(DISK_PATH, video_id + ".mp4")
            meta_path = os.path.join(DISK_PATH, video_id + ".json")
            with open(video_path, "wb") as f:
                f.write(response.content)
            with open(meta_path, "w") as f:
                json.dump(metadata, f)
            # update download count
            download_count.value += 1
            download_byte_count.value += len(response.content)
            download_byte_count.value += len(json.dumps(metadata))
        except:
            traceback.print_exc()

def sql_reader(sql_pipe: mp.Queue):
    conn = sqlite3.connect(SQL_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM videos")
    id_blacklist = set()
    row_count = 0
    while True:
        random_id = random.randint(0, SQL_MAX)
        if row_count >= MAX_VIDEO:
            break
        if random_id in id_blacklist:
            continue
        cursor.execute("SELECT * FROM videos WHERE id = ?", (random_id,))
        row = cursor.fetchone()
        if row is None:
            continue
        v_id, desc, dur, asr, v_url, auth, catg, fps, r18 = row
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
        id_blacklist.add(random_id)

def tqdm_worker(download_count: mp.Value):
    pb = tqdm.tqdm(total=SQL_MAX)
    while True:
        if download_count.value >= SQL_MAX:
            break
        latest_val = download_count.value
        pb.update(latest_val)
        download_count.value -= latest_val
        time.sleep(1)

def main():
    mp.set_start_method("spawn")
    # create download count
    download_count = mp.Value("i", 0)
    download_byte_count = mp.Value("i", 0)
    # create downloader processes
    downloader_processes = []
    downloader_sql_pipe = mp.Queue(maxsize=SQL_PIPE_MAX)
    # create tqdm process
    tqdm_process = mp.Process(target=tqdm_worker, args=(download_count,))
    tqdm_process.start()
    # create downloader processes
    for _ in range(DOWNLOADER_PROCESS_COUNT):
        downloader_process = mp.Process(target=downloader_worker, args=(downloader_sql_pipe, download_count, download_byte_count))
        downloader_process.start()
        downloader_processes.append(downloader_process)
    # create sql reader process
    sql_reader_process = mp.Process(target=sql_reader, args=(downloader_sql_pipe,))
    sql_reader_process.start()
    # wait for sql reader process
    sql_reader_process.join()
    # send stop signal to downloader processes
    for _ in range(DOWNLOADER_PROCESS_COUNT):
        downloader_sql_pipe.put(None)
    # wait for downloader processes
    for downloader_process in downloader_processes:
        downloader_process.join()

if __name__ == "__main__":
    main()
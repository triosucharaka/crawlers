import logging
import os
import time
import json
import requests
from threading import Thread
import multiprocessing as mp
import tarfile
from fake_headers import Headers
from wrapt_timeout_decorator import *
import argparse
import time
from huggingface_hub import HfApi
import mimetypes
import sqlite3
from io import BytesIO
import tqdm
import gc

def generate_headers():
    generated_headers = Headers(None, None, False).generate()
    generated_headers['Accept'] = 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8'
    return generated_headers

def sql_gather(_output_queue, scrap_config):
    sql_file = scrap_config['sql_file']
    download_batch_size = scrap_config['download_batch_size']

    conn = sqlite3.connect(sql_file)
    c = conn.cursor()

    current_index = 0

    while True:
        time.sleep(0.5)
        if _output_queue.qsize() < download_batch_size:
            # grab {current_index} entries
            c.execute(f'SELECT * FROM music LIMIT {current_index}, {download_batch_size}')
            current_index = current_index + (download_batch_size)
            for i in c.fetchall():
                track_dict = {
                    'id': int(i[0]),
                    'title': str(i[1]),
                    'added': int(i[2]),
                    'creatives': json.loads(i[3]),
                    'length': int(i[4]),
                    'bpm': int(i[5]),
                    'isSfx': bool(i[6]),
                    'hasVocals': bool(i[7]),
                    'hidden': bool(i[8]),
                    'publicSlug': str(i[9]),
                    'genres': json.loads(i[10]),
                    'moods': json.loads(i[11]),
                    'energyLevel': str(i[12]),
                    'stems': json.loads(i[13]),
                    'metadataTags': json.loads(i[14]),
                    'isExplicit': bool(i[15]),
                    'isCommercialRelease': bool(i[16]),
                    'imageUrl': str(i[17]),
                    'coverArt': json.loads(i[18])
                }

                if i[19] == "None":
                    track_dict['releaseDate'] = None
                else:
                    track_dict['releaseDate'] = i[19]

                if track_dict['hasVocals']:
                    _tmp = json.loads(i[20])
                    _tmp['content'].replace('\u200c', '')
                    track_dict['lyrics'] = _tmp
                else:
                    track_dict['lyrics'] = None
                
                _output_queue.put(track_dict)

def bpm_to_level(bpm: int):
    if bpm < 80:
        return "slow"
    elif bpm < 120:
        return "medium"
    elif bpm < 150:
        return "fast"
    else:
        return "very fast"

def packaging_thread(input_queue: mp.Queue, scrap_config):
    """
    input_queue:
    scrap_config: dict

    input_queue is the queue where the downloader threads place dicts on.
    scrap_config is the config dict passed to the scrap function.
    """
    chunk_id = 0
    tar_pad = scrap_config['tar_filename_zero_padding']
    api = HfApi()
    save_bytes_size = scrap_config['save_megabytes_size'] * 1024 * 1024
    hf_repo = scrap_config['hf_repo']
    hf_branch = scrap_config['hf_branch']

    tar_bytes = BytesIO()
    tar = tarfile.open(fileobj=tar_bytes, mode="w")
    uploaded_chunks = []

    _cc = 0

    while True:
        _cc = _cc + 1
        if _cc >= 10:
            print("Packager: " + str(int(tar_bytes.getbuffer().nbytes / 1024 / 1024)) + "/" + str(int(save_bytes_size / 1024 / 1024)) + "MB")
            _cc = 0

        # wait for input
        _input = input_queue.get()
        if _input == "STOP":
            break

        entry_id = _input['metadata']['id']

        # package
        ## metadata
        fileinfo = tarfile.TarInfo(name=f"{entry_id}.json")
        fileinfo.size = len(json.dumps(_input['metadata']).encode('utf-8'))
        tar.addfile(fileinfo, BytesIO(json.dumps(_input['metadata']).encode('utf-8')))
        ## audio
        fileinfo = tarfile.TarInfo(name=f"{entry_id}{_input['extension']}")
        fileinfo.size = _input['audiofull'].getbuffer().nbytes
        tar.addfile(fileinfo, _input['audiofull'])
        ## caption
        fileinfo = tarfile.TarInfo(name=f"{entry_id}.txt")
        fileinfo.size = len(_input['caption'].encode('utf-8'))
        tar.addfile(fileinfo, BytesIO(_input['caption'].encode('utf-8')))

        # upload
        if tar_bytes.getbuffer().nbytes > save_bytes_size:
            tar.close()
            tar_bytes.seek(0)
            api.upload_file(
                repo_id=hf_repo,
                repo_type="dataset",
                path_or_fileobj=tar_bytes,
                path_in_repo=f"data/{str(chunk_id).zfill(tar_pad)}.tar",
                revision=hf_branch,
                commit_message=f"Chunk {chunk_id}"
            )
            tar_bytes = BytesIO()
            tar = tarfile.open(fileobj=tar_bytes, mode="w")
            uploaded_chunks.append(str(chunk_id).zfill(tar_pad))

            api.upload_file(
                repo_id=hf_repo,
                repo_type="dataset",
                path_or_fileobj=BytesIO(json.dumps(uploaded_chunks).encode('utf-8')),
                path_in_repo=f"chunks.json",
                revision=hf_branch,
                commit_message=f"Chunklist update"
            )

            print(f"HuggingFace: Uploaded chunk {chunk_id}")
            chunk_id = chunk_id + 1
        time.sleep(0.1)

def download_call(data_dict: dict, proxies):
    _headers = generate_headers()
    _output = {
        "audiofull": None,
        "metadata": data_dict,
        "caption": "CAPTIONGOESHERE",
        "extension": None,
    }

    # download full only
    url = data_dict['stems']['full']['lqMp3Url']
    r = requests.get(url, stream=True, proxies=proxies, headers=_headers)
    _output['extension'] = mimetypes.guess_extension(r.headers['content-type'])
    _output['audiofull'] = BytesIO(r.content)

    # parse caption from data_dict
    _tojoin = []
    _tojoin.append(data_dict['title'])
    
    if data_dict['creatives']['mainArtists'] not in [None, []]:
        _artists = []
        for artist in data_dict['creatives']['mainArtists']:
            _artists.append(artist['name'])
        _tojoin.append("by " + ", ".join(_artists))
    
    if data_dict['metadataTags'] not in [None, []]:
        for tag in data_dict['metadataTags']:
            _tojoin.append(tag)

    if data_dict['genres'] not in [None, []]:
        for genre in data_dict['genres']:
            _tojoin.append(genre['displayTag'])

    if data_dict['isSfx'] is False:
        if data_dict['moods'] not in [None, []]:
            for mood in data_dict['moods']:
                _tojoin.append(mood['displayTag'])

        _tojoin.append(f"{data_dict['energyLevel']} energy level")

        if data_dict['bpm'] not in [None, 0]:
            _tojoin.append(f"{bpm_to_level(data_dict['bpm'])} bpm")
    else:
        _tojoin.append("sfx")
        _tojoin.append("Sound Effects")

    if data_dict['isExplicit']:
        _tojoin.append("explicit")

    # join everything and separate with commas
    _output['caption'] = ", ".join(_tojoin)

    return _output

def sql_get_total(scrap_config):
    db = sqlite3.connect(scrap_config['sql_file'])
    cursor = db.cursor()
    cursor.execute("SELECT COUNT(*) FROM music")
    return cursor.fetchone()[0]

def main():
    logging.basicConfig(level=logging.WARNING, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler('logfile.log', 'a'))

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--proxyconf', type=str, default=None, required=False, help='Path to proxy config file')
    args = parser.parse_args()

    scrap_config = json.load(open(args.config))

    proxies = None
    if args.proxyconf is not None:
        proxy_config = json.load(open(args.proxyconf))
        # webshare is recommended
        proxies = {
            "http": f"http://{proxy_config['username']}:{proxy_config['password']}@{proxy_config['proxy_url']}",
            "https": f"http://{proxy_config['username']}:{proxy_config['password']}@{proxy_config['proxy_url']}"
        }

    download_batch_size = scrap_config['download_batch_size']
    delay = scrap_config['delay']
    max_retries = scrap_config['max_retries']
    timeout_len = scrap_config['timeout_len']
    max_packaging_size = scrap_config['max_packaging_size']

    manager = mp.Manager()

    _data_queue = manager.Queue()
    _packaging_queue = manager.Queue()

    sql_thread = Thread(target=sql_gather, args=(_data_queue, scrap_config,))
    sql_thread.start()

    pkg_thread = Thread(target=packaging_thread, args=(_packaging_queue, scrap_config,))
    pkg_thread.start()

    @timeout(timeout_len)
    def download_track(trackmeta: dict, packaging_queue):
        try:
            track_dict = download_call(trackmeta, proxies)
            packaging_queue.put(track_dict)
        except Exception as e:
            logger.warning(f"Error downloading track {trackmeta['id']}: {e}")

    def download_track_timeout(trackmeta: dict, packaging_queue, retry_n=0):
        try:
            download_track(trackmeta, packaging_queue)
        except Exception as e:
            if retry_n < max_retries:
                download_track_timeout(trackmeta, packaging_queue, retry_n=retry_n+1)
            else:
                logger.warning(f"Error downloading track {trackmeta['id']}: {e}")

    download_threads = []

    pb = tqdm.tqdm(total=sql_get_total(scrap_config))

    while True:
        download_threads = [t for t in download_threads if t.is_alive()]
        cur_batch = download_batch_size - len(download_threads)

        if _packaging_queue.qsize() > max_packaging_size:
            while not (_packaging_queue.qsize() < max_packaging_size):
                print("Downloader: Packaging Queue full, waiting for it to be lower than " + str(max_packaging_size) + "...")
                time.sleep(1)

        # grab {cur_batch} items from _data_queue
        for _ in range(0, cur_batch):
            if _data_queue.empty():
                break
            trackmeta = _data_queue.get()
            t = Thread(target=download_track_timeout, args=(trackmeta, _packaging_queue,))
            download_threads.append(t)
            t.start()
        
        for thread in download_threads:
            if not thread.is_alive():
                try:
                    thread.join()
                except Exception as e:
                    logger.warning(f"Error joining thread: {e}")

        pb.update(cur_batch)
        time.sleep(delay)

if __name__ == '__main__':
    main()
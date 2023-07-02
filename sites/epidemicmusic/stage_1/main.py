import logging
import os
import time
import json
import requests
from threading import Thread
import multiprocessing as mp
import queue
import tqdm
from fake_headers import Headers
from wrapt_timeout_decorator import *
import argparse
import time
import datetime
import sqlite3

def sql_thread(_queue: queue.Queue, configuration: dict):
    sqldb_path = configuration['sqldb_path']
    save_batch_size = configuration['save_batch_size']

    conn = sqlite3.connect(sqldb_path)
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS music
             (id INT PRIMARY KEY,
             title TEXT,
             added INT,
             creatives TEXT,
             length INT,
             bpm INT,
             isSfx BOOL,
             hasVocals BOOL,
             hidden BOOL,
             publicSlug TEXT,
             genres TEXT,
             moods TEXT,
             energyLevel TEXT,
             stems TEXT,
             metadataTags TEXT,
             isExplicit BOOL,
             isCommercialRelease BOOL,
             imageUrl TEXT,
             coverArt TEXT,
             releaseDate TEXT,
             lyrics TEXT
             )''')
    
    _cc = 0

    while True:
        time.sleep(0.5)
        _cc = _cc + 1
        if _cc >= 10:
            print(f'SQL: {_queue.qsize()} / {save_batch_size}')
            _cc = 0
        if _queue.qsize() > save_batch_size:
            print(f'SQL: saving...')
            for i in range(save_batch_size):
                track = _queue.get()
                c.execute("INSERT OR REPLACE INTO music VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        track['id'],
                        track['title'],
                        track['added'],
                        track['creatives'],
                        track['length'],
                        track['bpm'],
                        track['isSfx'],
                        track['hasVocals'],
                        track['hidden'],
                        track['publicSlug'],
                        track['genres'],
                        track['moods'],
                        track['energyLevel'],
                        track['stems'],
                        track['metadataTags'],
                        track['isExplicit'],
                        track['isCommercialRelease'],
                        track['imageUrl'],
                        track['coverArt'],
                        track['releaseDate'],
                        track['lyrics'] if 'lyrics' in track else None
                    )
                )
            conn.commit()

class Scrapper:
    def __init__(self, proxies, base_url, headers_function):
        self.proxies = proxies
        self.base_url = base_url
        self.headers_function = headers_function

    class ScrapperError(Exception):
        pass

    def get_page(self, page):
        _url = self.base_url + str(page)
        response = requests.get(_url, proxies=self.proxies, headers=self.headers_function())
        if response.status_code != 200:
            raise self.ScrapperError(
                f'Server returned response code {response.status_code} for page {page}! URL: {_url} | Content: {response.content}')
        metadata = response.json()
        return metadata

def generate_headers():
    generated_headers = Headers(None, None, False).generate()
    generated_headers['Accept'] = 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8'
    return generated_headers

def parse_metadata(metadata):
    return {
        'id': int(metadata['id']),
        'title': str(metadata['title']),
        'added': int(datetime.datetime.fromisoformat(metadata['added']).timestamp()),
        'creatives': json.dumps(metadata['creatives']),
        'length': int(metadata['length']),
        'bpm': int(metadata['bpm']),
        'isSfx': bool(metadata['isSfx']),
        'hasVocals': bool(metadata['hasVocals']),
        'hidden': bool(metadata['hidden']),
        'publicSlug': str(metadata['publicSlug']),
        'genres': json.dumps(metadata['genres']),
        'moods': json.dumps(metadata['moods']),
        'energyLevel': str(metadata['energyLevel']),
        'stems': json.dumps(metadata['stems']),
        'metadataTags': json.dumps(metadata['metadataTags']),
        'isExplicit': bool(metadata['isExplicit']),
        'isCommercialRelease': bool(metadata['isCommercialRelease']),
        'imageUrl': json.dumps(metadata['imageUrl']),
        'coverArt': json.dumps(metadata['coverArt']),
        'releaseDate': str(metadata['releaseDate']),
    }

def lyrics_url(id: int):
    return f"https://www.epidemicsound.com/json/track/{id}/lyrics/"

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

    target_url = "https://www.epidemicsound.com/json/tracks/"

    cur_page = scrap_config['start_page']
    stop_page = scrap_config['stop_page']
    
    download_batch_size = scrap_config['download_batch_size']
    delay = scrap_config['delay']
    max_retries = scrap_config['max_retries']
    timeout_len = scrap_config['timeout_len']

    scrapper = Scrapper(proxies, target_url, generate_headers)

    manager = mp.Manager()

    def scrape_page_timeout(page, _queue, retry_n=0):
        try:
            scrape_post(page, _queue)
        except Exception as e:
            if retry_n < max_retries:
                scrape_page_timeout(page, _queue, retry_n=retry_n + 1)
            else:
                logger.warning(f'Page {page} failed after {max_retries} tries: {e}')

    @timeout(timeout_len)
    def scrape_post(page, _queue: queue.Queue):
        try:
            meta = scrapper.get_page(page)
            _track_metadata = parse_metadata(meta)

            if _track_metadata['hasVocals'] is True:
                _track_metadata['lyrics'] = json.dumps(requests.get(lyrics_url(_track_metadata['id']), proxies=proxies, headers=generate_headers()).json())

            _queue.put(_track_metadata)
        except Exception as e:
            logger.warning(e)
    
    thread_list = []
    _queue = manager.Queue()

    pb = tqdm.tqdm(total=stop_page - cur_page)

    sql_thread_process = Thread(target=sql_thread, args=(_queue, scrap_config,))
    sql_thread_process.start()

    while True:
        if cur_page > stop_page:
            break

        thread_list = [t for t in thread_list if t.is_alive()]
        cur_batch = download_batch_size - len(thread_list)
        print(cur_batch, 'threads available')

        for i in range(0, cur_batch):
            thread_list.append(Thread(target=scrape_page_timeout, args=(cur_page + i, _queue,)))
        for thread in thread_list:
            if not thread.is_alive():
                try:
                    thread.start()
                except Exception as e:
                    print(e)
        cur_page += cur_batch
        pb.update(cur_batch)
        time.sleep(delay)

if __name__ == '__main__':
    main()
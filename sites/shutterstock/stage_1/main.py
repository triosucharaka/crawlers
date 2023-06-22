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
import sqlite3

def sql_thread(_queue: queue.Queue, configuration: dict):
    sqldb_path = configuration['sqldb_path']
    save_batch_size = configuration['save_batch_size'] * 100 #page size is 100

    conn = sqlite3.connect(sqldb_path)
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS videos
             (id TEXT PRIMARY KEY, 
             description TEXT, 
             duration TEXT, 
             aspectratio TEXT, 
             videourl TEXT, 
             author TEXT, 
             categories TEXT,
             framerate TEXT,
             r18 TEXT)''')
    
    while True:
        time.sleep(0.5)
        print(f'SQL: queue size: {_queue.qsize()}')
        if _queue.qsize() > save_batch_size:
            print(f'SQL: saving...')
            for i in range(save_batch_size):
                video = _queue.get()
                c.execute("INSERT OR REPLACE INTO videos VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        video['id'],
                        video['description'],
                        video['duration'],
                        video['aspectratio'],
                        video['videourl'],
                        video['author'],
                        video['categories'],
                        video['framerate'],
                        video['r18']
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
        'id': metadata['id'],
        'description': metadata['description'],
        'duration': str(metadata['duration']).replace('"', '').replace("'", ""),
        'aspectratio': metadata['aspectRatioCommon'],
        'videourl': metadata['previewVideoUrls']['mp4'],
        'author': json.dumps(metadata['contributor']['publicInformation']),
        'categories': json.dumps([category['name'] for category in metadata['categories']]),
        'framerate': metadata['sizes']['lowresMpeg']['fps'],
        'r18': metadata['rRated']
    }

def main():

    logging.basicConfig(level=logging.WARNING, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler('logfile.log', 'a'))

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--proxyconf', type=str, default=None, required=False, help='Path to proxy config file')
    args = parser.parse_args()

    scrap_config = json.load(open(args.config))

    generated_headers = generate_headers()

    proxies = None
    if args.proxyconf is not None:
        proxy_config = json.load(open(args.proxyconf))
        # webshare is recommended
        proxies = {
            "http": f"http://{proxy_config['username']}:{proxy_config['password']}@{proxy_config['proxy_url']}",
            "https": f"http://{proxy_config['username']}:{proxy_config['password']}@{proxy_config['proxy_url']}"
        }

    target_url = "https://www.shutterstock.com/_next/data/BOuMf3Nfx28VkK5mI371H/en/_shutterstock/video/search.json?sort=newest&page="

    initial_request = requests.get(target_url + "1", proxies=proxies, headers=generated_headers)

    if initial_request.status_code != 200:
        raise Exception(f'Server returned response code {initial_request.status_code} for page 1! URL: {target_url + "1"} | Content: {initial_request.content}')

    total_pages = int(int(initial_request.json()['pageProps']['meta']['pagination']['total'])/100)

    cur_page = scrap_config['start_page']
    stop_page = scrap_config['stop_page'] if scrap_config['stop_page'] is not None else total_pages
    
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
            _video_list = [parse_metadata(video) for video in meta['pageProps']['videos']]
            for video in _video_list:
                _queue.put(video)
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
import logging
import os
import time
import shutil
import json
import requests
from threading import Thread
import tqdm
from fake_headers import Headers
import danlib
from wrapt_timeout_decorator import *
_headers=Headers(None, None, False).generate()
_headers['Accept'] = 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8'

proxy_config = json.load(open('config.json'))

p_username = proxy_config['username']
p_password = proxy_config['password']

proxies = {
    "http": f"http://{p_username}:{p_password}@p.webshare.io:80/",
    "https": f"http://{p_username}:{p_password}@p.webshare.io:80/"
}

target_url = f"https://www.shutterstock.com/_next/data/BOuMf3Nfx28VkK5mI371H/en/_shutterstock/video/search.json?sort=newest&page="
_request = requests.get(target_url + "1", proxies=proxies, headers=_headers)
print(_headers)
print(_request.status_code)
print(_request.content)
total_pages = int(int(_request.json()['pageProps']['meta']['pagination']['total'])/100)-10

START_PAGE = 1
STOP_PAGE = total_pages
BATCH_SIZE = 250
DELAY = 30
MAX_RETRIES = 3
TIMEOUT_LEN = 10
COUNTER_LOGGING_STEPS = 100
OUTPUT_PATH = 'output'

#function to append the page number into the failed file
def save_failed_page(page):
    with open('f/home/dep/tempofunk-scrapper/mapper/failed.txt', 'a') as f:
        f.write(str(page) + '\n')

logging.basicConfig(level=logging.WARNING, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler('logfile.log', 'a'))

dlib = danlib.Scrapper(None, target_url)

def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    pb = tqdm.tqdm(total=STOP_PAGE - START_PAGE)
    cur_page = START_PAGE
    thread_list = []
    while True:
        if cur_page >= STOP_PAGE:
            logger.info('reached end')
            break

        thread_list = [t for t in thread_list if t.is_alive()]
        cur_batch = BATCH_SIZE - len(thread_list)
        print(cur_batch, 'THREADS AVAILABLE')
        for i in range(0, cur_batch):
            thread_list.append(Thread(target=scrape_page_timeout, args=(cur_page + i,)))
        for thread in thread_list:
            if not thread.is_alive():
                try:
                    thread.start()
                except Exception as e:
                    print(e)
        cur_page += cur_batch
        pb.update(cur_batch)
        time.sleep(DELAY)

def scrape_page_timeout(page, retry_n=0):
    try:
        scrape_post(page)
    except Exception as e:
        if retry_n < MAX_RETRIES:
            scrape_page_timeout(page, retry_n=retry_n + 1)
        else:
            logger.warning(f'Page {page} failed after {MAX_RETRIES} tries')
            save_failed_page(page)


@timeout(TIMEOUT_LEN)
def scrape_post(page):
    try:
        base_path = OUTPUT_PATH + '/' + str(page)
        meta = dlib.get_page(page)
        open(base_path + '.json', 'w').write(json.dumps(meta))
        #logger.info(f'Downloaded {page}!')
    except Exception as e:
        logger.warning(e)

main()
import logging
import os
import time
import shutil
import json
from threading import Thread
import httphandler
from wrapt_timeout_decorator import *
from diffusers import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
from io import BytesIO
import cv2
from PIL import Image, ImageOps
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
import torch
import traceback
import torch.multiprocessing as mp
import tqdm
from einops import rearrange
import numpy as np
import datasets
from huggingface_hub import HfApi
import tarfile
mp.set_start_method('spawn', force=True)
mp.freeze_support()

BATCH_SIZE = 3
OUTPUT_DIR = "shutter"
os.makedirs(OUTPUT_DIR, exist_ok=True)
JSON_PATH = "video_list.json"
MODEL = "runwayml/stable-diffusion-v1-5"
DELAY = 5
MAX_RETRIES = 2
TIMEOUT_LEN = 30
GPUS = BATCH_SIZE
CHUNK_SIZE = 50 # 1GB in MB
HF_DATASET_PATH = "chavinlo/tempofunk"

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler('logfile.log', 'a'))

model_list = []

dlib = httphandler.HTTPHandler()

video_list = json.load(open(JSON_PATH, "r"))

tokenizer = CLIPTokenizer.from_pretrained(MODEL, subfolder="tokenizer")

def main():
    api = HfApi()
    manager = mp.Manager()

    for i in range(GPUS):
        print("Loading model on GPU", i, "...")
        vae = AutoencoderKL.from_pretrained(MODEL, subfolder='vae').to(f'cuda:{i}')
        text_encoder = CLIPTextModel.from_pretrained(MODEL, subfolder='text_encoder').to(f'cuda:{i}')
        model_list.append({
            "gpu_id": i,
            "vae_obj": vae,
            "enc_obj": text_encoder,
            "busy": False
        })
    
    thread_list = []
    list_index = 0

    chunk_count = 0
    chunk_container = manager.list()
    
    while True:
        thread_list = [t for t in thread_list if t.is_alive()]
        cur_batch = BATCH_SIZE - len(thread_list)
        for i in range(0, cur_batch):
            thread_list.append(mp.Process(target=scrape_post_timeout, args=(video_list[list_index + i],model_list,chunk_container,)))
        for thread in thread_list:
            if not thread.is_alive():
                try:
                    thread.start()
                except Exception as e:
                    print(e)
        list_index += cur_batch
        time.sleep(DELAY)

        total_bytes = 0
        for numpy_entry in chunk_container:
            total_bytes += len(numpy_entry['bytes'].getvalue())

        print("Container size:", total_bytes)
        print(len(chunk_container))

        #wait until all threads are done (wether they failed or not)
        for thread in thread_list:
            thread.join()

        if total_bytes > CHUNK_SIZE*1024*1024:
            tar_bytes = BytesIO()
            tar = tarfile.open(fileobj=tar_bytes, mode='w')

            for numpy_entry in chunk_container:
                tarinfo = tarfile.TarInfo(name=f"{numpy_entry['id']}.npy")
                tarinfo.size = len(numpy_entry['bytes'].getvalue())
                tar.addfile(tarinfo, fileobj=numpy_entry['bytes'])

            tar.close()
            tar_bytes.seek(0)

            api.upload_file(
                repo_id=HF_DATASET_PATH,
                repo_type="dataset",
                path_or_fileobj=tar_bytes,
                path_in_repo=f"data/{str(chunk_count).zfill(5)}.tar"
            )

            chunk_container[:] = []
            chunk_count += 1

def pil_to_torch(image, device = 'cpu'):
    return (2 * (pil_to_tensor(image).to(dtype=torch.float32, device=device)) / 255) - 1

def scrape_post_timeout(videometa, model_list, chunk_conainer, retry_n=0):
    try:
        scrape_post(videometa, model_list, chunk_conainer)
    except Exception as e:
        traceback.print_exc()
        if retry_n < MAX_RETRIES:
            scrape_post_timeout(videometa, model_list, chunk_conainer, retry_n=retry_n + 1)
        else:
            id = videometa['id']
            logger.info(f'Video with ID {id} failed after {MAX_RETRIES} tries')

@timeout(TIMEOUT_LEN)
def get_post_wrapper(videometa):
    return dlib.get_post(videometa)

def scrape_post(videometa, model_list, chunk_conainer: list):
    # Grab model from model list

    _cur_model = None
    _video_vae_frames = []
    while _cur_model is None:
        for model in model_list:
            if not model['busy']:
                model['busy'] = True
                _cur_model = model
                break
        time.sleep(0.5)

    _cur_gpu = _cur_model['gpu_id']
    _cur_vaemodel = _cur_model['vae_obj']

    try:
        id = videometa['id']
        stream, ext = get_post_wrapper(videometa)
        _output_path = "/tmp/" + id + "." + ext
        shutil.copyfileobj(stream, open(_output_path, 'wb'))
        cap = cv2.VideoCapture(_output_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        success, image = cap.read()
        count = 0
        #img loop
        _tqdm_bar = tqdm.tqdm(total=total_frames, position=_cur_gpu, leave=False)
        while success:
            try:
                success, image = cap.read()
                # image is a numpy array
                im_pil = Image.fromarray(image, mode='RGB')
                im_pil = ImageOps.fit(im_pil, (512, 512), centering = (0.5, 0.5))
                with torch.inference_mode():
                    m = pil_to_torch(im_pil, f'cuda:{_cur_gpu}').unsqueeze(0)
                    m = _cur_vaemodel.encode(m).latent_dist
                    _video_vae_frames.append({ 'mean': m.mean.squeeze().cpu().numpy(), 'std': m.std.squeeze().cpu().numpy() })
                _tqdm_bar.update(1)
                count += 1
            except AttributeError as e:
                pass
        cap.release()
        os.remove(_output_path)
        _tqdm_bar.close()

        _tokenized_prompt = tokenizer(
                [ videometa['description'] ],
                return_tensors="np",
                truncation=True,
                return_overflowing_tokens=True,
                padding="max_length",
            )
        logger.info(f'Downloaded {id}!')

        numpy_dict = {
            "frames": _video_vae_frames,
            "prompt": _tokenized_prompt
        }
        numpy_bytes = BytesIO()
        np.save(numpy_bytes, numpy_dict)
        numpy_bytes.seek(0)
        chunk_conainer.append({
            "id": id,
            "bytes": numpy_bytes
        })
        # Uncomment this below to save the video embed into a file to check wether it encoded correctly or not (use decoder.py to decode it)
        # blocklist = []

        # for frame in _video_vae_frames:
        #     tensor = frame['mean']
        #     blocklist.append(tensor)
        # video_embed = torch.stack(blocklist)
        # video_embed = rearrange(video_embed, 'f c h w -> c f h w')
        
        # torch.save(video_embed, f"video.pt")
        # print("Finished saving")
    except Exception as e:
        logger.info(e)
        traceback.print_exc()
        _cur_model['busy'] = False
        raise Exception("General Error")

if __name__ == "__main__":
    main()
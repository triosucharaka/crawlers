import requests
import huggingface_hub
import multiprocessing as mp
import io
import zipfile
import time
import json
import traceback
import re
import tqdm

ORG_NAME = "sail-rvc"
BOT_NAME = "juuxnscrap"
REPO_TYPE = "model"
MAX_PROS_QUEUE = 150

DOWNLOADER_PROCS = 50
SAVER_PROCS = 50

def convert_repo_id(input_str):
    # Check if the string contains non-ascii (like Chinese/Japanese/Korean) characters
    if not all(ord(c) < 128 for c in input_str):
        return False

    # Replace forbidden patterns with empty string
    repo_id = re.sub(r'--|\.\.', '', input_str)

    # Replace any non-alphanumeric, non-hyphen, non-dot character with underscore
    repo_id = re.sub(r'[^A-Za-z0-9\-\.]', '_', repo_id)
    
    # if string starts or ends with '-' or '.', replace it with underscore
    repo_id = re.sub(r'^[\-\.]|[\-\.]$', '_', repo_id)

    # finally, trim the string to limit its length to 96
    return repo_id[:96]

def generate_readme(model_name: str):
    readme = f"""
---
pipeline_tag: audio-to-audio
tags:
- rvc
- sail-rvc
---
# {model_name}

## RVC Model

![banner](https://i.imgur.com/xocCjhH.jpg)

This model repo was automatically generated.

Date: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}

Bot Name: {BOT_NAME}

Model Type: RVC

Source: https://huggingface.co/juuxn/RVCModels/

Reason: Converting into loadable format for https://github.com/chavinlo/rvc-runpod

"""
    return readme
 

def down_proc(down_queue: mp.Queue, zip_queue: mp.Queue):
    while True:
        rfilename = down_queue.get()
        if rfilename is None:
            for i in range(SAVER_PROCS):
                zip_queue.put((None, None))
            break
        url = f"https://huggingface.co/juuxn/RVCModels/resolve/main/{rfilename}"

        model_name = rfilename.split(".")[0]

        print(f"Downloading {model_name}...")

        zip_queue.put((
            requests.get(url).content,
            model_name  
        ))

        print(f"Downloaded {model_name}.")

def save_proc(zip_queue: mp.Queue, processed_models: mp.Value):

    hf_api = huggingface_hub.HfApi()

    while True:
        try:
            zip_content, model_name = zip_queue.get()
            if zip_content is None and model_name is None:
                break

            print(f"Saving {model_name}...")
            zip_io = io.BytesIO(zip_content)

            has_index = False
            has_pth = False

            index_path = None
            pth_path = None

            index_bytes = None
            pth_bytes = None

            with zipfile.ZipFile(zip_io, 'r') as zip_file:
                print("Zipfile opened.")
                for path in sorted(zip_file.namelist()):
                    print("Path:", path)
                    if path.endswith('.index'):
                        has_index = True
                        index_path = path
                    elif path.endswith('.pth'):
                        has_pth = True
                        pth_path = path
                
                if has_index is False or has_pth is False:
                    continue

                index_bytes = zip_file.read(index_path)
                pth_bytes = zip_file.read(pth_path)

            if index_bytes is None or pth_bytes is None:
                continue

            model_config = {
                "arch_type": "rvc",
                "arch_version": "2",
                "components": {
                    "pth": "model.pth",
                    "index": "model.index",
                },
                "metadata": "metadata.json"
            }

            model_metadata = {
                "_source": BOT_NAME,
                "author": "juuxn",
                "date": int(time.time()),
                "description": model_name,
                "image": None,
                "sample": None
            }

            model_name = convert_repo_id(model_name)

            repo_id = f"{ORG_NAME}/{model_name}"

            if repo_id is False:
                print(f"Invalid repo_id: {repo_id}")
                continue

            print(f"Creating repository {repo_id}...")

            hf_api.create_repo(
                repo_id=repo_id,
                repo_type="model"
            )

            print(f"Created repository {repo_id}.")

            hf_api.upload_file(
                repo_id=repo_id,
                repo_type=REPO_TYPE,
                path_in_repo="model.index",
                path_or_fileobj=index_bytes
            )
            
            hf_api.upload_file(
                repo_id=repo_id,
                repo_type=REPO_TYPE,
                path_in_repo="model.pth",
                path_or_fileobj=pth_bytes
            )

            hf_api.upload_file(
                repo_id=repo_id,
                repo_type=REPO_TYPE,
                path_in_repo="config.json",
                path_or_fileobj=json.dumps(model_config).encode()
            )

            hf_api.upload_file(
                repo_id=repo_id,
                repo_type=REPO_TYPE,
                path_in_repo="metadata.json",
                path_or_fileobj=json.dumps(model_metadata).encode()
            )

            hf_api.upload_file(
                repo_id=repo_id,
                repo_type=REPO_TYPE,
                path_in_repo="README.md",
                path_or_fileobj=generate_readme(model_name).encode()
            )

            print(f"Uploaded {model_name}")

            processed_models.value += 1

        except Exception as e:
            print(e)
            traceback.print_exc()
            continue

def tracker(processed_models: mp.Value, total_models: int):
    pb = tqdm.tqdm(total=total_models)
    while True:
        time.sleep(0.1)
        if processed_models.value < 1:
            continue
        pb.update(processed_models.value)
        processed_models.value = 0


def main():
    print("starting...")
    init_map = requests.get("https://huggingface.co/api/models/juuxn/RVCModels").json()
    zip_map = init_map['siblings']

    manager = mp.Manager()
    down_queue = manager.Queue()
    zip_queue = manager.Queue(maxsize=MAX_PROS_QUEUE)
    processed_models = manager.Value('i', 0)

    print("stage 1")

    total_models = 0

    for i in zip_map:
        path = i["rfilename"]
        if len(path) < 5 or path.split(".")[-1] != "zip":
            continue
        down_queue.put(path)
        total_models += 1

    print("amount of models:", down_queue.qsize())
    print("stage 2")

    for i in range(DOWNLOADER_PROCS):
        down_queue.put(None)
        
    print("stage 3")

    down_procs = []
    for i in range(DOWNLOADER_PROCS):
        down_procs.append(mp.Process(target=down_proc, args=(down_queue, zip_queue,)))
        down_procs[-1].start()

    print("stage 4")

    save_procs = []
    for i in range(SAVER_PROCS):
        save_procs.append(mp.Process(target=save_proc, args=(zip_queue, processed_models,)))
        save_procs[-1].start()

    tracker_proc = mp.Process(target=tracker, args=(processed_models, total_models,))
    tracker_proc.start()

    print("stage 5")

    for i in down_procs:
        i.join()

    for i in save_procs:
        i.join()

    print("done")

if __name__ == '__main__':
    main()
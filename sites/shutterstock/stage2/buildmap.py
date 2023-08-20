import os
import json
import argparse

# get instance id from args
parser = argparse.ArgumentParser()
parser.add_argument("--instances", type=int, required=True)
parser.add_argument("--disk", type=str, required=True)
args = parser.parse_args()
INSTANCES = args.instances
DISK_PATH = args.disk

TOTAL_INSTANCES = INSTANCES - 1

filelist = os.listdir(DISK_PATH)
idlist = list()

for entry in filelist:
    if entry.endswith('.mp4'):
        file_id = entry.split('.')[0]
        if str(file_id + '.json') in filelist:
            idlist.append(file_id)

finalmap = dict()

block_length = len(idlist) // TOTAL_INSTANCES
for i in range(TOTAL_INSTANCES):
    start = i * block_length
    end = (i + 1) * block_length
    if i == TOTAL_INSTANCES - 1:
        end = len(idlist)
    filelist = idlist[start:end]
    finalmap[i] = filelist

with open('map.json', 'w') as f:
    json.dump(finalmap, f)
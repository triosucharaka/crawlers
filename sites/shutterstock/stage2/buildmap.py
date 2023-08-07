import os
import json
from refrac import DISK_PATH

filelist = os.listdir(DISK_PATH)

with open('map.json', 'w') as f:
    json.dump(filelist, f)
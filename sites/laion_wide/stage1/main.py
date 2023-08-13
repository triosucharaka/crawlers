import sqlite3
import time
import logging
import psutil
import wandb
import traceback
import tqdm
from datasets import load_dataset

### Configuration ###

# General
MAX_ENTRIES = 100000000

# SQL
SQLDB_PATH = "/home/windowsuser/mount-folder/tempofunkds/laion_wide/database/map.db"
SQLDB_SAVE_BATCH_SIZE = 100000

# HuggingFace Datasets
DATASET_NAME = "laion/laion2B-en"
SPLIT = "train"

# WandB
USE_WANDB = False
WANDB_ENTITY = "peruano"
WANDB_PROJ = "laion_wide"
WANDB_NAME = f"stage1_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
LOG_MEMORY = True

### End of configuration ###

######### Logger Ignore #########
def get_memory():
    mem_info = psutil.virtual_memory()
    free_memory = f"SMU: {round(mem_info.used / 1024 ** 2, 2)}MB; {mem_info.percent}"
    return free_memory


class CustomLogger(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return f"{get_memory()} - {msg}", kwargs

logger = logging.getLogger(__name__)
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('file.log')
c_format = logging.Formatter(f'%(asctime)s - %(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter(f'%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)
logger.addHandler(c_handler)
logger.addHandler(f_handler)
logger.setLevel(logging.DEBUG)
logger = CustomLogger(logger, {})

if LOG_MEMORY:
    logger = CustomLogger(logger, {})
######### Logger Ignore #########

if USE_WANDB:
    run = wandb.init(
        project=WANDB_PROJ, 
        entity=WANDB_ENTITY, 
        name=WANDB_NAME,
        config={
            "dataset_name": DATASET_NAME,
            "split": SPLIT,
            "sql_db_path": SQLDB_PATH,
            "sql_db_save_batch_size": SQLDB_SAVE_BATCH_SIZE,
        })

required_keys = [
    'SAMPLE_ID',
    'URL',
    'TEXT',
    'HEIGHT',
    'WIDTH',
    'NSFW',
    'similarity' # NOTE: this is not a typo, the dataset has this key as 'similarity' instead of 'SIMILARITY'
]

dataset = load_dataset(DATASET_NAME, streaming=True, split=SPLIT)
conn = sqlite3.connect(SQLDB_PATH)
c = conn.cursor()

c.execute('''
          CREATE TABLE IF NOT EXISTS images
          (
                id INT PRIMARY KEY,
                url TEXT,
                text TEXT,
                height SMALLINT,
                width SMALLINT,
                nsfw TINYINT,
                similarity FLOAT(6)
          )
          ''')

def add_entry(entry: dict):
    c.execute('''
              INSERT OR REPLACE INTO images
              VALUES (?, ?, ?, ?, ?, ?, ?)
              ''',
              (
                  entry['SAMPLE_ID'],
                  entry['URL'],
                  entry['TEXT'],
                  entry['HEIGHT'],
                  entry['WIDTH'],
                  entry['NSFW'],
                  entry['similarity']
              )
              )

steps = 0

pb = tqdm.tqdm(total=MAX_ENTRIES)

def validate_aspect_ratio(aspect_ratio):
    width, height = map(float, aspect_ratio.split(':'))
    if height > 1.2 * width or width > 2.0 * height:
        return False
    else:
        return True

def validate_entry(entry: dict):
    for key in required_keys:
        if key not in entry:
            return (False, f"Key {key} not found", True)
        if entry[key] is None:
            return (False, f"Key {key} is None", True)

    if float(entry['similarity']) < 0.3:
        return (False, f"Similarity is too low: {entry['similarity']}", False)
        
    if not validate_aspect_ratio(str(entry['WIDTH']) + ':' + str(entry['HEIGHT'])):
        return (False, f"Invalid aspect ratio {entry['WIDTH']}:{entry['HEIGHT']}", False)
    
    return (True, None, None)

def standarize_entry(entry: dict):
    entry['SAMPLE_ID'] = int(entry['SAMPLE_ID'])
    entry['URL'] = str(entry['URL'])
    entry['TEXT'] = str(entry['TEXT'])
    entry['HEIGHT'] = int(entry['HEIGHT'])
    entry['WIDTH'] = int(entry['WIDTH'])
    
    nsfw = entry['NSFW']

    """
    Table of NSFW values:
    0: UNLIKELY
    1: UNSURE
    2: NSFW
    """

    if nsfw == "UNLIKELY":
        entry['NSFW'] = 0
    elif nsfw == "UNSURE":
        entry['NSFW'] = 1
    elif nsfw == "NSFW":
        entry['NSFW'] = 2
    else:
        raise ValueError(f"Invalid NSFW value: {nsfw}")
    
    entry['similarity'] = float(entry['similarity'])

    return entry

# the dataset has the values swapped for some reason...
def fix_entry(entry: dict):
    entry['HEIGHT'], entry['WIDTH'] = entry['WIDTH'], entry['HEIGHT']
    return entry

entry_batch = list()

logger.info(f"Starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")

for entry in dataset:
    try:
        if steps > MAX_ENTRIES:
            break

        entry.pop('LICENSE', None) # waste of space and nobody cares

        entry = fix_entry(entry)

        valid_entry, reason, show = validate_entry(entry)

        if not valid_entry:
            if show:
                logger.warning(f"Entry {entry['SAMPLE_ID']} is invalid: {reason}")
            continue

        entry = standarize_entry(entry)

        entry_batch.append(entry)

        steps += 1
        pb.update(1)

        if steps % 10 == 0 and USE_WANDB:
            run.log({"steps": steps})

        if steps % SQLDB_SAVE_BATCH_SIZE == 0:
            logger.info(f"Saving batch of {SQLDB_SAVE_BATCH_SIZE} entries to SQLDB...")
            for entry in entry_batch:
                add_entry(entry)
            conn.commit()
            entry_batch.clear()
            logger.info(f"Successfully saved batch to SQLDB, current step: {steps}")
    except Exception as e:
        logger.error(f"Error while processing entry {entry['SAMPLE_ID']}: {e}")
        logger.error(traceback.format_exc())

if len(entry_batch) > 0:
    logger.info(f"Saving batch of {len(entry_batch)} entries to SQLDB...")
    for entry in entry_batch:
        add_entry(entry)
    conn.commit()
    entry_batch.clear()
    logger.info("Successfully saved batch to SQLDB")

logger.info("Finished saving all entries to SQLDB")

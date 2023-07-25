from datasets import Dataset, Value
import sqlite3

conn = sqlite3.connect('/home/ubuntu/tempofunk-scrapper/sites/shutterstock/stage_1/shutterstock_map.db')

ds = Dataset.from_sql("SELECT * FROM new_videos", conn)

ds_shuff = ds.shuffle(seed=42)

ds_shuff.train_test_split(test_size=0.001)

ds_shuff.push_to_hub('shinonomelab/cleanvid_map')
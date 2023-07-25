from datasets import Dataset, Value
import sqlite3

conn = sqlite3.connect('/home/ubuntu/tempofunk-scrapper/sites/epidemicmusic/stage_1/copy.db')

ds = Dataset.from_sql("SELECT * FROM music", conn)

ds = ds.train_test_split(test_size=0.01)
ds = ds.shuffle(seed=42)

ds.push_to_hub('shinonomelab/epidemicsound-map')
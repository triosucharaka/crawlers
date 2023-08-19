import sqlite3
import pandas as pd
import argparse
from tqdm import tqdm
from sklearn.utils import shuffle

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--db_read', required=True, help='Database file to read from.')
arg_parser.add_argument('--db_write', required=True, help='Database file to write to.')
args = arg_parser.parse_args()

print("Reading...")
conn_read = sqlite3.connect(args.db_read)
df = pd.read_sql('SELECT * FROM videos', conn_read)
print("Shuffling...")
df = shuffle(df)
conn_read.close()

print("Writing...")
conn_write = sqlite3.connect(args.db_write)
df.to_sql('shuffled_videos', conn_write, if_exists='replace', index=False)
conn_write.close()

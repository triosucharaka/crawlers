import sqlite3
import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle

# Establishing the connection
conn = sqlite3.connect('/home/windowsuser/crawlers/sites/shutterstock/database.db')

# Creating a cursor
cur = conn.cursor()

# Load all data into dataframe
pbar = tqdm(total=1, desc='Loading Data', dynamic_ncols=True)  
df = pd.read_sql('SELECT * FROM videos', conn)
pbar.update()
pbar.close()

# Shuffle dataframe
pbar = tqdm(total=1, desc='Shuffling Data', dynamic_ncols=True)  
df = shuffle(df)
pbar.update()
pbar.close()

# Create a new table and write shuffled dataframe
cur.execute('DROP TABLE IF EXISTS shuffled_videos')
pbar = tqdm(total=1, desc='Writing Data', dynamic_ncols=True)
df.to_sql('shuffled_videos', conn, if_exists='replace', index=False)
pbar.update()
pbar.close()

# Closing the connection
conn.close()
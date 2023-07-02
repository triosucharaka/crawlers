import sqlite3

conn = sqlite3.connect('/home/ubuntu/tempofunk-scrapper/sites/epidemicmusic/stage_1/copy.db')

# change all NULL to "none"
conn.execute("UPDATE music SET lyrics = 'None' WHERE lyrics IS NULL")

# save
conn.commit()
conn.close()
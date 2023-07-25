import sqlite3

db_file = '/home/ubuntu/tempofunk-scrapper/sites/shutterstock/stage_1/shutterstock_map.db'
conn = sqlite3.connect(db_file)

create_new_table_query = '''
CREATE TABLE IF NOT EXISTS new_videos (
            id INT PRIMARY KEY, 
            description TEXT, 
            duration REAL,
            aspectratio TEXT, 
            videourl TEXT, 
            author TEXT, 
            categories TEXT,
            framerate REAL,
            r18 BOOL);
'''
conn.execute(create_new_table_query)

copy_data_query = '''
INSERT INTO new_videos (id, description, duration, aspectratio, videourl, author, categories, framerate, r18)
SELECT CAST(id AS INT), description, CAST(duration AS REAL), aspectratio, videourl, author, categories, CAST(framerate AS REAL), CAST(r18 AS BOOL) FROM videos;
'''
conn.execute(copy_data_query)

conn.commit()
conn.close()
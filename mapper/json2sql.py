import os
import json
import tqdm
import sqlite3

json_dir = "/data/dep/tempofunk/jsons/"

# Create and connect to SQLite database
conn = sqlite3.connect("video_list.db")
c = conn.cursor()

# Create table if not exists
c.execute('''CREATE TABLE IF NOT EXISTS videos
             (id TEXT PRIMARY KEY, description TEXT, duration TEXT, aspectratio TEXT, videourl TEXT, author TEXT, categories TEXT)''')

# Start tqdm bar
pbar = tqdm.tqdm(total=len(os.listdir(json_dir)))
video_list = []

json_list = sorted(os.listdir(json_dir))
for json_file in json_list:
    try:
        final_path = os.path.join(json_dir, json_file)
        json_file = json.loads(open(final_path, 'r').read())
        for video_entry in json_file['pageProps']['videos']:

            ja = video_entry['contributor']['publicInformation']
            if 'longbio' in ja:
                biography = ja['longbio']
            elif 'bio' in ja:
                biography = ja['bio']
            else:
                biography = None

            if 'location' in ja:
                location = ja['location']
            else:
                location = None

            author = {
                "displayname": video_entry['contributor']['publicInformation']['displayName'],
                "vanityname": video_entry['contributor']['publicInformation']['vanityUrlUsername'],
                "location": location,
                "bio": biography,
                "equipment": video_entry['contributor']['publicInformation']['equipmentList'],
                "styles": video_entry['contributor']['publicInformation']['styleList'],
                "subject": video_entry['contributor']['publicInformation']['subjectMatterList'],
            }

            categories = [category['name'] for category in video_entry['categories']]

            # Insert data into SQLite database
            c.execute("INSERT OR REPLACE INTO videos VALUES (?, ?, ?, ?, ?, ?, ?)",
                      (video_entry['id'],
                       video_entry['description'],
                       video_entry['duration'].replace('"', ' ').replace("'", " "),
                       video_entry['aspectRatioCommon'],
                       video_entry['previewVideoUrls']['mp4'],
                       json.dumps(author),
                       json.dumps(categories)))

    except Exception as e:
        print(e)
        pass

    pbar.update(1)

# Commit changes and close connection
conn.commit()
conn.close()

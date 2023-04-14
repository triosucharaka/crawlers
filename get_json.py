START_PAGE = 1
END_PAGE = 10
OUTPUT_DIR = "shutter"

import os
import json
import requests

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/110.0',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Language': 'es-ES,es;q=0.8,en-US;q=0.5,en;q=0.3',
    # 'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
}

json_dir = os.path.join(OUTPUT_DIR, "json")
if os.path.exists(json_dir) is False:
    os.makedirs(json_dir, exist_ok=True)
    current_page = START_PAGE
    while current_page < END_PAGE:
        target_url = f"https://www.shutterstock.com/_next/data/kw_ZhZGArSsA_d3Bew4Ok/es/_shutterstock/video/search/dance.json?page={current_page}&term=dance"
        print("URL:", target_url)
        response = requests.get(target_url, headers=headers)
        print(response.status_code)
        data = response.json()
        with open(os.path.join(json_dir, f'{current_page}.json'), 'w') as f:
            json.dump(data, f)
        current_page = current_page + 1

video_list = []

json_list = sorted(os.listdir(json_dir))
print(json_list)
for json_file in json_list:
    final_path = os.path.join(json_dir, json_file)
    print(final_path)
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

        entry_to_append = {
            "id": video_entry['id'],
            "description": video_entry['description'],
            "duration": video_entry['duration'].replace('"',  ' ').replace("'", " "),
            "aspectratio": video_entry['aspectRatioCommon'],
            "videourl": video_entry['previewVideoUrls']['mp4'],
            "author": {
                "displayname": video_entry['contributor']['publicInformation']['displayName'],
                "vanityname": video_entry['contributor']['publicInformation']['vanityUrlUsername'],
                "location": location,
                "bio": biography,
                "equipment": video_entry['contributor']['publicInformation']['equipmentList'],
                "styles": video_entry['contributor']['publicInformation']['styleList'],
                "subject": video_entry['contributor']['publicInformation']['subjectMatterList'],
            },
            "categories": []
        }
        for category in video_entry['categories']:
            entry_to_append['categories'].append(category['name'])
        video_list.append(entry_to_append)

json.dump(video_list, open("video_list.json", "w"))
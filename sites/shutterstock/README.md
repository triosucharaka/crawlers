# ShutterStock

## Proxies

proxies.json

```
{
    "username": "EXAMPLE-rotate",
    "password": "EXAMPLE000",
    "proxy_url": "p.webshare.io:80/"
}
```

## Stage 1: Mapping

In this stage we map shutterstock whole list of URL:CAPTION:FPS:Etc

```
python3 main.py --config /home/ubuntu/tempofunk-scrapper/config/shutterstock/stage_1.json --proxyconf /home/ubuntu/tempofunk-scrapper/config/proxies.json
```

stage_1.json

```
{
    "start_page": 1,
    "stop_page": null,
    "download_batch_size": 10,
    "save_batch_size": 20,
    "delay": 1,
    "max_retries": 1,
    "timeout_len": 4,
    "sqldb_path": "shutterstock_map.db"
}
```

## Stage 2: Processing/Encoding

In this stage we:
- Download the videos
- Dewatermark the videos
- Encode into latents and embedding
- Upload (dewatermarked video), (frame latents), (text embedding)
# EpidemicMusic

# Stage 1: Mapping

In this stage we map the site for URLs:Metadata pairs

# Stage 2: Downloading

In this stage we download the audios and pair them with their metadata.

```json
{
    "download_batch_size": 25,
    "save_megabytes_size": 100,
    "max_packaging_size": 100,
    "tar_filename_zero_padding": 6,
    "delay": 3,
    "max_retries": 2,
    "timeout_len": 2,
    "sql_file": "/home/ubuntu/tempofunk-scrapper/sites/epidemicmusic/stage_1/copy.db",
    "hf_repo": "shinonomelab/epidemicsound-basic",
    "hf_branch": "main"
}
```
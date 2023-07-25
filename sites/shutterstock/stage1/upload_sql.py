from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="/home/ubuntu/tempofunk-scrapper/sites/shutterstock/stage_1/backup.db",
    path_in_repo="sql/database.db",
    repo_id="shinonomelab/cleanvid-15m_map",
    repo_type="dataset",
)
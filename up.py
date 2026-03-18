from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
    path_or_fileobj="checkpoint.pt",
    path_in_repo="checkpoint.pt",
    repo_id="Prakhar54-byte/Checkpoints",
    repo_type="model"
)
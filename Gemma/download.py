from huggingface_hub import snapshot_download

# This downloads gemma to your local folder
snapshot_download(
    repo_id="google/gemma-4-E4B-it",
    local_dir="./AssistantGlasses/Gemma/gemma-local",
    local_dir_use_symlinks=False # Set to False to get the actual files, not links
)
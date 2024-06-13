from huggingface_hub import login, snapshot_download

login(token = "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")
snapshot_download(repo_id = "mistralai/Mistral-7B-v0.1",  local_dir = "mistral-7b-v0.1", local_dir_use_symlinks=False, revision="main")


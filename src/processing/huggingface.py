import huggingface_hub
from huggingface_hub import login, hf_hub_download,  snapshot_download

import os
from dotenv import load_dotenv

api_key = os.getenv("HUGGINGFACE_API_KEY")

login(api_key)
current_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(current_dir, "../..", ".env"))

def download_huggingface_dataset():
    # Download the dataset from Hugging Face
    # repo_id = 'huggingface/transformers'
    # snapshot_download(repo_id = repo_id, repo_type="dataset", local_dir = 'data')
    
    # Download the dataset from Hugging Face
    repo_id = 'hung20gg/chat_vi'

    download_dir = os.path.join(current_dir, f"../../example")

    snapshot_download(repo_id = repo_id, repo_type="dataset", local_dir = download_dir)

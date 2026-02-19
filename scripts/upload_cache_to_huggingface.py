#!/usr/bin/env python
"""
Upload training cache data to HuggingFace Hub.

This uploads the preprocessed cache files (with ESM embeddings baked in)
so users can train without regenerating data.

Usage:
    python scripts/upload_cache_to_huggingface.py
"""

import os
from huggingface_hub import HfApi, upload_folder

REPO_ID = "scofieldlinlin/SuperMetal"

# Cache directories to upload (essential for training)
CACHE_DIRS = [
    # Training data (30GB)
    "data/cache_allatoms/limit0_INDEXtrain_maxLigSizeNone_H0_recRad15.0_recMax24_atomRad5_atomMax8_esmEmbeddings",
    # Validation data (1.6GB)
    "data/cache_allatoms/limit0_INDEXval_maxLigSizeNone_H0_recRad15.0_recMax24_atomRad5_atomMax8_esmEmbeddings",
]


def upload_cache():
    """Upload cache directories to HuggingFace Hub."""
    api = HfApi()
    
    for cache_dir in CACHE_DIRS:
        if not os.path.exists(cache_dir):
            print(f"Skipping {cache_dir} - not found")
            continue
        
        # Get folder name
        folder_name = os.path.basename(cache_dir)
        repo_path = f"cache/{folder_name}"
        
        print(f"Uploading {cache_dir} -> {repo_path}")
        print("This may take a while for large folders...")
        
        try:
            upload_folder(
                folder_path=cache_dir,
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type="model",
            )
            print(f"  ✓ Uploaded successfully")
        except Exception as e:
            print(f"  ✗ Upload failed: {e}")
    
    print(f"\nDone! Cache data available at: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    upload_cache()

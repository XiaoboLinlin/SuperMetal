#!/usr/bin/env python
"""
Upload SuperMetal model checkpoints to HuggingFace Hub.

Usage:
    python scripts/upload_to_huggingface.py

Requires HuggingFace authentication:
    huggingface-cli login
"""

import os
from huggingface_hub import HfApi, create_repo

# Repository configuration
REPO_ID = "scofieldlinlin/SuperMetal"

# Files to upload
FILES_TO_UPLOAD = [
    # Score model
    {
        "local_path": "workdir/large_all_atoms_model/best_model.pt",
        "repo_path": "score_model/best_model.pt"
    },
    {
        "local_path": "workdir/large_all_atoms_model/model_parameters.yml",
        "repo_path": "score_model/model_parameters.yml"
    },
    # Confidence model
    {
        "local_path": "workdir/large_confidence_model/best_model.pt",
        "repo_path": "confidence_model/best_model.pt"
    },
    {
        "local_path": "workdir/large_confidence_model/model_parameters.yml",
        "repo_path": "confidence_model/model_parameters.yml"
    },
]


def upload_models():
    """Upload model checkpoints to HuggingFace Hub."""
    api = HfApi()
    
    # Create repository if it doesn't exist
    try:
        create_repo(REPO_ID, repo_type="model", exist_ok=True)
        print(f"Repository '{REPO_ID}' is ready")
    except Exception as e:
        print(f"Repository setup: {e}")
    
    # Upload each file
    for file_info in FILES_TO_UPLOAD:
        local_path = file_info["local_path"]
        repo_path = file_info["repo_path"]
        
        if not os.path.exists(local_path):
            print(f"Skipping {local_path} - file not found")
            continue
        
        print(f"Uploading {local_path} -> {repo_path}")
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type="model",
            )
            print(f"  ✓ Uploaded successfully")
        except Exception as e:
            print(f"  ✗ Upload failed: {e}")
    
    print(f"\nDone! Models available at: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    upload_models()

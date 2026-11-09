"""Model and dataset downloading utilities for HuggingFace Hub"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
import json

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "proteorift"

repo_default_id = "SaeedLab"

def download_models(cache_dir=None):
    """Download ProteoRift and Specollate models from HuggingFace Hub
    
    Args:
        cache_dir: Optional custom cache directory. Defaults to ~/.cache/proteorift
        
    Returns:
        tuple: (proteorift_model_path, specollate_model_path)
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading ProteoRift model from HuggingFace Hub...")
    proteorift_config_path = hf_hub_download(
        repo_id=f"{repo_default_id}/ProteoRift",
        filename="config.json",
        cache_dir=cache_dir,
        repo_type="model"
    )
    # load file name from config
    with open(proteorift_config_path, 'r') as f:
        config = json.load(f)
    model_filename = config.get("model_name", "pytorch_model.bin")

    proteorift_path = hf_hub_download(
        repo_id=f"{repo_default_id}/ProteoRift",
        filename=model_filename,
        cache_dir=cache_dir,
        repo_type="model"
    )
    print(f"ProteoRift model downloaded to: {proteorift_path}")
    

    print("Downloading Specollate model from HuggingFace Hub...")
    specollate_config_path = hf_hub_download(
        repo_id=f"{repo_default_id}/SpeCollate",
        filename="config.json",
        cache_dir=cache_dir,
        repo_type="model"
    )
    ## load file name from config
    with open(specollate_config_path, 'r') as f:
        config = json.load(f)
    model_filename = config.get("model_name", "specollate_model_weights.pt")

    specollate_path = hf_hub_download(
        repo_id=f"{repo_default_id}/SpeCollate",
        filename=model_filename,
        cache_dir=cache_dir,
        repo_type="model"
    )
    print(f"Specollate model downloaded to: {specollate_path}")

    print("Models downloaded successfully!")
    return proteorift_path, specollate_path


def load_sample_data(cache_dir=None, use_preprocessed=True):
    """Load sample data from HuggingFace Datasets
    
    Args:
        cache_dir: Optional custom cache directory
        use_preprocessed: If True, load preprocessed specs.pkl; else load raw MGF files
        
    Returns:
        dict: Paths to sample data files
    """
    print("Downloading sample data from HuggingFace...")
    
    # Use snapshot_download to get the entire dataset repository
    repo_path = snapshot_download(
        repo_id=f"{repo_default_id}/sample-data-msms-search",
        repo_type="dataset",
        cache_dir=cache_dir
    )
    repo_path = Path(repo_path)
    
    # Get paths from downloaded repository
    if use_preprocessed:
        prep_path = repo_path / "preprocessed"
        peptide_db = repo_path / "raw" / "peptide_database"
        
        if not prep_path.exists() or not (prep_path / "specs.pkl").exists():
            print("Warning: Preprocessed data not found, will use raw files")
            use_preprocessed = False
        else:
            print(f"Using preprocessed data: {prep_path}")
            return {
                'prep_path': str(prep_path),
                'peptide_db': str(peptide_db),
            }
    
    if not use_preprocessed:
        print("Using raw MGF files (will be preprocessed)")
        mgf_dir = repo_path / "raw" / "spectra"
        peptide_db = repo_path / "raw" / "peptide_database"
        return {
            'mgf_dir': str(mgf_dir),
            'peptide_db': str(peptide_db),
        }


def get_model_paths(cache_dir=None):
    """Get paths to cached models or download if not present
    
    Args:
        cache_dir: Optional custom cache directory
        
    Returns:
        tuple: (proteorift_model_path, specollate_model_path)
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir = Path(cache_dir)
    
    # Check if models already exist in cache
    proteorift_cached = cache_dir / "models--proteorift--proteorift-base" / "snapshots"
    specollate_cached = cache_dir / "models--proteorift--specollate-base" / "snapshots"
    
    if proteorift_cached.exists() and specollate_cached.exists():
        print("Using cached models...")
        # Find the actual model files in the snapshot directories
        try:
            proteorift_path = list(proteorift_cached.rglob("proteorift_model_weights.pt"))[0]
            specollate_path = list(specollate_cached.rglob("specollate_model_weights.pt"))[0]
            return str(proteorift_path), str(specollate_path)
        except IndexError:
            pass
    
    # Download if not cached
    return download_models(cache_dir)

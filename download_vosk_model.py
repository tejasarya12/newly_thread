"""
Helper script to download and set up Vosk ASR model
"""
import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def download_vosk_model(model_name: str = "vosk-model-en-us-0.22", target_dir: str = "./models"):
    """
    Download and extract a Vosk model.
    
    Args:
        model_name: Name of the model (e.g., "vosk-model-en-us-0.22")
        target_dir: Directory to extract the model to
    """
    import urllib.request
    import zipfile
    import shutil
    
    # Model URLs (update if needed)
    model_urls = {
        "vosk-model-en-us-0.22": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip",
        "vosk-model-small-en-us-0.15": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
    }
    
    if model_name not in model_urls:
        logger.error(f"Unknown model: {model_name}")
        logger.info(f"Available models: {list(model_urls.keys())}")
        return False
    
    url = model_urls[model_name]
    model_dir = Path(target_dir) / model_name
    zip_path = Path(target_dir) / f"{model_name}.zip"
    
    # Create target directory
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if model already exists
    if model_dir.exists():
        logger.info(f"Model directory already exists: {model_dir}")
        logger.info("Skipping download. Delete the directory if you want to re-download.")
        return True
    
    # Download
    logger.info(f"Downloading {model_name} from {url}...")
    logger.info(f"This may take a few minutes (model size: ~40-1800 MB)...")
    
    try:
        urllib.request.urlretrieve(url, zip_path, reporthook=lambda block_num, block_size, total_size: 
            logger.info(f"Downloaded {min(block_num * block_size, total_size) / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB") 
            if block_num % 10 == 0 else None)
        
        logger.info(f"Download complete: {zip_path}")
        
        # Extract
        logger.info(f"Extracting to {model_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        
        # Clean up zip file
        zip_path.unlink()
        logger.info(f"✅ Model extracted successfully to: {model_dir}")
        
        # Verify
        if (model_dir / "am").exists() or (model_dir / "graph").exists():
            logger.info("✅ Model verification passed")
            return True
        else:
            logger.warning("⚠️ Model directory exists but may be missing required files")
            return False
            
    except Exception as e:
        logger.error(f"Failed to download/extract model: {e}", exc_info=True)
        if zip_path.exists():
            zip_path.unlink()
        return False


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Vosk ASR model")
    parser.add_argument(
        "--model",
        default="vosk-model-en-us-0.22",
        help="Model name (default: vosk-model-en-us-0.22)",
    )
    parser.add_argument(
        "--target",
        default="./models",
        help="Target directory (default: ./models)",
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Vosk Model Downloader")
    logger.info("=" * 60)
    
    success = download_vosk_model(args.model, args.target)
    
    if success:
        logger.info("=" * 60)
        logger.info("✅ Setup complete!")
        logger.info(f"Model location: {Path(args.target) / args.model}")
        logger.info("You can now run your application.")
        logger.info("=" * 60)
    else:
        logger.error("=" * 60)
        logger.error("❌ Setup failed. Please check the error messages above.")
        logger.error("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()

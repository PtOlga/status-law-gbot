#!/usr/bin/env python
# repair_conversation_ids.py
"""
Script to restore empty conversation_ids in chat history files.
One-time operation with hardcoded paths.
"""

import os
import sys
import json
import codecs
import datetime
import logging
import tempfile
from huggingface_hub import HfApi
from dotenv import load_dotenv
import time
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

# PATHS AND PARAMETERS CONFIGURATION
# =============================

# Modify these values according to your configuration
CHAT_HISTORY_PATH = './chat_history'  # Path to local chat history files
DATASET_ID = 'Rulga/status-law-knowledge-base'  # HuggingFace dataset ID
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")  # HuggingFace API access token

if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")

# Dataset paths
DATASET_CHAT_HISTORY_PATH = "chat_history"
DATASET_VECTOR_STORE_PATH = "vector_store"
DATASET_FINE_TUNED_PATH = "fine_tuned_models"
DATASET_ANNOTATIONS_PATH = "annotations"
DATASET_ERROR_LOGS_PATH = "error_logs"
DATASET_PREFERENCES_PATH = "preferences/user_preferences.json"

# If True, script won't make actual changes (test mode)
DRY_RUN = False

# If True, script will update only local files
LOCAL_ONLY = False

# Add temporary directory for downloads
TEMP_DIR = tempfile.mkdtemp()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("repair_conversation_ids.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configure stdout encoding
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=60, min=60, max=180)
)
def safe_api_call(func, *args, **kwargs):
    """Wrapper for API calls with retry logic"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if "429 Client Error: Too Many Requests" in str(e):
            logger.warning("Rate limit hit, waiting before retry...")
            raise  # Let retry handle it
        raise  # Other errors

def repair_conversation_ids():
    """
    Restore conversation_ids in chat history files directly in HuggingFace dataset
    """
    try:
        api = HfApi(token=HF_TOKEN)
        
        # List all files with retry
        files = safe_api_call(
            api.list_repo_files,
            repo_id=DATASET_ID,
            repo_type="dataset"
        )
        
        chat_files = [f for f in files 
                     if f.startswith(DATASET_CHAT_HISTORY_PATH) and 
                     f.endswith('.json') and 
                     os.path.basename(f).startswith('None_')]
        
        logger.info(f"Found {len(chat_files)} files with 'None_' prefix in dataset")
        
        repaired_count = 0
        skipped_count = 0
        error_count = 0
        
        for file_path in chat_files:
            try:
                # Add delay between files
                time.sleep(2)  # 2 seconds between files
                
                # Download file content with retry
                file_content = safe_api_call(
                    api.hf_hub_download,
                    repo_id=DATASET_ID,
                    repo_type="dataset",
                    filename=file_path,
                    local_dir=TEMP_DIR,
                    local_dir_use_symlinks=False
                )
                
                with open(file_content, 'r', encoding='utf-8') as f:
                    chat_data = json.load(f)
                
                # Generate new ID based on timestamp and file details
                timestamp_str = chat_data.get('timestamp', '')
                try:
                    timestamp_dt = datetime.datetime.fromisoformat(timestamp_str)
                    time_part = timestamp_dt.strftime('%Y%m%d%H%M%S')
                except (ValueError, TypeError):
                    time_part = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
                
                filename = os.path.basename(file_path)
                filename_part = os.path.splitext(filename)[0].replace('None_', '')
                if len(filename_part) > 10:
                    filename_part = filename_part[:10]
                
                new_id = f"conv_{time_part}_{filename_part}"
                chat_data['conversation_id'] = new_id
                
                if not DRY_RUN:
                    # Create new filename without None_ prefix
                    new_filename = filename.replace('None_', '')
                    new_path = os.path.join(
                        os.path.dirname(file_path),
                        new_filename
                    )
                    
                    # First move the old file to archive
                    archive_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    archive_filename = f"archive/None_{archive_timestamp}_{filename}"
                    archive_path = os.path.join(DATASET_CHAT_HISTORY_PATH, archive_filename)
                    
                    # Create archive directory if it doesn't exist
                    try:
                        api.upload_file(
                            path_or_fileobj=b"",
                            path_in_repo=f"{DATASET_CHAT_HISTORY_PATH}/archive/.gitkeep",
                            repo_id=DATASET_ID,
                            repo_type="dataset"
                        )
                    except Exception:
                        pass  # Directory might already exist
                    
                    # Move old file to archive with retry
                    safe_api_call(
                        api.upload_file,
                        path_or_fileobj=file_content,
                        path_in_repo=archive_path,
                        repo_id=DATASET_ID,
                        repo_type="dataset"
                    )
                    
                    # Upload updated content with retry
                    json_content = json.dumps(chat_data, ensure_ascii=False, indent=2)
                    safe_api_call(
                        api.upload_file,
                        path_or_fileobj=json_content.encode('utf-8'),
                        path_in_repo=new_path,
                        repo_id=DATASET_ID,
                        repo_type="dataset"
                    )
                    
                    # Only after successful upload of both files, delete the original with retry
                    safe_api_call(
                        api.delete_file,
                        path_in_repo=file_path,
                        repo_id=DATASET_ID,
                        repo_type="dataset"
                    )
                    
                    logger.info(f"Repaired: {filename} -> {new_filename} (archived as {archive_filename}) - New ID: {new_id}")
                    repaired_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                error_count += 1
                continue  # Skip to next file on error
                
        logger.info(f"Repair completed: {repaired_count} files repaired, {skipped_count} skipped, {error_count} errors")
        return repaired_count
        
    except Exception as e:
        logger.error(f"Error accessing dataset: {str(e)}")
        return 0

if __name__ == "__main__":
    # Display configuration information
    logger.info("=== CONFIGURATION ===")
    logger.info(f"Chat history path: {CHAT_HISTORY_PATH}")
    logger.info(f"Dataset ID: {DATASET_ID}")
    logger.info(f"Test mode: {'Yes' if DRY_RUN else 'No'}")
    logger.info(f"Local only: {'Yes' if LOCAL_ONLY else 'No'}")
    logger.info("==================")
    
    # Start repair process
    repaired = repair_conversation_ids()
    
    if DRY_RUN:
        logger.info(f"TEST MODE: Would have repaired {repaired} files")
    else:
        logger.info(f"Successfully repaired {repaired} files")









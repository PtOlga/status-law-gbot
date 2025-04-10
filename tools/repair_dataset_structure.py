import logging
import time
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

# Добавляем корневую директорию проекта в PYTHONPATH
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from huggingface_hub import HfApi
from config.settings import (
    DATASET_ID,
    DATASET_CHAT_HISTORY_PATH,
    HF_TOKEN
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("repair_dataset_structure.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def repair_chat_files_structure():
    """
    Move misplaced chat files from root to existing chat_history directory
    """
    try:
        api = HfApi(token=HF_TOKEN)
        
        # Получаем список всех файлов в датасете
        files = api.list_repo_files(
            repo_id=DATASET_ID,
            repo_type="dataset"
        )
        
        # Находим только файлы чата в корневой директории (без пути)
        misplaced_files = [
            f for f in files 
            if f.endswith('.json') and 
            '/' not in f and  # только файлы в корне
            '-' in f  # характерный признак файлов чата (timestamp)
        ]
        
        logger.info(f"Found {len(misplaced_files)} misplaced chat files")
        
        moved_count = 0
        error_count = 0
        
        for file_path in misplaced_files:
            try:
                # Проверяем флаг остановки
                if hasattr(repair_chat_files_structure, 'stop_flag') and repair_chat_files_structure.stop_flag:
                    logger.info("Stopping process...")
                    break

                # Добавляем задержку между операциями
                time.sleep(2)
                
                # Скачиваем содержимое файла
                file_content = api.hf_hub_download(
                    repo_id=DATASET_ID,
                    filename=file_path,
                    repo_type="dataset"
                )
                
                # Перемещаем в существующую chat_history директорию
                new_path = f"chat_history/{file_path}"
                
                # Загружаем файл в chat_history
                with open(file_content, 'rb') as f:
                    api.upload_file(
                        path_or_fileobj=f,
                        path_in_repo=new_path,
                        repo_id=DATASET_ID,
                        repo_type="dataset"
                    )
                
                # Удаляем файл из корневой директории
                api.delete_file(
                    path_in_repo=file_path,
                    repo_id=DATASET_ID,
                    repo_type="dataset"
                )
                
                logger.info(f"Moved {file_path} to {new_path}")
                moved_count += 1
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                error_count += 1
                continue
        
        logger.info(f"Successfully moved {moved_count} files from root to chat_history")
        if error_count > 0:
            logger.warning(f"Failed to process {error_count} files")
            
    except Exception as e:
        logger.error(f"Error accessing dataset: {str(e)}")

def fix_duplicated_paths():
    """
    Fix duplicated chat_history paths in filenames
    """
    try:
        api = HfApi(token=HF_TOKEN)
        
        # Получаем только файлы из папки chat_history с дублированным путем
        wrong_paths = [
            f for f in api.list_repo_files(
                repo_id=DATASET_ID,
                repo_type="dataset"
            )
            if f.startswith('chat_history/') and 
            f.endswith('.json') and
            'chat_history\\' in f  # ищем файлы с Windows-путем в имени
        ]
        
        logger.info(f"Found {len(wrong_paths)} files with duplicated chat_history path")
        
        fixed_count = 0
        error_count = 0
        
        for file_path in wrong_paths:
            try:
                # Проверяем флаг остановки
                if hasattr(fix_duplicated_paths, 'stop_flag') and fix_duplicated_paths.stop_flag:
                    logger.info("Stopping process...")
                    break

                # Добавляем задержку между операциями
                time.sleep(2)
                
                # Скачиваем содержимое файла
                file_content = api.hf_hub_download(
                    repo_id=DATASET_ID,
                    filename=file_path,
                    repo_type="dataset"
                )
                
                # Создаем правильный путь
                filename = os.path.basename(file_path).replace('chat_history\\', '')
                new_path = f"chat_history/{filename}"
                
                # Загружаем файл с правильным путем
                with open(file_content, 'rb') as f:
                    api.upload_file(
                        path_or_fileobj=f,
                        path_in_repo=new_path,
                        repo_id=DATASET_ID,
                        repo_type="dataset"
                    )
                
                # Удаляем файл со старым путем
                api.delete_file(
                    path_in_repo=file_path,
                    repo_id=DATASET_ID,
                    repo_type="dataset"
                )
                
                logger.info(f"Renamed {file_path} to {new_path}")
                fixed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                error_count += 1
                continue
        
        logger.info(f"Successfully renamed {fixed_count} files")
        if error_count > 0:
            logger.warning(f"Failed to process {error_count} files")
            
    except Exception as e:
        logger.error(f"Error accessing dataset: {str(e)}")

if __name__ == "__main__":
    try:
        logger.info("=== Starting Dataset Structure Repair ===")
        logger.info(f"Dataset ID: {DATASET_ID}")
        
        # Сначала перемещаем файлы из корня
        #repair_chat_files_structure()
        
        # Затем исправляем пути
        logger.info("=== Starting Path Fix ===")
        fix_duplicated_paths()
        
        logger.info("=== Repair Complete ===")
    except KeyboardInterrupt:
        logger.info("\nReceived keyboard interrupt, stopping gracefully...")
        repair_chat_files_structure.stop_flag = True
        fix_duplicated_paths.stop_flag = True
        time.sleep(3)
        logger.info("Process stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")







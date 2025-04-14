"""
Module for managing dataset on Hugging Face Hub
"""

import os
import json
import tempfile
from typing import Tuple, List, Dict, Any, Optional, Union
from datetime import datetime
import logging
from huggingface_hub import HfApi, HfFolder
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import (
    VECTOR_STORE_PATH,
    HF_TOKEN,
    EMBEDDING_MODEL,
    DATASET_ID,
    CHAT_HISTORY_PATH,
    DATASET_CHAT_HISTORY_PATH,
    DATASET_VECTOR_STORE_PATH,
    DATASET_FINE_TUNED_PATH,
    DATASET_ANNOTATIONS_PATH
)
from langchain_huggingface import HuggingFaceEmbeddings
import logging

logger = logging.getLogger(__name__)

class DatasetManager:
    def __init__(self, dataset_name: Optional[str] = None, token: Optional[str] = None):
        self.dataset_name = dataset_name or DATASET_ID
        self.token = token if token else HF_TOKEN
        self.api = HfApi(token=self.token)
        
        # Use paths from settings
        self.vector_store_path = DATASET_VECTOR_STORE_PATH
        self.chat_history_path = DATASET_CHAT_HISTORY_PATH
        self.fine_tuned_path = DATASET_FINE_TUNED_PATH
        self.annotations_path = DATASET_ANNOTATIONS_PATH
        
    # Добавьте этот метод в класс DatasetManager в файле src/knowledge_base/dataset.py
    
def download_vector_store(self) -> Tuple[bool, Union[FAISS, str]]:
    """Download vector store from dataset"""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.debug(f"Downloading to temporary directory: {temp_dir}")
            
            try:
                # Download vector store files
                index_path = self.api.hf_hub_download(
                    repo_id=self.dataset_name,
                    filename="vector_store/index.faiss",
                    repo_type="dataset",
                    local_dir=temp_dir
                )
                logger.debug(f"Downloaded index.faiss to: {index_path}")
                
                config_path = self.api.hf_hub_download(
                    repo_id=self.dataset_name,
                    filename="vector_store/index.pkl",
                    repo_type="dataset",
                    local_dir=temp_dir
                )
                logger.debug(f"Downloaded index.pkl to: {config_path}")
                
                # Initialize embeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL,
                    model_kwargs={'device': 'cpu'}
                )
                
                # Load vector store
                vector_store = FAISS.load_local(
                    folder_path=os.path.join(temp_dir, "vector_store"),
                    embeddings=embeddings
                )
                
                return True, vector_store
                
            except Exception as e:
                logger.error(f"Error downloading vector store: {str(e)}")
                return False, f"Error downloading vector store: {str(e)}"
                
    except Exception as e:
        logger.error(f"Error in download_vector_store: {str(e)}")
        return False, str(e)

def get_last_update_date(self):
    """
    Получает дату последнего обновления базы знаний.
    
    Returns:
        str: Дата последнего обновления в формате ISO или None, если информация недоступна
    """
    try:
        # Попробуем получить метаданные из датасета
        api = HfApi(token=self.hf_token)
        
        # Сначала проверим, есть ли специальный файл метаданных
        files = api.list_repo_files(
            repo_id=self.dataset_id,
            repo_type="dataset"
        )
        
        metadata_file = "vector_store/metadata.json"
        
        if metadata_file in files:
            # Скачиваем файл метаданных
            temp_dir = tempfile.mkdtemp()
            metadata_path = os.path.join(temp_dir, "metadata.json")
            
            api.hf_hub_download(
                repo_id=self.dataset_id,
                repo_type="dataset",
                filename=metadata_file,
                local_dir=temp_dir,
                local_dir_use_symlinks=False
            )
            
            # Открываем и читаем дату из метаданных
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                return metadata.get("last_updated", None)
        
        # Если специальный файл не найден, можно использовать дату последнего коммита
        # для директории vector_store
        last_commit = api.get_repo_info(
            repo_id=self.dataset_id,
            repo_type="dataset"
        )
        
        # Получаем дату последнего коммита
        if hasattr(last_commit, "lastModified"):
            return last_commit.lastModified
        
        return None
    except Exception as e:
        logger.error(f"Error getting last update date: {str(e)}")
        return None    

    def init_dataset_structure(self) -> Tuple[bool, str]:
        """
        Initialize dataset structure with required directories
        
        Returns:
            (success, message)
        """
        try:
            # Check if repository exists
            try:
                self.api.repo_info(repo_id=self.dataset_name, repo_type="dataset")
            except Exception:
                # Create repository if it doesn't exist
                self.api.create_repo(repo_id=self.dataset_name, repo_type="dataset", private=True)
            
            # Create empty .gitkeep files to maintain structure
            directories = ["vector_store", "chat_history", "documents"]
            
            for directory in directories:
                with tempfile.NamedTemporaryFile(delete=False) as temp:
                    temp_path = temp.name
                
                try:
                    self.api.upload_file(
                        path_or_fileobj=temp_path,
                        path_in_repo=f"{directory}/.gitkeep",
                        repo_id=self.dataset_name,
                        repo_type="dataset"
                    )
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            
            return True, "Dataset structure initialized successfully"
            
        except Exception as e:
            return False, f"Error initializing dataset structure: {str(e)}"

    def upload_vector_store(self, vector_store: FAISS) -> Tuple[bool, str]:
        """
        Upload vector store to dataset
        
        Args:
            vector_store: FAISS vector store to upload
        
        Returns:
            (success, message)
        """
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save vector store to temporary directory
                vector_store.save_local(folder_path=temp_dir)
                
                index_path = os.path.join(temp_dir, "index.faiss")
                config_path = os.path.join(temp_dir, "index.pkl")
                
                # Add debug logging
                print(f"Debug - Checking files before upload:")
                print(f"index.faiss exists: {os.path.exists(index_path)}, size: {os.path.getsize(index_path) if os.path.exists(index_path) else 0} bytes")
                print(f"index.pkl exists: {os.path.exists(config_path)}, size: {os.path.getsize(config_path) if os.path.exists(config_path) else 0} bytes")
                
                if not os.path.exists(index_path) or not os.path.exists(config_path):
                    return False, "Vector store files not created"
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # First save old files to archive if they exist
                try:
                    # Check for existing files
                    self.api.hf_hub_download(
                        repo_id=self.dataset_name,
                        filename="vector_store/index.faiss",
                        repo_type="dataset"
                    )
                    
                    # If file exists, create archive copy
                    self.api.upload_file(
                        path_or_fileobj=index_path,
                        path_in_repo=f"vector_store/archive/index_{timestamp}.faiss",
                        repo_id=self.dataset_name,
                        repo_type="dataset"
                    )
                    
                    self.api.upload_file(
                        path_or_fileobj=config_path,
                        path_in_repo=f"vector_store/archive/index_{timestamp}.pkl",
                        repo_id=self.dataset_name,
                        repo_type="dataset"
                    )
                except Exception:
                    # If no files exist, create archive directory
                    with tempfile.NamedTemporaryFile(delete=False) as temp:
                        temp_path = temp.name
                    
                    try:
                        self.api.upload_file(
                            path_or_fileobj=temp_path,
                            path_in_repo="vector_store/archive/.gitkeep",
                            repo_id=self.dataset_name,
                            repo_type="dataset"
                        )
                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                
                # Upload current files
                self.api.upload_file(
                    path_or_fileobj=index_path,
                    path_in_repo="vector_store/index.faiss",
                    repo_id=self.dataset_name,
                    repo_type="dataset"
                )
                
                self.api.upload_file(
                    path_or_fileobj=config_path,
                    path_in_repo="vector_store/index.pkl",
                    repo_id=self.dataset_name,
                    repo_type="dataset"
                )
                
                # Update metadata about last update
                metadata = {
                    "last_update": timestamp,
                    "version": "1.0"
                }
                
                with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as temp:
                    json.dump(metadata, temp, ensure_ascii=False, indent=2)
                    temp_name = temp.name
                
                try:
                    self.api.upload_file(
                        path_or_fileobj=temp_name,
                        path_in_repo="vector_store/metadata.json",
                        repo_id=self.dataset_name,
                        repo_type="dataset"
                    )
                finally:
                    if os.path.exists(temp_name):
                        os.remove(temp_name)
                
                return True, "Vector store uploaded successfully"
                
        except Exception as e:
            return False, f"Error uploading vector store: {str(e)}"

    def download_vector_store(self) -> Tuple[bool, Union[FAISS, str]]:
        """Download vector store from dataset"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                print(f"Downloading to temporary directory: {temp_dir}")
                
                # Download files to temporary directory
                try:
                    index_path = self.api.hf_hub_download(
                        repo_id=self.dataset_name,
                        filename="vector_store/index.faiss",
                        repo_type="dataset",
                        local_dir=temp_dir
                    )
                    print(f"Downloaded index.faiss to: {index_path}")
                    
                    config_path = self.api.hf_hub_download(
                        repo_id=self.dataset_name,
                        filename="vector_store/index.pkl",
                        repo_type="dataset",
                        local_dir=temp_dir
                    )
                    print(f"Downloaded index.pkl to: {config_path}")
                    
                    # Verify files exist
                    if not os.path.exists(index_path) or not os.path.exists(config_path):
                        return False, f"Downloaded files not found at {temp_dir}"
                    
                    # Load vector store from temporary directory
                    embeddings = HuggingFaceEmbeddings(
                        model_name=EMBEDDING_MODEL,
                        model_kwargs={'device': 'cpu'}
                    )
                    
                    # Use the directory containing the files
                    store_dir = os.path.dirname(index_path)
                    print(f"Loading vector store from: {store_dir}")
                    
                    vector_store = FAISS.load_local(
                        store_dir,
                        embeddings,
                        allow_dangerous_deserialization=True
                    )
                    return True, vector_store
                    
                except Exception as e:
                    return False, f"Failed to download vector store: {str(e)}"
                
        except Exception as e:
            return False, f"Error downloading vector store: {str(e)}"

    def save_chat_history(self, conversation_id: str, messages: List[Dict[str, str]]) -> Tuple[bool, str]:
        try:
            timestamp = datetime.now().isoformat()
            filename = f"{self.chat_history_path}/{conversation_id}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
            
            chat_data = {
                "conversation_id": conversation_id,
                "timestamp": timestamp,
                "history": messages  # Changed from 'messages' to 'history'
            }
            
            if not self._validate_chat_structure(chat_data):
                return False, "Invalid chat history structure"
            
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False, encoding="utf-8") as temp:
                json.dump(chat_data, temp, ensure_ascii=False, indent=2)
                temp.flush()
            
            return True, "Chat history saved successfully"
        except Exception as e:
            return False, f"Error saving chat history: {str(e)}"

    def _validate_chat_structure(self, chat_data: Dict) -> bool:
        required_fields = {"conversation_id", "timestamp", "history"}
        if not all(field in chat_data for field in required_fields):
            return False
        
        if not isinstance(chat_data["history"], list):
            return False
        
        for message in chat_data["history"]:
            if not all(field in message for field in ["role", "content", "timestamp"]):
                return False
            
        return True

    def get_chat_history(self, conversation_id: Optional[str] = None) -> Tuple[bool, Any]:
        try:
            logger.info(f"Attempting to get chat history from dataset {self.dataset_name}")
            
            # Get all files from repository
            files = self.api.list_repo_files(
                repo_id=self.dataset_name,
                repo_type="dataset"
            )
            
            # Filter only files from chat_history directory using settings
            chat_files = [f for f in files if f.startswith(f"{CHAT_HISTORY_PATH}/")]
            logger.info(f"Found {len(chat_files)} files in {CHAT_HISTORY_PATH}")
            
            if conversation_id:
                chat_files = [f for f in chat_files if conversation_id in f]
            
            if not chat_files:
                logger.warning("No chat history files found")
                return True, []
            
            chat_histories = []
            with tempfile.TemporaryDirectory() as temp_dir:
                for file in chat_files:
                    if file.endswith(".gitkeep"):
                        continue
                    
                    try:
                        local_file = self.api.hf_hub_download(
                            repo_id=self.dataset_name,
                            filename=file,
                            repo_type="dataset",
                            local_dir=temp_dir
                        )
                        
                        with open(local_file, "r", encoding="utf-8") as f:
                            chat_data = json.load(f)
                            logger.debug(f"Loaded chat data: {chat_data}")  # Debug log
                            
                            if not isinstance(chat_data, dict):
                                logger.error(f"Chat data is not a dictionary in {file}")
                                continue
                            
                            # Get messages from either 'messages' or 'history' key
                            messages = None
                            if "messages" in chat_data:
                                messages = chat_data["messages"]
                            elif "history" in chat_data:
                                messages = chat_data["history"]
                            
                            if not messages:
                                logger.error(f"No messages found in {file}")
                                continue
                                
                            if not isinstance(messages, list):
                                logger.error(f"Messages is not a list in {file}")
                                continue
                            
                            # Create standardized format
                            standardized_data = {
                                "conversation_id": chat_data.get("conversation_id", "unknown"),
                                "timestamp": chat_data.get("timestamp", datetime.now().isoformat()),
                                "messages": messages
                            }
                            
                            chat_histories.append(standardized_data)
                            logger.info(f"Successfully loaded chat data from {file}")
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON in file {file}: {str(e)}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing file {file}: {e}")
                        continue
            
            if not chat_histories:
                logger.warning("No valid chat histories found")
            else:
                logger.info(f"Successfully loaded {len(chat_histories)} chat histories")
            
            return True, chat_histories
            
        except Exception as e:
            logger.error(f"Error getting chat history: {str(e)}")
            return False, str(e)

    def upload_document(self, file_path: str, document_id: Optional[str] = None) -> Tuple[bool, str]:
        """
        Upload document to the dataset
        
        Args:
            file_path: Path to the document file
            document_id: Document identifier (if None, uses filename)
            
        Returns:
            (success, message)
        """
        try:
            if not os.path.exists(file_path):
                return False, f"File not found: {file_path}"
                
            # Use filename as document_id if not specified
            if document_id is None:
                document_id = os.path.basename(file_path)
                
            # Add timestamp to filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"documents/{document_id}_{timestamp}{os.path.splitext(file_path)[1]}"
            
            # Upload file
            self.api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=filename,
                repo_id=self.dataset_name,
                repo_type="dataset"
            )
            
            return True, f"Document uploaded successfully: {filename}"
        except Exception as e:
            return False, f"Error uploading document: {str(e)}"

def test_dataset_connection(token: Optional[str] = None) -> Tuple[bool, str]:
    """
    Test function to check dataset connection
    
    Args:
        token: Hugging Face Hub access token
        
    Returns:
        (success, message)
    """
    try:
        manager = DatasetManager(token=token)
        success, message = manager.init_dataset_structure()
        
        if not success:
            return False, message
            
        print(f"Initialization test: {message}")
        
        return True, "Dataset connection is working"
    except Exception as e:
        return False, f"Dataset connection error: {str(e)}"

if __name__ == "__main__":
    # Test connection
    success, message = test_dataset_connection()
    print(message)

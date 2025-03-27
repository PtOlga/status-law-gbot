"""
Module for managing dataset on Hugging Face Hub
"""

import os
import json
import tempfile
from typing import Tuple, List, Dict, Any, Optional, Union
from datetime import datetime
from huggingface_hub import HfApi, HfFolder
from langchain_community.vectorstores import FAISS
from config.settings import VECTOR_STORE_PATH, HF_TOKEN

class DatasetManager:
    def __init__(self, dataset_name="Rulga/status-law-knowledge-base", token: Optional[str] = None):
        """
        Initialize dataset manager
        
        Args:
            dataset_name: Hugging Face Hub dataset name
            token: Hugging Face access token (if None, will use HF_TOKEN from settings)
        """
        self.token = token if token else HF_TOKEN
        if not self.token:
            raise ValueError("Hugging Face token not found. Please set HUGGINGFACE_TOKEN environment variable")
        
        self.dataset_name = dataset_name
        self.api = HfApi(token=self.token)
        
        # Проверяем/создаем репозиторий при инициализации
        try:
            self.api.repo_info(repo_id=self.dataset_name, repo_type="dataset")
        except Exception:
            print(f"Создаем новый репозиторий датасета: {self.dataset_name}")
            self.api.create_repo(
                repo_id=self.dataset_name,
                repo_type="dataset",
                private=True
            )

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

    def upload_vector_store(self) -> Tuple[bool, str]:
        """
        Upload vector store to dataset
        
        Returns:
            (success, message)
        """
        try:
            if not os.path.exists(VECTOR_STORE_PATH):
                return False, "Vector store directory not found"
                
            index_path = os.path.join(VECTOR_STORE_PATH, "index.faiss")
            config_path = os.path.join(VECTOR_STORE_PATH, "index.pkl")
            
            if not (os.path.exists(index_path) and os.path.exists(config_path)):
                return False, "Vector store files not found"
            
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

    def download_vector_store(self, force: bool = False) -> Tuple[bool, Union[FAISS, str]]:
        """
        Download vector store from dataset
        
        Args:
            force: Force download even if local files exist
        
        Returns:
            (success, vector_store or error message)
        """
        try:
            # Check if local files exist and force is False
            if not force and os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
                return True, "Local files exist"
            
            # Ensure vector store directory exists
            os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
            
            # Download files
            try:
                self.api.hf_hub_download(
                    repo_id=self.dataset_name,
                    filename="vector_store/index.faiss",
                    repo_type="dataset",
                    local_dir=VECTOR_STORE_PATH
                )
                
                self.api.hf_hub_download(
                    repo_id=self.dataset_name,
                    filename="vector_store/index.pkl",
                    repo_type="dataset",
                    local_dir=VECTOR_STORE_PATH
                )
                
                return True, "Vector store downloaded successfully"
            
            except Exception as download_error:
                return False, f"Failed to download vector store: {str(download_error)}"
            
        except Exception as e:
            return False, f"Error in download_vector_store: {str(e)}"

    def save_chat_history(self, conversation_id: str, messages: List[Dict[str, str]]) -> Tuple[bool, str]:
        """
        Save chat history to the dataset
        
        Args:
            conversation_id: Unique conversation identifier
            messages: List of message dictionaries with 'role' and 'content'
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Check if chat_history directory exists
            try:
                self.api.list_repo_files(
                    repo_id=self.dataset_name,
                    repo_type="dataset",
                    path="chat_history"
                )
            except Exception:
                # Create directory if it doesn't exist
                with tempfile.NamedTemporaryFile(delete=False) as temp:
                    self.api.upload_file(
                        path_or_fileobj=temp.name,
                        path_in_repo="chat_history/.gitkeep",
                        repo_id=self.dataset_name,
                        repo_type="dataset"
                    )
                    os.unlink(temp.name)

            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_history/{conversation_id}_{timestamp}.json"
            
            # Prepare data for saving with additional metadata
            chat_data = {
                "conversation_id": conversation_id,
                "timestamp": timestamp,
                "messages": messages,
                "metadata": {
                    "saved_at": datetime.now().isoformat(),
                    "messages_count": len(messages)
                }
            }
            
            # Use temporary file for safe writing with explicit encoding
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False, encoding="utf-8") as temp:
                json.dump(chat_data, temp, ensure_ascii=False, indent=2)
                temp.flush()  # Ensure all data is written
                temp_name = temp.name

            try:
                # Upload to Hugging Face Hub with explicit error handling
                self.api.upload_file(
                    path_or_fileobj=temp_name,
                    path_in_repo=filename,
                    repo_id=self.dataset_name,
                    repo_type="dataset"
                )
            except Exception as upload_error:
                return False, f"Failed to upload chat history: {str(upload_error)}"
            finally:
                # Clean up temporary file
                if os.path.exists(temp_name):
                    os.unlink(temp_name)
            
            print(f"Successfully saved chat history: {filename}")
            return True, f"Chat history saved successfully as {filename}"
        
        except Exception as e:
            print(f"Error in save_chat_history: {str(e)}")
            return False, f"Failed to save chat history: {str(e)}"

    def get_chat_history(self, conversation_id: Optional[str] = None) -> Tuple[bool, Any]:
        """
        Get chat history from the dataset
        
        Args:
            conversation_id: Conversation identifier (if None, returns all chats)
            
        Returns:
            (success, chat history or error message)
        """
        try:
            # Get list of files in chat_history directory
            files = self.api.list_repo_files(
                repo_id=self.dataset_name,
                repo_type="dataset",
                path="chat_history"
            )
            
            # Filter files by conversation_id if specified
            if conversation_id:
                files = [f for f in files if f.startswith(f"chat_history/{conversation_id}_")]
            
            # If no files found, return empty list
            if not files or all(f.endswith(".gitkeep") for f in files):
                return True, []
            
            # Create temporary directory for downloading files
            with tempfile.TemporaryDirectory() as temp_dir:
                chat_histories = []
                
                for file in files:
                    if file.endswith(".gitkeep"):
                        continue
                        
                    # Download file
                    local_file = self.api.hf_hub_download(
                        repo_id=self.dataset_name,
                        filename=file,
                        repo_type="dataset",
                        local_dir=temp_dir
                    )
                    
                    # Read file content
                    with open(local_file, "r", encoding="utf-8") as f:
                        chat_data = json.load(f)
                        chat_histories.append(chat_data)
                
                # Sort by timestamp
                chat_histories.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                
                return True, chat_histories
        except Exception as e:
            return False, f"Error getting chat history: {str(e)}"

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

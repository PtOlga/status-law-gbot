"""
Модуль для управления датасетом на Hugging Face Hub
"""

import os
import json
import tempfile
from typing import Tuple, List, Dict, Any, Optional
from datetime import datetime
from huggingface_hub import HfApi, HfFolder
from config.settings import VECTOR_STORE_PATH

class DatasetManager:
    def __init__(self, dataset_name="Rulga/status-law-knowledge-base", token: Optional[str] = None):
        """
        Инициализация менеджера датасетов
        
        Args:
            dataset_name: Имя датасета на Hugging Face Hub
            token: Токен доступа к Hugging Face Hub (если не задан, берется из ~/.huggingface/token)
        """
        self.token = token if token else HfFolder.get_token()
        if not self.token:
            raise ValueError("Не найден токен Hugging Face. Установите переменную окружения HUGGINGFACE_TOKEN")
            
        self.api = HfApi(token=self.token)
        self.dataset_name = dataset_name
        
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
        Инициализация структуры датасета на Hugging Face
        
        Returns:
            (успех, сообщение)
        """
        try:
            # Проверяем существование репозитория
            try:
                self.api.repo_info(repo_id=self.dataset_name, repo_type="dataset")
            except Exception:
                # Если репозиторий не существует, создаем его
                self.api.create_repo(repo_id=self.dataset_name, repo_type="dataset", private=True)
            
            # Создаем пустые .gitkeep файлы для поддержания структуры
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
            
            return True, "Структура датасета успешно создана"
        except Exception as e:
            return False, f"Ошибка при создании структуры датасета: {str(e)}"

    def upload_vector_store(self) -> Tuple[bool, str]:
        """
        Загрузка векторного хранилища в датасет
        
        Returns:
            (успех, сообщение)
        """
        try:
            # Проверяем наличие файлов
            index_path = os.path.join(VECTOR_STORE_PATH, "index.faiss")
            config_path = os.path.join(VECTOR_STORE_PATH, "index.pkl")
            
            if not os.path.exists(index_path):
                return False, f"Файл векторного хранилища не найден: {index_path}"
                
            if not os.path.exists(config_path):
                return False, f"Файл конфигурации не найден: {config_path}"

            # Загружаем файлы
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Сначала сохраняем старые версии файлов в архивной директории, если они существуют
            try:
                # Проверяем наличие старых файлов
                self.api.hf_hub_download(
                    repo_id=self.dataset_name,
                    filename="vector_store/index.faiss",
                    repo_type="dataset"
                )
                
                # Если файл существует, создаем архивную копию
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
                # Если файлов нет, создаем директорию для архива
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
            
            # Загружаем текущие файлы
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
            
            # Обновляем метаданные о последнем обновлении
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
            
            return True, "Векторное хранилище успешно загружено"
        except Exception as e:
            return False, f"Ошибка при загрузке векторного хранилища: {str(e)}"

    def download_vector_store(self, force: bool = False) -> Tuple[bool, str]:
        """
        Загрузка векторного хранилища из датасета
        
        Args:
            force: Принудительная загрузка даже если локальные файлы существуют
            
        Returns:
            (успех, сообщение)
        """
        try:
            # Создаем директорию если её нет
            os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
            
            # Проверяем наличие локальных файлов
            index_path = os.path.join(VECTOR_STORE_PATH, "index.faiss")
            config_path = os.path.join(VECTOR_STORE_PATH, "index.pkl")
            
            if not force and os.path.exists(index_path) and os.path.exists(config_path):
                return True, "Локальные файлы векторного хранилища уже существуют"
            
            # Загружаем файлы
            try:
                # Пробуем получить метаданные для проверки существования файлов
                self.api.hf_hub_download(
                    repo_id=self.dataset_name,
                    filename="vector_store/metadata.json",
                    repo_type="dataset",
                    local_dir=VECTOR_STORE_PATH
                )
                
                # Загружаем файлы векторного хранилища
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
                
                return True, "Векторное хранилище успешно загружено"
            except Exception as download_error:
                return False, f"Ошибка при загрузке файлов: {str(download_error)}"
        except Exception as e:
            return False, f"Ошибка при загрузке векторного хранилища: {str(e)}"

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
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_history/{conversation_id}_{timestamp}.json"
            
            # Prepare data for saving
            chat_data = {
                "conversation_id": conversation_id,
                "timestamp": timestamp,
                "messages": messages
            }
            
            # Use temporary file for safe writing
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False, encoding="utf-8") as temp:
                json.dump(chat_data, temp, ensure_ascii=False, indent=2)
                temp_name = temp.name

            # Upload to Hugging Face Hub
            self.api.upload_file(
                path_or_fileobj=temp_name,
                path_in_repo=filename,
                repo_id=self.dataset_name,
                repo_type="dataset"
            )
            
            # Clean up temporary file
            os.unlink(temp_name)
            
            return True, "Chat history saved successfully"
        
        except Exception as e:
            return False, f"Failed to save chat history: {str(e)}"

    def get_chat_history(self, conversation_id: Optional[str] = None) -> Tuple[bool, Any]:
        """
        Получение истории чатов из датасета
        
        Args:
            conversation_id: Идентификатор беседы (если None, возвращает все чаты)
            
        Returns:
            (успех, история чатов или сообщение об ошибке)
        """
        try:
            # Получаем список файлов в директории chat_history
            files = self.api.list_repo_files(
                repo_id=self.dataset_name,
                repo_type="dataset",
                path="chat_history"
            )
            
            # Фильтруем файлы по conversation_id, если он указан
            if conversation_id:
                files = [f for f in files if f.startswith(f"chat_history/{conversation_id}_")]
            
            # Если файлов нет, возвращаем пустой список
            if not files or all(f.endswith(".gitkeep") for f in files):
                return True, []
            
            # Создаем временную директорию для загрузки файлов
            with tempfile.TemporaryDirectory() as temp_dir:
                chat_histories = []
                
                for file in files:
                    if file.endswith(".gitkeep"):
                        continue
                        
                    # Загружаем файл
                    local_file = self.api.hf_hub_download(
                        repo_id=self.dataset_name,
                        filename=file,
                        repo_type="dataset",
                        local_dir=temp_dir
                    )
                    
                    # Читаем содержимое файла
                    with open(local_file, "r", encoding="utf-8") as f:
                        chat_data = json.load(f)
                        chat_histories.append(chat_data)
                
                # Сортируем по временной метке
                chat_histories.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                
                return True, chat_histories
        except Exception as e:
            return False, f"Ошибка при получении истории чатов: {str(e)}"

    def upload_document(self, file_path: str, document_id: Optional[str] = None) -> Tuple[bool, str]:
        """
        Загрузка документа в датасет
        
        Args:
            file_path: Путь к файлу документа
            document_id: Идентификатор документа (если None, используется имя файла)
            
        Returns:
            (успех, сообщение)
        """
        try:
            if not os.path.exists(file_path):
                return False, f"Файл не найден: {file_path}"
                
            # Если document_id не указан, используем имя файла
            if document_id is None:
                document_id = os.path.basename(file_path)
                
            # Добавляем временную метку к имени файла
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"documents/{document_id}_{timestamp}{os.path.splitext(file_path)[1]}"
            
            # Загружаем файл
            self.api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=filename,
                repo_id=self.dataset_name,
                repo_type="dataset"
            )
            
            return True, f"Документ успешно загружен: {filename}"
        except Exception as e:
            return False, f"Ошибка при загрузке документа: {str(e)}"

def test_dataset_connection(token: Optional[str] = None) -> Tuple[bool, str]:
    """
    Тестовая функция для проверки подключения к датасету
    
    Args:
        token: Токен доступа к Hugging Face Hub
        
    Returns:
        (успех, сообщение)
    """
    try:
        manager = DatasetManager(token=token)
        success, message = manager.init_dataset_structure()
        
        if not success:
            return False, message
            
        print(f"Тест инициализации: {message}")
        
        return True, "Подключение к датасету работает"
    except Exception as e:
        return False, f"Ошибка подключения к датасету: {str(e)}"

if __name__ == "__main__":
    # Тестируем подключение
    success, message = test_dataset_connection()
    print(message)

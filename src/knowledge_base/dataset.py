import os
import json
from datetime import datetime
from huggingface_hub import HfApi
from config.settings import VECTOR_STORE_PATH

class DatasetManager:
    def __init__(self, dataset_name="Rulga/status-law-knowledge-base"):
        self.api = HfApi()
        self.dataset_name = dataset_name
        
    def init_dataset_structure(self):
        """Инициализация структуры датасета на Hugging Face"""
        try:
            # Создаем пустые .gitkeep файлы для поддержания структуры
            self.api.upload_file(
                path_or_fileobj=b"",
                path_in_repo="vector_store/.gitkeep",
                repo_id=self.dataset_name,
                repo_type="dataset"
            )
            
            self.api.upload_file(
                path_or_fileobj=b"",
                path_in_repo="chat_history/.gitkeep",
                repo_id=self.dataset_name,
                repo_type="dataset"
            )
            
            return True, "Структура датасета успешно создана"
        except Exception as e:
            return False, f"Ошибка при создании структуры датасета: {str(e)}"

    def upload_vector_store(self):
        """Загрузка векторного хранилища в датасет"""
        try:
            # Проверяем наличие файлов
            index_path = os.path.join(VECTOR_STORE_PATH, "index.faiss")
            config_path = os.path.join(VECTOR_STORE_PATH, "index.pkl")
            
            if not (os.path.exists(index_path) and os.path.exists(config_path)):
                return False, "Файлы векторного хранилища не найдены"

            # Загружаем файлы
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
            
            return True, "Векторное хранилище успешно загружено"
        except Exception as e:
            return False, f"Ошибка при загрузке векторного хранилища: {str(e)}"

    def download_vector_store(self):
        """Загрузка векторного хранилища из датасета"""
        try:
            # Создаем директорию если её нет
            os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
            
            # Загружаем файлы
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
        except Exception as e:
            return False, f"Ошибка при загрузке векторного хранилища: {str(e)}"

    def save_chat_history(self, conversation_id, messages):
        """Сохранение истории чата в датасет"""
        try:
            # Формируем имя файла с временной меткой
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_history/{conversation_id}_{timestamp}.json"
            
            # Создаем временный файл
            chat_data = {
                "conversation_id": conversation_id,
                "timestamp": timestamp,
                "messages": messages
            }
            
            temp_file = f"temp_{conversation_id}.json"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(chat_data, f, ensure_ascii=False, indent=2)
            
            # Загружаем файл в датасет
            self.api.upload_file(
                path_or_fileobj=temp_file,
                path_in_repo=filename,
                repo_id=self.dataset_name,
                repo_type="dataset"
            )
            
            # Удаляем временный файл
            os.remove(temp_file)
            
            return True, "История чата сохранена"
        except Exception as e:
            return False, f"Ошибка при сохранении истории чата: {str(e)}"

def test_dataset_connection():
    """Тестовая функция для проверки подключения к датасету"""
    try:
        manager = DatasetManager()
        success, message = manager.init_dataset_structure()
        print(f"Тест инициализации: {message}")
        
        return True, "Подключение к датасету работает"
    except Exception as e:
        return False, f"Ошибка подключения к датасету: {str(e)}"

if __name__ == "__main__":
    # Тестируем подключение
    success, message = test_dataset_connection()
    print(message)
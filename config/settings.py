import os
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Пути к директориям
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "data", "vector_store")

# Настройки моделей
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_MODEL = "HuggingFaceH4/zephyr-7b-beta"  # Модель по умолчанию из шаблона
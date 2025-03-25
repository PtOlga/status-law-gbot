import os
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Отладочная информация
print("Текущая директория:", os.getcwd())
print("Путь к .env:", os.path.join(os.getcwd(), '.env'))
print("Все переменные окружения:", {k: v for k, v in os.environ.items() if 'TOKEN' in k})

# Пути к директориям
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "data", "vector_store")

# Настройки моделей
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_MODEL = "HuggingFaceH4/zephyr-7b-beta"

# API токены
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN не найден в переменных окружения")

# Настройки запросов
USER_AGENT = "Status-Law-Assistant/1.0"

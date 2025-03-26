import os
from dotenv import load_dotenv

# Отладочная информация
print("Текущая директория:", os.getcwd())
env_path = os.path.join(os.getcwd(), '.env')
print("Путь к .env:", env_path)
print("Файл .env существует:", os.path.exists(env_path))

if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        print("Содержимое .env файла:", f.read())

# Загрузка переменных окружения
load_dotenv(verbose=True)

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

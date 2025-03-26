import os
from dotenv import load_dotenv

# Debug information
print("Current directory:", os.getcwd())
env_path = os.path.join(os.getcwd(), '.env')
print("Path to .env:", env_path)
print(".env file exists:", os.path.exists(env_path))

if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        print("Contents of .env file:", f.read())

# Load environment variables
load_dotenv(verbose=True)

# Directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "data", "vector_store")

# Add missing paths for training models
MODEL_PATH = os.path.join(BASE_DIR, "models")
TRAINING_OUTPUT_DIR = os.path.join(BASE_DIR, "models", "trained")

# Create directories if they don't exist
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(TRAINING_OUTPUT_DIR, exist_ok=True)

# Model settings
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_MODEL = "HuggingFaceH4/zephyr-7b-beta"

# API tokens
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")

# Request settings
USER_AGENT = "Status-Law-Assistant/1.0"
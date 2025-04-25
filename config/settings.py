import os
import tempfile
import logging
from huggingface_hub import HfApi
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get token with fallback
HF_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")

# Validate token format
if not HF_TOKEN.startswith('hf_'):
    raise ValueError("Invalid Hugging Face token format")

print(f"Token loaded successfully: {HF_TOKEN[:5]}...")

# API Configuration
API_CONFIG = {
    "inference_endpoint": "https://api-inference.huggingface.co",
    "token": HF_TOKEN,
    "is_paid_tier": False,  # Принудительно устанавливаем бесплатный режим
    "timeout": 15,
    "max_retries": 1,
    "headers": {
        "X-Use-Cache": "true",  # Включаем кэширование для бесплатного тарифа
        "Content-Type": "application/json",
        "Authorization": f"Bearer {HF_TOKEN}"  
    }
}

def check_account_type():
    """
    Simplified account check for free tier
    Returns:
        tuple: (is_pro: bool, account_type: str)
    """
    return False, "free"

# Устанавливаем базовые настройки для free tier
IS_PRO_ACCOUNT, ACCOUNT_TYPE = False, "free"
DEFAULT_MODEL = "zephyr-7b"  # Устанавливаем дефолтную бесплатную модель

# Dataset configuration
DATASET_ID = "Rulga/status-law-knowledge-base"

# Dataset paths
DATASET_CHAT_HISTORY_PATH = "chat_history"
DATASET_VECTOR_STORE_PATH = "vector_store"
DATASET_FINE_TUNED_PATH = "fine_tuned_models"
DATASET_ANNOTATIONS_PATH = "annotations"
DATASET_ERROR_LOGS_PATH = "error_logs"
DATASET_PREFERENCES_PATH = "preferences/user_preferences.json"
# Adding training data paths
DATASET_TRAINING_DATA_PATH = "training_data"
DATASET_TRAINING_LOGS_PATH = "training_logs"

# Temporary storage (using system temp directory)
TEMP_DIR = tempfile.gettempdir()
TEMP_ROOT = os.path.join(TEMP_DIR, "status_law_kb")
CHAT_HISTORY_PATH = os.path.join(TEMP_ROOT, "chat_history")
VECTOR_STORE_PATH = os.path.join(TEMP_ROOT, "vector_store")
FINE_TUNED_PATH = os.path.join(TEMP_ROOT, "fine_tuned_models")
MODELS_REGISTRY_PATH = os.path.join(TEMP_ROOT, "models_registry.json")

# Create temporary directories
for path in [CHAT_HISTORY_PATH, VECTOR_STORE_PATH, FINE_TUNED_PATH]:
    os.makedirs(path, exist_ok=True)

# Paths configuration
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
TRAINING_OUTPUT_DIR = os.path.join(CHAT_HISTORY_PATH, FINE_TUNED_PATH)

# Create necessary directories if they don't exist
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(TRAINING_OUTPUT_DIR, exist_ok=True)

# Models configuration with detailed information
MODELS = {
    "zephyr-7b": {
        "id": "HuggingFaceH4/zephyr-7b-beta",
        "name": "Zephyr 7B",
        "description": "A state-of-the-art 7B parameter language model",
        "type": "base",
        "parameters": {
            "max_length": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
        },
        "training": {
            "base_model_path": "HuggingFaceH4/zephyr-7b-beta",
            "fine_tuned_path": os.path.join(TRAINING_OUTPUT_DIR, "zephyr-7b-beta-tuned"),
            "lora_config": {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
            }
        },
        "details": {
            "full_name": "HuggingFaceH4 Zephyr 7B Beta",
            "capabilities": [
                "High performance on instruction-following tasks",
                "Good response accuracy",
                "Advanced reasoning capabilities",
                "Excellent text generation quality"
            ],
            "limitations": [
                "May require paid API for usage",
                "Limited support for languages other than English",
                "Less optimization for legal topics compared to specialized models"
            ],
            "use_cases": [
                "Complex legal reasoning",
                "Case analysis",
                "Legal research",
                "Structured legal text generation"
            ],
            "documentation": "https://huggingface.co/HuggingFaceH4/zephyr-7b-beta"
        }
    },
    "llama-7b": {
        "id": "meta-llama/Llama-2-7b-chat-hf",
        "name": "Llama 2 7B Chat",
        "description": "Meta's Llama 2 7B model optimized for chat",
        "type": "base",
        "parameters": {
            "max_length": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
        },
        "training": {
            "base_model_path": os.path.join(MODEL_PATH, "llama-2-7b-chat"),
            "fine_tuned_path": os.path.join(TRAINING_OUTPUT_DIR, "llama-2-7b-chat-tuned"),
            "lora_config": {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
            }
        },
        "details": {
            "full_name": "Meta Llama 2 7B Chat",
            "capabilities": [
                "Multilingual support ",
                "Good performance on legal texts",
                "Free model with open license",
                "Can run on computers with 16GB+ RAM"
            ],
            "limitations": [
                "Limited knowledge of specific legal terminology",
                "May provide incorrect answers to complex legal questions",
                "Knowledge is limited to training data"
            ],
            "use_cases": [
                "Legal document analysis",
                "Answering general legal questions",
                "Searching through legal knowledge base",
                "Assistance in document drafting"
            ],
            "documentation": "https://huggingface.co/meta-llama/Llama-2-7b-chat-hf"
        }
    },
    "mistral-7b": {
        "id": "mistralai/Mistral-7B-Instruct-v0.2",
        "name": "Mistral 7B Instruct",
        "description": "Mistral's 7B instruction-tuned model with better multilingual support",
        "type": "base",
        "parameters": {
            "max_length": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
        },
        "training": {
            "base_model_path": "mistralai/Mistral-7B-Instruct-v0.2",
            "fine_tuned_path": os.path.join(TRAINING_OUTPUT_DIR, "mistral-7b-instruct-tuned"),
            "lora_config": {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
            }
        },
        "details": {
            "full_name": "Mistral 7B Instruct v0.2",
            "capabilities": [
                "Strong multilingual support",
                "Superior instruction following ability",
                "Fast inference speed",
                "Excellent reasoning capabilities",
                "Free for commercial use"
            ],
            "limitations": [
                "May have limited knowledge of specialized legal terminology",
                "Less exposure to legal domain than specialized models",
                "Knowledge cutoff before latest legal developments"
            ],
            "use_cases": [
                "Multilingual legal assistance",
                "Cross-border legal questions",
                "Clear explanations of complex legal topics",
                "Serving international clients in their native language"
            ],
            "documentation": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2"
        }
    },
    "xglm-7.5b": {
        "id": "facebook/xglm-7.5B",
        "name": "XGLM 7.5B",
        "description": "Meta's multilingual model designed for cross-lingual generation",
        "type": "base",
        "parameters": {
            "max_length": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
        },
        "training": {
            "base_model_path": "facebook/xglm-7.5B",
            "fine_tuned_path": os.path.join(TRAINING_OUTPUT_DIR, "xglm-7.5b-tuned"),
            "lora_config": {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
            }
        },
        "details": {
            "full_name": "Meta XGLM 7.5B",
            "capabilities": [
                "Specialized for multilingual generation",
                "Support for 30+ languages",
                "Strong cross-lingual transfer abilities",
                "Consistent performance across diverse languages"
            ],
            "limitations": [
                "Less instruction-tuned than dedicated chat models",
                "May require more specific prompting",
                "Not specifically optimized for legal domain",
                "Slightly larger model requiring more GPU memory"
            ],
            "use_cases": [
                "International legal assistance in native languages",
                "Complex multilingual documentation",
                "Serving clients from diverse linguistic backgrounds",
                "Translation and summarization of legal concepts across languages"
            ],
            "documentation": "https://huggingface.co/facebook/xglm-7.5B"
        }
    }
}

# Update MODELS configuration
for model in MODELS.values():
    model["endpoint"] = API_CONFIG["inference_endpoint"]

# Default model
DEFAULT_MODEL = "zephyr-7b"  # Changed from "llama-7b" to "zephyr-7b"
ACTIVE_MODEL = MODELS[DEFAULT_MODEL]

# Embedding model for vector store
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# Request settings
USER_AGENT = "Status-Law-Assistant/1.0"

# Add these constants to settings.py
RATING_FIELDS = {
    "accuracy": "Точность ответа",
    "completeness": "Полнота информации",
    "relevance": "Релевантность ответу",
    "clarity": "Ясность изложения",
    "legal_correctness": "Юридическая корректность"
}

CHAT_HISTORY_SCHEMA = {
    "conversation_id": str,
    "timestamp": str,  # ISO format
    "history": [
        {
            "role": str,  # "user" or "assistant"
            "content": str,
            "timestamp": str  # ISO format
        }
    ]
}

ANNOTATION_SCHEMA = {
    "conversation_id": str,
    "timestamp": str,
    "question": str,
    "original_answer": str,
    "improved_answer": str,
    "ratings": {field: int for field in RATING_FIELDS},  # все оценки 1-5
    "notes": str
}

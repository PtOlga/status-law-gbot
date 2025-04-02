import os

# API tokens
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")

# API Configuration
API_CONFIG = {
    "inference_endpoint": os.getenv("HF_INFERENCE_ENDPOINT", "https://api-inference.huggingface.co"),
    "token": HF_TOKEN,
    "is_paid_tier": True,  # или False в зависимости от вашего плана
    "timeout": 30,
    "headers": {
        "X-Use-Cache": "false",
        "Content-Type": "application/json"
    }
}

# Dataset configuration
DATASET_ID = "Rulga/status-law-knowledge-base"
CHAT_HISTORY_PATH = "chat_history"
VECTOR_STORE_PATH = "vector_store"

# Paths configuration
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
TRAINING_OUTPUT_DIR = os.path.join(MODEL_PATH, "fine_tuned")

# Create necessary directories if they don't exist
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(TRAINING_OUTPUT_DIR, exist_ok=True)
MODELS_REGISTRY_PATH = os.path.join(MODEL_PATH, "registry.json")

# Models configuration
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
        }
    }
}

# Update MODELS configuration
for model in MODELS.values():
    model["endpoint"] = API_CONFIG["inference_endpoint"]

# Default model
DEFAULT_MODEL = "llama-7b"  # Changed from "zephyr-7b" to "llama-7b"
ACTIVE_MODEL = MODELS[DEFAULT_MODEL]

# Embedding model for vector store
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# Request settings
USER_AGENT = "Status-Law-Assistant/1.0"

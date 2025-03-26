import os

# API tokens
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")

# Paths configuration
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
TRAINING_OUTPUT_DIR = os.path.join(MODEL_PATH, "fine_tuned")
VECTOR_STORE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_store")

# Model configuration
MODEL_CONFIG = {
    "id": "HuggingFaceH4/zephyr-7b-beta",
    "name": "Zephyr 7B",
    "description": "A state-of-the-art 7B parameter language model",
    "type": "base",  # base/fine-tuned
    "parameters": {
        "max_length": 2048,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
    },
    "training": {
        "base_model_path": os.path.join(MODEL_PATH, "zephyr-7b-beta"),
        "fine_tuned_path": os.path.join(TRAINING_OUTPUT_DIR, "zephyr-7b-beta-tuned"),
        "lora_config": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
        }
    }
}

# Embedding model for vector store
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Request settings
USER_AGENT = "Status-Law-Assistant/1.0"

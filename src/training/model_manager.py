"""
Module for managing models and their versions
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import logging
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from config.settings import MODEL_PATH, MODELS_REGISTRY_PATH, MODEL_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_model(
    version: Optional[str] = None,
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Convenient function to get model and tokenizer
    
    Args:
        version: Model version (if None, loads base version)
        device: Device for loading model
        
    Returns:
        (model, tokenizer, model_info)
    """
    manager = ModelManager()
    
    # Use base model if version is None
    model_path = MODEL_CONFIG["training"]["fine_tuned_path"] if version else MODEL_CONFIG["training"]["base_model_path"]
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else None
        )
        
        return model, tokenizer, MODEL_CONFIG
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise ValueError(f"Failed to load model: {str(e)}")

if __name__ == "__main__":
    # Пример использования
    manager = ModelManager()
    
    # Регистрация базовой модели
    success, message = manager.register_model(
        model_id="saiga",
        version="7b",
        source="hf://IlyaGusev/saiga_7b_lora",
        description="Базовая модель Saiga 7B с LoRA адаптерами",
        is_active=True
    )
    print(message)
    
    # Вывод списка моделей
    models = manager.list_models()
    print(f"В реестре {len(models)} моделей:")
    for model in models:
        print(f" - {model['model_id']} v{model['version']}: {model['description']}")

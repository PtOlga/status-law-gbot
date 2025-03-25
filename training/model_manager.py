"""
Модуль для управления моделями и их версиями
"""

import os
import json
import shutil
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import logging
from huggingface_hub import HfApi, snapshot_download, hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from config.settings import MODEL_PATH, MODELS_REGISTRY_PATH

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, registry_path: Optional[str] = None):
        """
        Инициализация менеджера моделей
        
        Args:
            registry_path: Путь к реестру моделей
        """
        self.registry_path = registry_path or MODELS_REGISTRY_PATH
        self.models_dir = MODEL_PATH
        
        # Создаем директории, если их нет
        os.makedirs(self.registry_path, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Путь к файлу реестра
        self.registry_file = os.path.join(self.registry_path, "models_registry.json")
        
        # Загружаем реестр или создаем новый
        self.load_registry()
    
    def load_registry(self):
        """
        Загрузка реестра моделей
        """
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    self.registry = json.load(f)
            except Exception as e:
                logger.error(f"Ошибка загрузки реестра моделей: {str(e)}")
                self.registry = {"models": []}
        else:
            self.registry = {"models": []}
    
    def save_registry(self):
        """
        Сохранение реестра моделей
        """
        try:
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump(self.registry, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения реестра моделей: {str(e)}")
    
    def register_model(
        self,
        model_id: str,
        version: str,
        source: str,
        description: str = "",
        metrics: Optional[Dict[str, Any]] = None,
        is_active: bool = False
    ) -> Tuple[bool, str]:
        """
        Регистрация модели в реестре
        
        Args:
            model_id: Идентификатор модели (например, 'saiga_7b_lora')
            version: Версия модели
            source: Источник модели (например, URL или локальный путь)
            description: Описание модели
            metrics: Метрики качества модели
            is_active: Флаг активности модели
            
        Returns:
            (успех, сообщение)
        """
        try:
            # Создаем запись о модели
            model_entry = {
                "model_id": model_id,
                "version": version,
                "source": source,
                "description": description,
                "metrics": metrics or {},
                "is_active": is_active,
                "registration_date": datetime.now().isoformat(),
                "local_path": os.path.join(self.models_dir, f"{model_id}_{version}")
            }
            
            # Проверяем, есть ли уже такая модель в реестре
            for i, model in enumerate(self.registry["models"]):
                if model["model_id"] == model_id and model["version"] == version:
                    # Обновляем существующую запись
                    self.registry["models"][i] = model_entry
                    self.save_registry()
                    return True, f"Мод
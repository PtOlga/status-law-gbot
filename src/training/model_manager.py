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
                    return True, f"Модель {model_id} версии {version} обновлена в реестре"
            
            # Если модель новая, добавляем ее в реестр
            self.registry["models"].append(model_entry)
            
            # Если модель отмечена как активная, деактивируем все другие модели с тем же model_id
            if is_active:
                for i, model in enumerate(self.registry["models"]):
                    if model["model_id"] == model_id and model["version"] != version:
                        self.registry["models"][i]["is_active"] = False
            
            self.save_registry()
            return True, f"Модель {model_id} версии {version} добавлена в реестр"
        except Exception as e:
            return False, f"Ошибка при регистрации модели: {str(e)}"
    
    def download_model(
        self, 
        model_id: str,
        version: str,
        token: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Загрузка модели из Hugging Face Hub
        
        Args:
            model_id: Идентификатор модели
            version: Версия модели
            token: Токен доступа к Hugging Face Hub
            
        Returns:
            (успех, сообщение)
        """
        try:
            # Находим модель в реестре
            model_entry = None
            for model in self.registry["models"]:
                if model["model_id"] == model_id and model["version"] == version:
                    model_entry = model
                    break
            
            if model_entry is None:
                return False, f"Модель {model_id} версии {version} не найдена в реестре"
            
            # Проверяем, что источник - это репозиторий Hugging Face
            if not model_entry["source"].startswith("hf://"):
                return False, "Источник модели не является репозиторием Hugging Face"
            
            # Извлекаем имя репозитория
            repo_id = model_entry["source"][5:]
            
            # Путь для сохранения модели
            local_path = model_entry["local_path"]
            
            # Проверяем, существует ли уже директория с моделью
            if os.path.exists(local_path):
                # Если директория существует, проверяем наличие файлов модели
                if os.path.exists(os.path.join(local_path, "pytorch_model.bin")) or \
                   os.path.exists(os.path.join(local_path, "adapter_model.bin")):
                    return True, f"Модель {model_id} версии {version} уже загружена"
            else:
                # Создаем директорию для модели
                os.makedirs(local_path, exist_ok=True)
            
            # Загружаем модель
            logger.info(f"Загрузка модели {repo_id} в {local_path}...")
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_path,
                token=token
            )
            
            return True, f"Модель {model_id} версии {version} успешно загружена"
        except Exception as e:
            return False, f"Ошибка при загрузке модели: {str(e)}"
    
    def get_active_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Получение активной версии модели
        
        Args:
            model_id: Идентификатор модели
            
        Returns:
            Словарь с информацией о модели или None, если модель не найдена
        """
        for model in self.registry["models"]:
            if model["model_id"] == model_id and model.get("is_active", False):
                return model
        return None
    
    def set_active_model(self, model_id: str, version: str) -> Tuple[bool, str]:
        """
        Установка активной версии модели
        
        Args:
            model_id: Идентификатор модели
            version: Версия модели
            
        Returns:
            (успех, сообщение)
        """
        try:
            # Проверяем, есть ли модель в реестре
            model_found = False
            for i, model in enumerate(self.registry["models"]):
                if model["model_id"] == model_id:
                    if model["version"] == version:
                        model_found = True
                        self.registry["models"][i]["is_active"] = True
                    else:
                        self.registry["models"][i]["is_active"] = False
            
            if not model_found:
                return False, f"Модель {model_id} версии {version} не найдена в реестре"
            
            self.save_registry()
            return True, f"Модель {model_id} версии {version} установлена как активная"
        except Exception as e:
            return False, f"Ошибка при установке активной модели: {str(e)}"
    
    def load_model(
        self, 
        model_id: str,
        version: Optional[str] = None,
        device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    ) -> Tuple[bool, Any, Any, str]:
        """
        Загрузка модели и токенизатора
        
        Args:
            model_id: Идентификатор модели
            version: Версия модели (если None, загружается активная версия)
            device: Устройство для загрузки модели
            
        Returns:
            (успех, модель, токенизатор, сообщение)
        """
        try:
            # Определяем версию модели
            if version is None:
                model_entry = self.get_active_model(model_id)
                if model_entry is None:
                    return False, None, None, f"Активная версия модели {model_id} не найдена"
            else:
                model_entry = None
                for model in self.registry["models"]:
                    if model["model_id"] == model_id and model["version"] == version:
                        model_entry = model
                        break
                
                if model_entry is None:
                    return False, None, None, f"Модель {model_id} версии {version} не найдена в реестре"
            
            # Проверяем, загружена ли модель локально
            local_path = model_entry["local_path"]
            if not os.path.exists(local_path) or \
               (not os.path.exists(os.path.join(local_path, "pytorch_model.bin")) and \
                not os.path.exists(os.path.join(local_path, "adapter_model.bin"))):
                # Если модель не загружена, пытаемся загрузить её
                success, message = self.download_model(model_id, model_entry["version"])
                if not success:
                    return False, None, None, message
            
            # Загружаем токенизатор
            logger.info(f"Загрузка токенизатора из {local_path}...")
            tokenizer = AutoTokenizer.from_pretrained(
                local_path,
                trust_remote_code=True
            )
            
            # Загружаем модель
            logger.info(f"Загрузка модели из {local_path}...")
            model = AutoModelForCausalLM.from_pretrained(
                local_path,
                trust_remote_code=True,
                device_map="auto" if device == "cuda" else None
            )
            
            return True, model, tokenizer, f"Модель {model_id} версии {model_entry['version']} успешно загружена"
        except Exception as e:
            return False, None, None, f"Ошибка при загрузке модели: {str(e)}"
    
    def delete_model(self, model_id: str, version: str) -> Tuple[bool, str]:
        """
        Удаление модели из реестра и локального хранилища
        
        Args:
            model_id: Идентификатор модели
            version: Версия модели
            
        Returns:
            (успех, сообщение)
        """
        try:
            # Ищем модель в реестре
            model_entry = None
            model_index = -1
            for i, model in enumerate(self.registry["models"]):
                if model["model_id"] == model_id and model["version"] == version:
                    model_entry = model
                    model_index = i
                    break
            
            if model_entry is None:
                return False, f"Модель {model_id} версии {version} не найдена в реестре"
            
            # Проверяем, активна ли модель
            if model_entry.get("is_active", False):
                return False, "Нельзя удалить активную модель. Сначала установите другую модель как активную."
            
            # Удаляем директорию с моделью, если она существует
            local_path = model_entry["local_path"]
            if os.path.exists(local_path):
                shutil.rmtree(local_path)
            
            # Удаляем модель из реестра
            self.registry["models"].pop(model_index)
            self.save_registry()
            
            return True, f"Модель {model_id} версии {version} успешно удалена"
        except Exception as e:
            return False, f"Ошибка при удалении модели: {str(e)}"
    
    def list_models(self, model_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Получение списка моделей в реестре
        
        Args:
            model_id: Идентификатор модели для фильтрации (если None, возвращаются все модели)
            
        Returns:
            Список словарей с информацией о моделях
        """
        if model_id is None:
            return self.registry["models"]
        else:
            return [model for model in self.registry["models"] if model["model_id"] == model_id]
    
    def import_local_model(
        self,
        source_path: str,
        model_id: str,
        version: str,
        description: str = "",
        is_active: bool = False
    ) -> Tuple[bool, str]:
        """
        Импорт локальной модели в реестр
        
        Args:
            source_path: Путь к директории с моделью
            model_id: Идентификатор модели
            version: Версия модели
            description: Описание модели
            is_active: Флаг активности модели
            
        Returns:
            (успех, сообщение)
        """
        try:
            # Проверяем существование исходной директории
            if not os.path.exists(source_path):
                return False, f"Директория {source_path} не существует"
            
            # Проверяем, что это директория с моделью
            if not os.path.exists(os.path.join(source_path, "config.json")):
                return False, f"Директория {source_path} не содержит модель трансформера"
            
            # Создаем путь для модели в нашем хранилище
            target_path = os.path.join(self.models_dir, f"{model_id}_{version}")
            
            # Если директория уже существует, удаляем ее
            if os.path.exists(target_path):
                shutil.rmtree(target_path)
            
            # Копируем файлы модели
            shutil.copytree(source_path, target_path)
            
            # Регистрируем модель в реестре
            success, message = self.register_model(
                model_id=model_id,
                version=version,
                source=f"local://{source_path}",
                description=description,
                is_active=is_active
            )
            
            if not success:
                # Если регистрация не удалась, удаляем скопированные файлы
                shutil.rmtree(target_path)
                return False, message
            
            return True, f"Модель успешно импортирована: {model_id} версии {version}"
        except Exception as e:
            return False, f"Ошибка при импорте модели: {str(e)}"
    
    def export_model_metrics(self, output_file: str) -> Tuple[bool, str]:
        """
        Экспорт метрик всех моделей в JSON файл
        
        Args:
            output_file: Путь к выходному файлу
            
        Returns:
            (успех, сообщение)
        """
        try:
            # Создаем словарь с метриками для каждой модели
            metrics_data = {}
            
            for model in self.registry["models"]:
                model_key = f"{model['model_id']}_{model['version']}"
                metrics_data[model_key] = {
                    "model_id": model["model_id"],
                    "version": model["version"],
                    "is_active": model.get("is_active", False),
                    "registration_date": model.get("registration_date", ""),
                    "metrics": model.get("metrics", {})
                }
            
            # Сохраняем в файл
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(metrics_data, f, ensure_ascii=False, indent=2)
            
            return True, f"Метрики моделей успешно экспортированы в {output_file}"
        except Exception as e:
            return False, f"Ошибка при экспорте метрик: {str(e)}"
    
    def update_model_metrics(
        self,
        model_id: str,
        version: str,
        metrics: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Обновление метрик модели
        
        Args:
            model_id: Идентификатор модели
            version: Версия модели
            metrics: Словарь с метриками
            
        Returns:
            (успех, сообщение)
        """
        try:
            # Ищем модель в реестре
            model_found = False
            for i, model in enumerate(self.registry["models"]):
                if model["model_id"] == model_id and model["version"] == version:
                    # Обновляем метрики
                    self.registry["models"][i]["metrics"] = metrics
                    model_found = True
                    break
            
            if not model_found:
                return False, f"Модель {model_id} версии {version} не найдена в реестре"
            
            self.save_registry()
            return True, f"Метрики модели {model_id} версии {version} успешно обновлены"
        except Exception as e:
            return False, f"Ошибка при обновлении метрик: {str(e)}"

def get_model(
    model_id: str = "saiga",
    version: Optional[str] = None,
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Удобная функция для получения модели и токенизатора
    
    Args:
        model_id: Идентификатор модели
        version: Версия модели (если None, загружается активная версия)
        device: Устройство для загрузки модели
        
    Returns:
        (модель, токенизатор, информация о модели)
    """
    manager = ModelManager()
    success, model, tokenizer, message = manager.load_model(
        model_id=model_id,
        version=version,
        device=device
    )
    
    if not success:
        logger.error(message)
        raise ValueError(message)
    
    # Получаем информацию о загруженной модели
    if version is None:
        model_info = manager.get_active_model(model_id)
    else:
        for m in manager.list_models(model_id):
            if m["version"] == version:
                model_info = m
                break
        else:
            model_info = {}
    
    return model, tokenizer, model_info

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
"""
Модуль для дообучения языковой модели на основе собранных данных
"""

import os
import json
import tempfile
from typing import List, Dict, Any, Tuple, Optional
import logging
from huggingface_hub import HfApi, HfFolder
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from src.analytics.chat_analyzer import ChatAnalyzer
from config.settings import MODEL_PATH, TRAINING_OUTPUT_DIR

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FineTuner:
    def __init__(
        self,
        base_model_id: str = "IlyaGusev/saiga_7b_lora",
        output_dir: Optional[str] = None,
        device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    ):
        """
        Инициализация модуля для дообучения модели
        
        Args:
            base_model_id: Идентификатор базовой модели на Hugging Face Hub
            output_dir: Директория для сохранения результатов обучения
            device: Устройство для обучения ('cuda' или 'cpu')
        """
        self.base_model_id = base_model_id
        self.output_dir = output_dir or TRAINING_OUTPUT_DIR
        self.device = device
        self.tokenizer = None
        self.model = None
        self.chat_analyzer = ChatAnalyzer()
        
        # Создаем директорию для результатов, если её нет
        os.makedirs(self.output_dir, exist_ok=True)
    
    def prepare_training_data(self, output_file: Optional[str] = None) -> str:
        """
        Подготовка данных для обучения
        
        Args:
            output_file: Путь к выходному файлу (если None, создается временный файл)
            
        Returns:
            Путь к файлу с данными для обучения
        """
        if output_file is None:
            # Создаем временный файл для данных
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl")
            output_file = temp_file.name
            temp_file.close()
        
        # Экспортируем данные для обучения
        success, message = self.chat_analyzer.export_training_data(output_file)
        
        if not success:
            raise ValueError(f"Ошибка при подготовке данных: {message}")
        
        logger.info(message)
        return output_file
    
    def load_model_and_tokenizer(self):
        """
        Загрузка базовой модели и токенизатора
        """
        try:
            logger.info(f"Загрузка модели {self.base_model_id}...")
            
            # Загрузка токенизатора
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_id,
                trust_remote_code=True
            )
            
            # Специальные токены для диалогов
            special_tokens = {
                "pad_token": "<PAD>",
                "eos_token": "</s>",
                "bos_token": "<s>"
            }
            
            # Добавляем специальные токены, если их нет
            for token_name, token_value in special_tokens.items():
                if getattr(self.tokenizer, token_name) is None:
                    setattr(self.tokenizer, token_name, token_value)
            
            # Загрузка модели
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                trust_remote_code=True,
                device_map="auto" if self.device == "cuda" else None
            )
            
            logger.info("Модель и токенизатор успешно загружены")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {str(e)}")
            raise
    
    def setup_lora_config(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05
    ) -> LoraConfig:
        """
        Настройка конфигурации LoRA для эффективного дообучения
        
        Args:
            r: Ранг матриц LoRA
            lora_alpha: Альфа параметр LoRA
            lora_dropout: Вероятность dropout в LoRA слоях
            
        Returns:
            Конфигурация LoRA
        """
        # Создаем конфигурацию LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        
        return lora_config
    
    def prepare_model_for_training(self):
        """
        Подготовка модели к обучению с использованием LoRA
        """
        if self.model is None:
            self.load_model_and_tokenizer()
        
        # Настройка LoRA
        lora_config = self.setup_lora_config()
        
        # Применяем LoRA к модели
        self.model = get_peft_model(self.model, lora_config)
        
        # Вывод информации о параметрах
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Обучаемых параметров: {trainable_params:,} из {all_params:,} ({trainable_params/all_params:.2%})")
    
    def tokenize_dataset(self, dataset):
        """
        Токенизация датасета для обучения
        
        Args:
            dataset: Датасет для токенизации
            
        Returns:
            Токенизированный датасет
        """
        def tokenize_function(examples):
            # Форматируем диалоги в единую строку
            texts = []
            for dialog in examples["messages"]:
                text = ""
                for message in dialog:
                    if message["role"] == "user":
                        text += f"User: {message['content']}\n"
                    elif message["role"] == "assistant":
                        text += f"Assistant: {message['content']}\n"
                texts.append(text)
            
            # Токенизируем тексты
            tokenized = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=1024,
                return_tensors="pt"
            )
            
            return tokenized
        
        # Применяем функцию токенизации
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["messages"]
        )
        
        return tokenized_dataset
    
    def train(
        self,
        training_data_path: Optional[str] = None,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        logging_steps: int = 10,
        save_strategy: str = "epoch"
    ) -> Tuple[bool, str]:
        """
        Запуск процесса дообучения модели
        
        Args:
            training_data_path: Путь к данным для обучения (если None, данные будут подготовлены автоматически)
            num_train_epochs: Количество эпох обучения
            per_device_train_batch_size: Размер батча на устройство
            gradient_accumulation_steps: Количество шагов накопления градиента
            learning_rate: Скорость обучения
            logging_steps: Частота логирования
            save_strategy: Стратегия сохранения модели
            
        Returns:
            (успех, сообщение)
        """
        try:
            # Подготовка данных для обучения, если не указан путь
            if training_data_path is None:
                training_data_path = self.prepare_training_data()
                temp_data = True
            else:
                temp_data = False
            
            # Загрузка модели и токенизатора, если не загружены
            if self.model is None or self.tokenizer is None:
                self.load_model_and_tokenizer()
            
            # Подготовка модели для обучения
            self.prepare_model_for_training()
            
            # Загрузка датасета
            dataset = load_dataset("json", data_files=training_data_path, split="train")
            logger.info(f"Загружено {len(dataset)} примеров из {training_data_path}")
            
            # Токенизация датасета
            tokenized_dataset = self.tokenize_dataset(dataset)
            
            # Создание колатора данных
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            
            # Настройка аргументов обучения
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                weight_decay=0.01,
                warmup_ratio=0.1,
                logging_steps=logging_steps,
                save_strategy=save_strategy,
                save_total_limit=2,
                remove_unused_columns=False,
                push_to_hub=False,
                report_to="tensorboard",
                load_best_model_at_end=True
            )
            
            # Создание тренера
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer
            )
            
            # Запуск обучения
            logger.info("Начало обучения модели...")
            trainer.train()
            
            # Сохранение модели
            logger.info(f"Сохранение обученной модели в {self.output_dir}")
            trainer.save_model(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            
            # Удаляем временный файл, если он был создан
            if temp_data and os.path.exists(training_data_path):
                os.remove(training_data_path)
            
            return True, f"Модель успешно обучена и сохранена в {self.output_dir}"
        except Exception as e:
            logger.error(f"Ошибка в процессе обучения: {str(e)}")
            return False, f"Ошибка в процессе обучения: {str(e)}"
    
    def upload_model_to_hub(
        self, 
        repo_id: str,
        private: bool = True,
        token: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Загрузка обученной модели на Hugging Face Hub
        
        Args:
            repo_id: Идентификатор репозитория на Hugging Face Hub
            private: Флаг приватности репозитория
            token: Токен доступа к Hugging Face Hub
            
        Returns:
            (успех, сообщение)
        """
        try:
            if not os.path.exists(os.path.join(self.output_dir, "pytorch_model.bin")):
                return False, "Обученная модель не найдена. Сначала выполните обучение."
            
            # Инициализация API
            api = HfApi(token=token)
            
            # Загрузка модели на Hub
            api.create_repo(repo_id=repo_id, private=private, repo_type="model", exist_ok=True)
            api.upload_folder(
                folder_path=self.output_dir,
                repo_id=repo_id,
                repo_type="model"
            )
            
            return True, f"Модель успешно загружена на Hugging Face Hub: {repo_id}"
        except Exception as e:
            return False, f"Ошибка при загрузке модели на Hub: {str(e)}"

def finetune_from_chat_history(epochs: int = 3) -> Tuple[bool, str]:
    """
    Функция для запуска процесса дообучения на основе истории чатов
    
    Args:
        epochs: Количество эпох обучения
        
    Returns:
        (успех, сообщение)
    """
    # Анализ чатов и подготовка данных
    analyzer = ChatAnalyzer()
    report = analyzer.generate_analytics_report()
    
    # Проверка наличия достаточного количества данных
    if report["qa_pairs_count"] < 10:
        return False, f"Недостаточно данных для дообучения. Найдено всего {report['qa_pairs_count']} пар вопрос-ответ."
    
    # Создание и запуск процесса дообучения
    tuner = FineTuner()
    success, message = tuner.train(num_train_epochs=epochs)
    
    return success, message

if __name__ == "__main__":
    # Пример использования
    success, message = finetune_from_chat_history()
    print(message)
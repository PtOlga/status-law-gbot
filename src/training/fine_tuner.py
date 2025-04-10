"""
Module for fine-tuning a language model on collected data
"""

import os
import json
import tempfile
from typing import List, Dict, Any, Tuple, Optional
import logging
from datetime import datetime
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
from src.analytics.chat_evaluator import ChatEvaluator
from config.settings import (
    HF_TOKEN,
    DATASET_ID,
    DATASET_CHAT_HISTORY_PATH,
    DATASET_FINE_TUNED_PATH,
    DATASET_TRAINING_DATA_PATH,
    DATASET_TRAINING_LOGS_PATH
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FineTuner:
    def __init__(self, base_model_id: str = "IlyaGusev/saiga_7b_lora",
                 output_dir: Optional[str] = None,
                 device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"):
        self.base_model_id = base_model_id
        self.output_dir = output_dir or DATASET_FINE_TUNED_PATH
        self.device = device
        self.tokenizer = None
        self.model = None
        self.chat_analyzer = ChatAnalyzer()
        
        # Создаём директорию для сохранения моделей в датасете
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
            
            # Load tokenizer using slow tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_id,
                trust_remote_code=True,
                use_fast=False  # Using slow tokenizer
            )
            
            # Special tokens for dialogues
            special_tokens = {
                "pad_token": "<PAD>",
                "eos_token": "</s>",
                "bos_token": "<s>",
                "unk_token": "<unk>"  # Adding unknown token
            }
            
            # Add special tokens if they don't exist
            self.tokenizer.add_special_tokens({"additional_special_tokens": list(special_tokens.values())})
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                trust_remote_code=True,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype="auto"  # Automatically choose optimal data type
            )
            
            # Resize embeddings for new tokens
            self.model.resize_token_embeddings(len(self.tokenizer))
            
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def setup_lora_config(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05
    ) -> LoraConfig:
        """
        Setup LoRA configuration for efficient fine-tuning
        
        Args:
            r: Rank of LoRA matrices
            lora_alpha: LoRA alpha parameter
            lora_dropout: Dropout probability in LoRA layers
            
        Returns:
            LoRA configuration
        """
        # Create LoRA configuration
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
        Prepare model for training using LoRA
        """
        if self.model is None:
            self.load_model_and_tokenizer()
        
        # Setup LoRA
        lora_config = self.setup_lora_config()
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        
        # Output parameter information
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} of {all_params:,} ({trainable_params/all_params:.2%})")
    
    def tokenize_dataset(self, dataset):
        """
        Tokenize dataset for training
        
        Args:
            dataset: Dataset to tokenize
            
        Returns:
            Tokenized dataset
        """
        def tokenize_function(examples):
            # Format dialogues into single string
            texts = []
            for dialog in examples["messages"]:
                text = ""
                for message in dialog:
                    if message["role"] == "user":
                        text += f"User: {message['content']}\n"
                    elif message["role"] == "assistant":
                        text += f"Assistant: {message['content']}\n"
                texts.append(text)
            
            # Tokenize texts
            tokenized = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=1024,
                return_tensors="pt"
            )
            
            return tokenized
        
        # Apply tokenization function
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["messages"]
        )
        
        return tokenized_dataset
    
    # Добавить этот метод в класс fine_tuner.py или в функции модуля:

def finetune_from_annotations(epochs=3, batch_size=4, learning_rate=2e-4, min_rating=4):
    """
    Fine-tune model using annotated QA pairs
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        min_rating: Minimum average rating for including examples
        
    Returns:
        (success, message)
    """
    try:
        import tempfile
        import os
        from src.analytics.chat_evaluator import ChatEvaluator
        from config.settings import HF_TOKEN, DATASET_ID, CHAT_HISTORY_PATH
        
        # Create evaluator
        evaluator = ChatEvaluator(
            hf_token=HF_TOKEN,
            dataset_id=DATASET_ID,
            chat_history_path=CHAT_HISTORY_PATH
        )
        
        # Create temporary file for training data
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.jsonl', delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Export high-quality examples
        success, message = evaluator.export_training_data(temp_path, min_rating)
        
        if not success:
            return False, f"Failed to export training data: {message}"
        
        # Count examples
        with open(temp_path, 'r') as f:
            example_count = sum(1 for _ in f)
        
        if example_count == 0:
            return False, "No high-quality examples found for fine-tuning"
        
        # Run actual fine-tuning using the export file
        from src.training.fine_tuner import finetune_from_file
        
        success, message = finetune_from_file(
            training_file=temp_path,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except:
            pass
        
        if success:
            return True, f"Successfully fine-tuned model with {example_count} annotated examples: {message}"
        else:
            return False, f"Fine-tuning failed: {message}"
        
    except Exception as e:
        return False, f"Error during fine-tuning from annotations: {str(e)}"
    
    def train(
        self,
        training_data_path: str,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 8,
        learning_rate: float = 2e-4,
        logging_steps: int = 10,
        save_strategy: str = "epoch"
    ) -> Tuple[bool, str]:
        """
        Train the model using provided data
    
        Args:
            training_data_path: Path to training data file
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size per device
            gradient_accumulation_steps: Number of steps to accumulate gradients
            learning_rate: Learning rate
            logging_steps: Number of steps between logging
            save_strategy: When to save checkpoints
        
        Returns:
            (success, message)
        """
        try:
            # Prepare model for training
            self.prepare_model_for_training()
            
            # Load dataset
            dataset = load_dataset('json', data_files=training_data_path)['train']
            
            # Tokenize dataset
            tokenized_dataset = self.tokenize_dataset(dataset)
            
            # Create data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            
            # Setup training arguments
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
            
            # Create trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer
            )
            
            # Start training
            trainer.train()
            
            # Save model
            trainer.save_model()
            
            return True, f"Model successfully trained and saved to {self.output_dir}"
            
        except Exception as e:
            return False, f"Training failed: {str(e)}"
    
    def upload_model_to_hub(
        self, 
        repo_id: str,
        private: bool = True,
        token: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Upload trained model to Hugging Face Hub
        
        Args:
            repo_id: Repository ID on Hugging Face Hub
            private: Repository privacy flag
            token: Hugging Face Hub access token
            
        Returns:
            (success, message)
        """
        try:
            if not os.path.exists(os.path.join(self.output_dir, "pytorch_model.bin")):
                return False, "Trained model not found. Please train the model first."
            
            # Initialize API
            api = HfApi(token=token)
            
            # Upload model to Hub
            api.create_repo(repo_id=repo_id, private=private, repo_type="model", exist_ok=True)
            api.upload_folder(
                folder_path=self.output_dir,
                repo_id=repo_id,
                repo_type="model"
            )
            
            return True, f"Model successfully uploaded to Hugging Face Hub: {repo_id}"
        except Exception as e:
            return False, f"Error uploading model to Hub: {str(e)}"

def finetune_from_chat_history(epochs: int = 3, 
                             batch_size: int = 4,
                             learning_rate: float = 2e-4) -> Tuple[bool, str]:
    """
    Function to start fine-tuning process based on evaluated chat history
    
    Args:
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        
    Returns:
        (success, message)
    """
    try:
        # Create evaluator instance - убираем лишний параметр
        evaluator = ChatEvaluator(
            hf_token=HF_TOKEN,
            dataset_id=DATASET_ID
        )
        
        # Create temporary file for training data in dataset
        training_data_path = os.path.join(DATASET_TRAINING_DATA_PATH, f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
        os.makedirs(os.path.dirname(training_data_path), exist_ok=True)
        
        # Export evaluated data
        success, message = evaluator.export_training_data(
            output_file=training_data_path,
            min_rating=3
        )
        
        if not success:
            if os.path.exists(training_data_path):
                os.remove(training_data_path)
            return False, f"Failed to prepare training data: {message}"
            
        # Count examples
        with open(training_data_path, 'r') as f:
            example_count = sum(1 for _ in f)
            
        if example_count == 0:
            if os.path.exists(training_data_path):
                os.remove(training_data_path)
            return False, "No evaluated examples found for fine-tuning"
        
        # Create and start fine-tuning process
        tuner = FineTuner()
        success, message = tuner.train(
            training_data_path=training_data_path,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        if success:
            return True, f"Successfully fine-tuned model with {example_count} evaluated examples: {message}"
        else:
            return False, f"Fine-tuning failed: {message}"
            
    except Exception as e:
        return False, f"Error during fine-tuning: {str(e)}"

def finetune_from_file(
    training_file: str,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4
) -> Tuple[bool, str]:
    """
    Fine-tune model using training data from file
    
    Args:
        training_file: Path to JSONL file with training data
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        
    Returns:
        (success, message)
    """
    try:
        # Create fine tuner instance
        tuner = FineTuner()
        
        # Start training process
        success, message = tuner.train(
            training_data_path=training_file,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        return success, message
        
    except Exception as e:
        return False, f"Error during fine-tuning: {str(e)}"

if __name__ == "__main__":
    # Usage example
    success, message = finetune_from_chat_history()
    print(message)

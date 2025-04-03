"""
Module for fine-tuning a language model on collected data
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
    def __init__(self, base_model_id: str = "IlyaGusev/saiga_7b_lora",
                 output_dir: Optional[str] = None,
                 device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"):
        self.base_model_id = base_model_id
        self.output_dir = output_dir or TRAINING_OUTPUT_DIR
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
        training_data_path: Optional[str] = None,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        logging_steps: int = 10,
        save_strategy: str = "epoch"
    ) -> Tuple[bool, str]:
        """
        Start model fine-tuning process
        """
        try:
            # Prepare training data if path not specified
            if training_data_path is None:
                training_data_path = self.prepare_training_data()
                temp_data = True
            else:
                temp_data = False
            
            # Load model and tokenizer if not loaded
            if self.model is None or self.tokenizer is None:
                self.load_model_and_tokenizer()
            
            # Prepare model for training
            self.prepare_model_for_training()
            
            # Load dataset
            dataset = load_dataset("json", data_files=training_data_path, split="train")
            logger.info(f"Loaded {len(dataset)} examples from {training_data_path}")
            
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
            logger.info("Starting model training...")
            trainer.train()
            
            # Save model
            logger.info(f"Saving trained model to {self.output_dir}")
            trainer.save_model(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            
            # Remove temporary file if created
            if temp_data and os.path.exists(training_data_path):
                os.remove(training_data_path)
            
            return True, f"Model successfully trained and saved to {self.output_dir}"
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return False, f"Error during training: {str(e)}"
    
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
    Function to start fine-tuning process based on chat history
    
    Args:
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        
    Returns:
        (success, message)
    """
    # Analyze chats and prepare data
    analyzer = ChatAnalyzer()
    report = analyzer.analyze_chats()
    
    if not report or "Failed to load chat history" in report:
        return False, "Failed to load chat history for training"
    
    # Extract QA pairs for training
    qa_pairs = analyzer.extract_question_answer_pairs()
    
    if len(qa_pairs) < 10:
        return False, f"Insufficient data for fine-tuning. Only {len(qa_pairs)} QA pairs found."
    
    # Create temporary file for training data
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        for pair in qa_pairs:
            json.dump({
                "messages": [
                    {"role": "user", "content": pair["question"]},
                    {"role": "assistant", "content": pair["answer"]}
                ]
            }, f, ensure_ascii=False)
            f.write('\n')
        training_data_path = f.name
    
    # Create and start fine-tuning process
    tuner = FineTuner()
    success, message = tuner.prepare_and_train(
        training_data_path=training_data_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    # Cleanup
    if os.path.exists(training_data_path):
        os.remove(training_data_path)
    
    return success, message

if __name__ == "__main__":
    # Usage example
    success, message = finetune_from_chat_history()
    print(message)

"""
Web interface for model management and training
"""

import os
import json
import gradio as gr
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

from src.analytics.chat_analyzer import ChatAnalyzer
from src.training.fine_tuner import FineTuner, finetune_from_chat_history
from src.training.model_manager import ModelManager
from config.settings import MODEL_PATH, TRAINING_OUTPUT_DIR

# Initialize managers
model_manager = ModelManager()
chat_analyzer = ChatAnalyzer()

def get_models_df():
    """
    Get dataframe with models from registry
    
    Returns:
        pandas.DataFrame: Dataframe with models
    """
    models = model_manager.list_models()
    
    if not models:
        return pd.DataFrame(columns=["model_id", "version", "description", "is_active", "registration_date"])
    
    # Create dataframe
    df = pd.DataFrame(models)
    
    # Select required columns
    columns = ["model_id", "version", "description", "is_active", "registration_date"]
    df = df[columns]
    
    # Sort by model_id and registration_date
    df = df.sort_values(by=["model_id", "registration_date"], ascending=[True, False])
    
    return df

def generate_chat_analysis():
    """Generate analysis of chat history"""
    return chat_analyzer.analyze_chats()

def register_model_action(model_id, version, source, description, set_active):
    """
    Model registration action
    
    Args:
        model_id: Model identifier
        version: Model version
        source: Model source
        description: Model description
        set_active: Set as active
        
    Returns:
        str: Operation result
    """
    # Input validation
    if not model_id or not version or not source:
        return "Error: all fields are required"
    
    # Register model
    success, message = model_manager.register_model(
        model_id=model_id,
        version=version,
        source=source,
        description=description,
        is_active=set_active
    )
    
    if not success:
        return f"Error: {message}"
    
    # If model download option is set, download it
    if source.startswith("hf://"):
        success, download_message = model_manager.download_model(model_id, version)
        if not success:
            return f"Model registered but not downloaded: {download_message}"
        message += f"\n{download_message}"
    
    return message

def import_local_model_action(source_path, model_id, version, description, set_active):
    """
    Local model import action
    
    Args:
        source_path: Path to model directory
        model_id: Model identifier
        version: Model version
        description: Model description
        set_active: Set as active
        
    Returns:
        str: Operation result
    """
    # Input validation
    if not source_path or not model_id or not version:
        return "Error: all fields are required"
    
    # Check directory existence
    if not os.path.exists(source_path):
        return f"Error: directory {source_path} does not exist"
    
    # Import model
    success, message = model_manager.import_local_model(
        source_path=source_path,
        model_id=model_id,
        version=version,
        description=description,
        is_active=set_active
    )
    
    return message

def set_active_model_action(model_row_index, models_df):
    """
    Set active model action
    
    Args:
        model_row_index: Model row index in dataframe
        models_df: Dataframe with models
        
    Returns:
        str: Operation result
    """
    try:
        # Get selected model information
        model_row = models_df.iloc[model_row_index]
        model_id = model_row["model_id"]
        version = model_row["version"]
        
        # Set as active
        success, message = model_manager.set_active_model(model_id, version)
        
        return message
    except Exception as e:
        return f"Error: {str(e)}"

def delete_model_action(model_row_index, models_df):
    """
    Delete model action
    
    Args:
        model_row_index: Model row index in dataframe
        models_df: Dataframe with models
        
    Returns:
        str: Operation result
    """
    try:
        # Get selected model information
        model_row = models_df.iloc[model_row_index]
        model_id = model_row["model_id"]
        version = model_row["version"]
        
        # Delete model
        success, message = model_manager.delete_model(model_id, version)
        
        return message
    except Exception as e:
        return f"Error: {str(e)}"

def start_finetune_action(epochs, batch_size, learning_rate):
    """Start model fine-tuning"""
    try:
        from src.training.fine_tuner import finetune_from_chat_history
        
        success, message = finetune_from_chat_history(
            epochs=epochs
        )
        
        return f"Training {'completed' if success else 'failed'}: {message}"
    except Exception as e:
        return f"Error starting training: {str(e)}"

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
from config.settings import MODEL_PATH, MODELS_REGISTRY_PATH, MODELS, ACTIVE_MODEL

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        """Initialize model manager"""
        self.registry_path = MODELS_REGISTRY_PATH
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        
    def _load_registry(self) -> List[Dict[str, Any]]:
        """Load models registry"""
        try:
            if os.path.exists(self.registry_path):
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error loading registry: {str(e)}")
            return []

    def _save_registry(self, registry: List[Dict[str, Any]]) -> bool:
        """Save models registry"""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(registry, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving registry: {str(e)}")
            return False

    def register_model(
        self,
        model_id: str,
        version: str,
        source: str,
        description: str = "",
        is_active: bool = False
    ) -> Tuple[bool, str]:
        """
        Register new model version
        
        Args:
            model_id: Model identifier
            version: Model version
            source: Model source (local path or Hugging Face model id)
            description: Model description
            is_active: Set as active model
            
        Returns:
            (success, message)
        """
        try:
            registry = self._load_registry()
            
            # Check if model version already exists
            for model in registry:
                if model["model_id"] == model_id and model["version"] == version:
                    return False, f"Model {model_id} version {version} already exists"
            
            # Add new model
            registry.append({
                "model_id": model_id,
                "version": version,
                "source": source,
                "description": description,
                "is_active": is_active,
                "registration_date": datetime.now().isoformat()
            })
            
            # If this model is set as active, deactivate others
            if is_active:
                for model in registry[:-1]:  # Skip the last one (just added)
                    if model["model_id"] == model_id:
                        model["is_active"] = False
            
            # Save registry
            if self._save_registry(registry):
                return True, f"Model {model_id} version {version} registered successfully"
            return False, "Failed to save registry"
            
        except Exception as e:
            return False, f"Error registering model: {str(e)}"

    def list_models(self) -> List[Dict[str, Any]]:
        """Get list of registered models"""
        return self._load_registry()

    def set_active_model(self, model_id: str, version: str) -> Tuple[bool, str]:
        """
        Set model version as active
        
        Args:
            model_id: Model identifier
            version: Model version
            
        Returns:
            (success, message)
        """
        try:
            registry = self._load_registry()
            model_found = False
            
            # Update active status
            for model in registry:
                if model["model_id"] == model_id:
                    model["is_active"] = (model["version"] == version)
                    if model["version"] == version:
                        model_found = True
            
            if not model_found:
                return False, f"Model {model_id} version {version} not found"
            
            # Save registry
            if self._save_registry(registry):
                return True, f"Model {model_id} version {version} set as active"
            return False, "Failed to save registry"
            
        except Exception as e:
            return False, f"Error setting active model: {str(e)}"

    def delete_model(self, model_id: str, version: str) -> Tuple[bool, str]:
        """
        Delete model version
        
        Args:
            model_id: Model identifier
            version: Model version
            
        Returns:
            (success, message)
        """
        try:
            registry = self._load_registry()
            
            # Find and remove model
            for i, model in enumerate(registry):
                if model["model_id"] == model_id and model["version"] == version:
                    if model["is_active"]:
                        return False, "Cannot delete active model"
                    registry.pop(i)
                    
                    # Save registry
                    if self._save_registry(registry):
                        return True, f"Model {model_id} version {version} deleted"
                    return False, "Failed to save registry"
            
            return False, f"Model {model_id} version {version} not found"
            
        except Exception as e:
            return False, f"Error deleting model: {str(e)}"

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
    try:
        model_path = ACTIVE_MODEL["training"]["fine_tuned_path"] if version else ACTIVE_MODEL["training"]["base_model_path"]
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else None
        )
        
        return model, tokenizer, ACTIVE_MODEL
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise ValueError(f"Failed to load model: {str(e)}")

if __name__ == "__main__":
    # Usage example
    manager = ModelManager()
    
    # Register base model from config
    success, message = manager.register_model(
        model_id=ACTIVE_MODEL["id"].split("/")[-1],  # Extract model name from full HF path
        version=ACTIVE_MODEL["type"],
        source=ACTIVE_MODEL["id"],
        description=ACTIVE_MODEL["description"],
        is_active=True
    )
    print(message)
    
    # Print models list
    models = manager.list_models()
    print(f"Registry contains {len(models)} models:")
    for model in models:
        print(f" - {model['model_id']} v{model['version']}: {model['description']}")

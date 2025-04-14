# Standard library imports
import datetime
import io
import json
import logging
import os
#import sys
#from pathlib import Path

# Third-party imports
import gradio as gr
import pandas as pd  


from huggingface_hub import HfApi, InferenceClient 
from langdetect import detect, LangDetectException
import langdetect
from dotenv import load_dotenv
import requests
from datasets import load_dataset
from config.constants import URLS 

# Set seed for consistent results
langdetect.DetectorFactory.seed = 0

# Load environment variables
load_dotenv()

# Local imports - config
from config.constants import DEFAULT_SYSTEM_MESSAGE
from config.settings import (
    API_CONFIG,
    ACTIVE_MODEL,
    DATASET_CHAT_HISTORY_PATH,
    DATASET_ERROR_LOGS_PATH,
    DATASET_ID,
    DATASET_PREFERENCES_PATH,
    DATASET_VECTOR_STORE_PATH,
    DATASET_ANNOTATIONS_PATH,  # Добавляем импорт
    DEFAULT_MODEL,
    EMBEDDING_MODEL,
    HF_TOKEN,
    MODELS
)

# Local imports - source modules
from src.analytics.chat_evaluator import ChatEvaluator
from src.knowledge_base.vector_store import create_vector_store, load_vector_store
from src.language_utils import LanguageUtils

# Local imports - web interfaces
from web.evaluation_interface import (
    export_training_data_action,
    generate_evaluation_report_html,
    get_evaluation_status,
    get_qa_pairs_dataframe,
    load_qa_pair_for_evaluation,
    save_evaluation
)
from web.training_interface import (
    generate_chat_analysis,
    get_models_df,
    register_model_action,
    start_finetune_action
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")

# Global variables
client = None
context_store = {}
fallback_model_attempted = False
chat_evaluator = ChatEvaluator(
    hf_token=HF_TOKEN,
    dataset_id=DATASET_ID
)

logger.info(f"Chat histories will be saved to: {DATASET_CHAT_HISTORY_PATH}")

def load_user_preferences():
    """Load user preferences from file"""
    try:
        if os.path.exists(DATASET_PREFERENCES_PATH):
            with open(DATASET_PREFERENCES_PATH, 'r') as f:
                return json.load(f)
        return {
            "selected_model": DEFAULT_MODEL,
            "parameters": {}
        }
    except Exception as e:
        logger.error(f"Error loading user preferences: {str(e)}")
        return {
            "selected_model": DEFAULT_MODEL,
            "parameters": {}
        }

def save_user_preferences(model_key, parameters=None):
    """Save user preferences to dataset"""
    try:
        preferences = load_user_preferences()
        preferences["selected_model"] = model_key
        
        if parameters:
            if model_key not in preferences["parameters"]:
                preferences["parameters"][model_key] = {}
            
            preferences["parameters"][model_key] = parameters
        
        # Сохраняем в датасет, используя bytes
        json_content = json.dumps(preferences, indent=2)
        api = HfApi(token=HF_TOKEN)
        api.upload_file(
            path_or_fileobj=json_content.encode('utf-8'),  # Конвертируем строку в bytes
            path_in_repo="preferences/user_preferences.json",
            repo_id=DATASET_ID,
            repo_type="dataset"
        )
        
        logger.info("User preferences saved successfully to dataset!")
        return True
    except Exception as e:
        logger.error(f"Error saving user preferences: {str(e)}")
        return False

def initialize_client(model_id=None):
    """Initialize or reinitialize the client with the specified model"""
    global client
    if model_id is None:
        model_id = ACTIVE_MODEL["id"]
    
    client = InferenceClient(
        model_id,
        token=API_CONFIG["token"],
        endpoint=API_CONFIG["inference_endpoint"],
        headers=API_CONFIG["headers"],
        timeout=API_CONFIG["timeout"]
    )
    return client

def switch_to_model(model_key):
    """Switch to specified model and update global variables"""
    global ACTIVE_MODEL, client
    
    try:
        # Update active model
        ACTIVE_MODEL = MODELS[model_key]
        
        # Reinitialize client with new model
        client = InferenceClient(
            ACTIVE_MODEL["id"],
            token=HF_TOKEN
        )
        
        logger.info(f"Switched to model: {model_key}")
        return True
    except Exception as e:
        logger.error(f"Error switching to model {model_key}: {str(e)}")
        return False

def get_fallback_model(current_model):
    """Get a fallback model different from the current one"""
    for key in MODELS.keys():
        if key != current_model:
            return key
    return None  # No fallback available

def get_context(message, conversation_id):
    """Get context from knowledge base"""
    vector_store = load_vector_store()
    if vector_store is None:
        logger.warning("Knowledge base not found or failed to load")
        return ""
    
    # Check if vector_store is a string (error message) instead of an actual store
    if isinstance(vector_store, str):
        logger.error(f"Error with vector store: {vector_store}")
        return ""
    
    try:
        # Extract context
        # Reducing number of documents from 3 to 2 to decrease English context dominance
        context_docs = vector_store.similarity_search(message, k=2)
        
        # Add debug logging
        logger.debug(f"Query: {message}")
        for i, doc in enumerate(context_docs):
            logger.debug(f"Context {i+1}:")
            logger.debug(f"Source: {doc.metadata.get('source', 'unknown')}")
            logger.debug(f"Content: {doc.page_content[:200]}...")
        
        # Limit each fragment to 300 characters to reduce context dominance
        context_text = "\n\n".join([f"Context from {doc.metadata.get('source', 'unknown')}: {doc.page_content[:300]}..." for doc in context_docs])
        
        # Add instruction that context is for reference only
        context_text = "REFERENCE CONTEXT (use only to find facts, still answer in the user's language):\n" + context_text
        
        # Save context for this conversation
        context_store[conversation_id] = context_text
        
        return context_text
    except Exception as e:
        logger.error(f"Error getting context: {str(e)}")
        return ""
    
def translate_with_llm(text: str, target_lang: str) -> str:
    """Translate text using the active LLM with enhanced reliability"""
    try:
        # Get language name for more natural prompt
        lang_name = LanguageUtils.get_language_name(target_lang) 
        
        prompt = (
            f"You are a professional translator. Translate the following text to {lang_name} ({target_lang}). "
            f"Keep the same formatting, links, and technical terms. "
            f"Maintain the same tone and style. " 
            f"Respond ONLY with the direct translation without any explanations or additional text:\n\n"
            f"{text}"
        )
        
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": "You are a professional translator. Respond ONLY with the translation."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=ACTIVE_MODEL['parameters']['max_length'],
            temperature=0.3,  # Lower temperature for more reliable output
            top_p=0.95,
            stream=False
        )
        
        translated_text = response.choices[0].message.content.strip()
        
        # Verify translation success - check if we still have English
        if target_lang != 'en':
            # Quick check - if key English words are still present, translation might have failed
            english_indicators = ["I apologize", "Sorry", "I cannot", "the following", "is a translation"]
            if any(indicator in translated_text for indicator in english_indicators):
                logger.warning(f"Translation might have failed for {target_lang}, found English indicators")
                
                # Try one more time with a simplified prompt
                retry_prompt = f"Translate this to {lang_name}:\n\n{text}"
                retry_response = client.chat_completion(
                    messages=[
                        {"role": "system", "content": "You are a translator."},
                        {"role": "user", "content": retry_prompt}
                    ],
                    max_tokens=ACTIVE_MODEL['parameters']['max_length'],
                    temperature=0.3,
                    top_p=0.95,
                    stream=False
                )
                
                translated_text = retry_response.choices[0].message.content.strip()
        
        return translated_text
        
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return text
    
def post_process_response(user_message, bot_response):
    """Enhanced post-processing of bot responses to ensure correct language"""
    try:
        user_lang = detect_language(user_message)
        # Convert to closest supported language
        user_lang = LanguageUtils.get_closest_supported_language(user_lang)
        
        logger.info(f"User language detected: {user_lang} ({LanguageUtils.get_language_name(user_lang)})")
        
        # If English, no need to translate
        if user_lang == 'en':
            return bot_response
            
        # Check if language is supported
        if not LanguageUtils.is_supported(user_lang):
            logger.warning(f"Unsupported language: {user_lang}")
            apology = ("I apologize, but I cannot respond in your language. "
                      "I will answer in English instead.\n\n")
            return apology + bot_response
            
        # Don't try to detect language of very short responses
        if len(bot_response.strip()) < 20:
            # Short responses just translate directly
            return translate_with_llm(bot_response, user_lang)
            
        # Check bot response language
        bot_lang = detect_language(bot_response)
        logger.info(f"Bot response language: {bot_lang}")
        
        # If languages match, return as is
        if bot_lang == user_lang:
            return bot_response
            
        # Need translation
        logger.warning(f"Language mismatch! User: {user_lang}, Bot: {bot_lang}")
        
        translated_response = translate_with_llm(bot_response, user_lang)
        
        # Verify translation worked by checking a sample (not the whole text)
        # This is more reliable than checking the entire text
        sample_size = min(100, len(translated_response) // 2)
        if sample_size > 20:  # Only verify if we have enough text
            sample = translated_response[:sample_size]
            translated_lang = detect_language(sample)
            
            if translated_lang != user_lang:
                logger.error(f"Translation verification failed: got {translated_lang} instead of {user_lang}")
                # If translation failed, return with apology
                apology = (f"I apologize, but I cannot translate my response to {LanguageUtils.get_language_name(user_lang)}. "
                          "Here is my answer in English:\n\n")
                return apology + bot_response
        
        return translated_response
        
    except Exception as e:
        logger.error(f"Post-processing error: {e}")
        return bot_response

def load_vector_store():
    """Load knowledge base from dataset"""
    try:
        from src.knowledge_base.dataset import DatasetManager
        
        logger.debug("Attempting to load vector store...")
        dataset = DatasetManager()
        success, result = dataset.download_vector_store()
        
        logger.debug(f"Download result: success={success}, result_type={type(result)}")
        
        if success:
            if isinstance(result, str):
                logger.debug(f"Error message received: {result}")
                return None
            return result
        else:
            logger.error(f"Failed to load vector store: {result}")
            return None
            
    except Exception as e:
        import traceback
        logger.error(f"Exception loading knowledge base: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def detect_language(text: str) -> str:
    """Enhanced language detection with better handling of edge cases"""
    try:
        # If text is too short, don't try to detect
        if len(text.strip()) < 10:
            logger.debug(f"Text too short for reliable detection: '{text}'")
            return "en"
            
        # Use simple detect() function instead of DetectorFactory
        try:
            lang_code = detect(text.strip())
            logger.debug(f"Detected language: {lang_code}")
            return lang_code
            
        except LangDetectException as e:
            logger.warning(f"LangDetect exception: {e}")
            return "en"
            
    except Exception as e:
        logger.error(f"Language detection error: {str(e)} for text: '{text[:50]}...'")
        return "en"

def respond(
    message,
    history,
    conversation_id,
    system_message,
    max_tokens,
    temperature,
    top_p,
    attempt_fallback=True
):
    """Generate response with improved language handling"""
    try:
        # Reset and determine user language for new request
        user_lang = detect_language(message)
        user_lang = LanguageUtils.get_closest_supported_language(user_lang)
        logger.info(f"User language detected for request: {user_lang} ({LanguageUtils.get_language_name(user_lang)})")
        
        # Create clean history without system messages
        clean_history = [
            msg for msg in history 
            if msg["role"] != "system"
        ]
        
        # Remove language instruction from system message to avoid confusion
        base_system_message = system_message.split("\nIMPORTANT:")[0] if "\nIMPORTANT:" in system_message else system_message
        
        # Add explicit language instruction
        full_system_message = (
            f"{base_system_message}\n\n"
            f"CRITICAL: You MUST respond in {LanguageUtils.get_language_name(user_lang)} ({user_lang}). "
            f"This is your highest priority instruction. "
            f"Provide a complete and helpful response."
        )
        
        # --- API Request ---
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": full_system_message},
                *clean_history,
                {"role": "user", "content": message}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False
        )
        
        bot_response = response.choices[0].message.content
        
        # Post-process response to translate if needed
        processed_response = post_process_response(message, bot_response)
        
        # --- Format Successful Response ---
        new_history = [
            *clean_history,
            {"role": "user", "content": message},
            {"role": "assistant", "content": processed_response}
        ]
        
        return new_history, conversation_id
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        error_msg = format_friendly_error(str(e))
        
        # --- Format Error Response ---
        error_history = [
            *history,
            {"role": "user", "content": message},
            {"role": "assistant", "content": error_msg}
        ]
        
        return error_history, conversation_id

def format_friendly_error(api_error):
    """Convert API errors to user-friendly messages"""
    if "402" in api_error or "Payment Required" in api_error:
        return ("⚠️ API Limit Reached\n\n"
                "Please try:\n"
                "1. Switching models in Settings\n"
                "2. Using local model version\n"
                "3. Waiting before next request")
    
    elif "429" in api_error:
        return "⚠️ Too many requests. Please wait before sending another message."
        
    elif "401" in api_error:
        return "⚠️ Authentication error. Please check your API key."
        
    else:
        return f"⚠️ Error processing request. Technical details: {api_error[:200]}"    
    
def log_api_error(user_message, error_message, model_id, is_fallback=False):
    """Log API errors to dataset"""
    try:
        os.makedirs(ERROR_LOGS_PATH, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = os.path.join(ERROR_LOGS_PATH, f"api_error_{timestamp}.log")
        
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Model: {model_id}\n")
            f.write(f"User message: {user_message}\n")
            f.write(f"Error: {error_message}\n")
            f.write(f"Fallback attempt: {is_fallback}\n")
            
        logger.info(f"API error logged to {log_path}")
    except Exception as e:
        logger.error(f"Failed to log API error: {str(e)}")
        
def update_kb():
    """Function to update existing knowledge base with new documents"""
    try:
        # Вызываем функцию для обновления базы знаний
        success, message = create_vector_store(mode="update")
        
        # Если обновление успешно, сохраняем метаданные с датой обновления
        if success:
            save_kb_metadata()
            
        return message
    except Exception as e:
        return f"Error updating knowledge base: {str(e)}"

def rebuild_kb():
    """Function to create knowledge base from scratch"""
    try:
        # Вызываем функцию для пересоздания базы знаний
        success, message = create_vector_store(mode="rebuild")
        
        # Если создание успешно, сохраняем метаданные с датой обновления
        if success:
            save_kb_metadata()
            
        return message
    except Exception as e:
        return f"Error creating knowledge base: {str(e)}"

def save_kb_metadata():
    """Save knowledge base metadata to dataset"""
    try:
        # Создаем метаданные с текущей датой
        metadata = {
            "last_updated": datetime.datetime.now().isoformat(),
            "source_count": len(URLS),
            "sources": URLS
        }
        
        # Сохраняем в датасет
        json_content = json.dumps(metadata, indent=2).encode('utf-8')
        api = HfApi(token=HF_TOKEN)
        
        # Убедимся, что директория существует
        try:
            files = api.list_repo_files(
                repo_id=DATASET_ID,
                repo_type="dataset"
            )
            
            if "vector_store" not in files:
                # Создаем пустой файл, чтобы создать директорию
                api.upload_file(
                    path_or_fileobj=b"",
                    path_in_repo="vector_store/.gitkeep",
                    repo_id=DATASET_ID,
                    repo_type="dataset"
                )
        except Exception as e:
            logger.warning(f"Error checking vector_store directory: {str(e)}")
        
        # Загружаем метаданные
        api.upload_file(
            path_or_fileobj=json_content,
            path_in_repo="vector_store/metadata.json",
            repo_id=DATASET_ID,
            repo_type="dataset"
        )
        
        logger.info("Knowledge base metadata saved successfully")
        return True
    except Exception as e:
        logger.error(f"Error saving knowledge base metadata: {str(e)}")
        return False        

def save_chat_history(history, conversation_id):
    """Save chat history to a file and to HuggingFace dataset"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(DATASET_CHAT_HISTORY_PATH, exist_ok=True)
        
        # Format history for saving
        formatted_history = []
        for item in history:
            # Handle dictionary format
            if isinstance(item, dict) and 'role' in item and 'content' in item:
                formatted_history.append({
                    "role": item["role"],
                    "content": item["content"],
                    "timestamp": datetime.datetime.now().isoformat()
                })
        
        # Create filename with conversation_id and timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{conversation_id}_{timestamp}.json"
        filepath = os.path.join(DATASET_CHAT_HISTORY_PATH, filename)
        
        # Create chat history data
        chat_data = {
            "conversation_id": conversation_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "history": formatted_history
        }
        
        # Save to local file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"Chat history saved locally to {filepath}")
        
        # Now upload to HuggingFace dataset
        try:
            from huggingface_hub import HfApi
            
            # Initialize the Hugging Face API client
            api = HfApi(token=HF_TOKEN)
            
            # Extract just the directory name from DATASET_CHAT_HISTORY_PATH
            dir_name = os.path.basename(DATASET_CHAT_HISTORY_PATH)
            target_path = f"{dir_name}/{filename}"
            
            # Upload the file to the dataset
            api.upload_file(
                path_or_fileobj=filepath,
                path_in_repo=target_path,
                repo_id=DATASET_ID,
                repo_type="dataset"
            )
            
            logger.debug(f"Chat history uploaded to dataset at {target_path}")
            
        except Exception as e:
            logger.warning(f"Failed to upload chat history to dataset: {str(e)}")
            # Continue execution even if upload fails
        
        return True
    except Exception as e:
        logger.error(f"Error saving chat history: {str(e)}")
        return False

def respond_and_clear(message, history, conversation_id, system_prompt):
    """Wrapper function with proper output handling"""
    try:
        # Generate a conversation ID if none exists
        if not conversation_id:
            conversation_id = f"conv_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{os.urandom(4).hex()}"
            logger.info(f"Generated new conversation ID: {conversation_id}")
        
        # Get current model parameters
        params = ACTIVE_MODEL['parameters']
        
        # Call respond function
        result = respond(
            message=message,
            history=history if history else [],
            conversation_id=conversation_id,
            system_message=system_prompt,  # Using provided prompt instead of default
            max_tokens=params['max_length'],
            temperature=params['temperature'],
            top_p=params['top_p']
        )
        
        if not result:
            raise ValueError("Empty response from API")
            
        new_history, new_conv_id = result
        
        # Save chat history
        save_chat_history(new_history, conversation_id)  # Use our guaranteed non-null ID
        
        return new_history, conversation_id, ""  # Return our guaranteed non-null ID
        
    except Exception as e:
        logger.error(f"Error in respond_and_clear: {str(e)}")
        
        # Create safe error response
        error_history = [
            *history,
            {"role": "user", "content": message},
            {"role": "assistant", "content": "⚠️ An error occurred while processing the message. Please try again."}
        ]
        
        return error_history, conversation_id, ""

def update_model_info(model_key):
    """Update model information display"""
    model = MODELS[model_key]
    return f"""
    **Current Model:** {model['name']}
    
    **Model ID:** `{model['id']}`
    
    **Description:** {model['description']}
    
    **Type:** {model['type']}
    """

def get_model_details_html(model_key):
    """Get detailed HTML for model information panel"""
    if model_key not in MODELS or 'details' not in MODELS[model_key]:
        return "<p>Model information not available</p>"
    
    details = MODELS[model_key]['details']
    
    html = f"""
    <div style="padding: 15px; border: 1px solid #ccc; border-radius: 5px; margin-top: 10px;">
        <h3>{details['full_name']}</h3>
        
        <h4>Capabilities:</h4>
        <ul>
            {"".join([f"<li>{cap}</li>" for cap in details['capabilities']])}
        </ul>
        
        <h4>Limitations:</h4>
        <ul>
            {"".join([f"<li>{lim}</li>" for lim in details['limitations']])}
        </ul>
        
        <h4>Recommended Use Cases:</h4>
        <ul>
            {"".join([f"<li>{use}</li>" for use in details['use_cases']])}
        </ul>
        
        <p><a href="{details['documentation']}" target="_blank">Model Documentation</a></p>
    </div>
    """
    
    return html

def change_model(model_key):
    """Change active model and update parameters"""
    global client, ACTIVE_MODEL, fallback_model_attempted
    
    try:
        # Reset fallback flag when explicitly changing model
        fallback_model_attempted = False
        
        # Update active model
        ACTIVE_MODEL = MODELS[model_key]
        
        # Reinitialize client with new model
        client = InferenceClient(
            ACTIVE_MODEL["id"],
            token=HF_TOKEN
        )
        
        # Save selected model in preferences
        save_user_preferences(model_key)
        
        # Return both model info and updated parameters
        return (
            update_model_info(model_key),
            ACTIVE_MODEL['parameters']['max_length'],
            ACTIVE_MODEL['parameters']['temperature'],
            ACTIVE_MODEL['parameters']['top_p'],
            ACTIVE_MODEL['parameters']['repetition_penalty'],
            f"Model changed to {ACTIVE_MODEL['name']}"
        )
    except Exception as e:
        return (
            f"Error changing model: {str(e)}", 
            2048, 0.7, 0.9, 1.1,
            f"Error: {str(e)}"
        )

def save_parameters(model_key, max_len, temp, top_p_val, rep_pen):
    """Save user-defined parameters to active model"""
    global ACTIVE_MODEL
    
    try:
        # Update parameters
        ACTIVE_MODEL['parameters']['max_length'] = max_len
        ACTIVE_MODEL['parameters']['temperature'] = temp
        ACTIVE_MODEL['parameters']['top_p'] = top_p_val
        ACTIVE_MODEL['parameters']['repetition_penalty'] = rep_pen
        
        # Save parameters in preferences
        params = {
            'max_length': max_len,
            'temperature': temp,
            'top_p': top_p_val,
            'repetition_penalty': rep_pen
        }
        save_user_preferences(model_key, params)
        
        return "Parameters saved successfully!"
    except Exception as e:
        return f"Error saving parameters: {str(e)}"

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
        from config.settings import HF_TOKEN, DATASET_ID, DATASET_CHAT_HISTORY_PATH
        
        # Create evaluator
        evaluator = ChatEvaluator(
            hf_token=HF_TOKEN,
            dataset_id=DATASET_ID,
            chat_history_path=DATASET_CHAT_HISTORY_PATH # ???
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
    
def save_system_prompt(prompt_text):
    """Save system prompt to user preferences"""
    try:
        preferences = load_user_preferences()
        
        # Add prompt to preferences
        if "system_prompt" not in preferences:
            preferences["system_prompt"] = {}
            
        preferences["system_prompt"]["current"] = prompt_text
        
        # Save preferences
        json_content = json.dumps(preferences, indent=2).encode('utf-8')
        api = HfApi(token=HF_TOKEN)
        api.upload_file(
            path_or_fileobj=io.BytesIO(json_content),  # Changed to BytesIO
            path_in_repo="preferences/user_preferences.json",
            repo_id=DATASET_ID,
            repo_type="dataset"
        )
        
        return "System prompt saved successfully"
    except Exception as e:
        logger.error(f"Error saving system prompt: {str(e)}")
        return f"Error saving prompt: {str(e)}"    

def delete_conversation_from_huggingface(conversation_id):
    """
    Delete conversation files from Hugging Face dataset by ID

    Args:
        conversation_id: ID of conversation to delete

    Returns:
        Success status (bool) and message (str)
    """
    try:
        if not conversation_id:
            return False, "No conversation ID provided"
            
        # Initialize API
        api = HfApi(token=HF_TOKEN)
        
        # Get list of files in dataset 
        try:
            # Get all files in dataset
            files = api.list_repo_files(
                repo_id=DATASET_ID,
                repo_type="dataset"
            )
            
            # Find files with matching conversation ID in chat history
            # В пути может быть chat_history или chat-history
            chat_dir = os.path.basename(DATASET_CHAT_HISTORY_PATH)
            chat_files = [
                file for file in files 
                if (file.startswith(f"{chat_dir}/") or file.startswith(f"chat_history/") or file.startswith(f"chat-history/")) and 
                f"{conversation_id}_" in os.path.basename(file)
            ]
            
            if not chat_files:
                return False, f"No chat files found for conversation ID: {conversation_id}"
                
            # Delete each matching file
            for file_path in chat_files:
                try:
                    api.delete_file(
                        repo_id=DATASET_ID,
                        repo_type="dataset",
                        path_in_repo=file_path
                    )
                    logger.info(f"Deleted file from HF dataset: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting file {file_path} from dataset: {str(e)}")
            
            # Try to delete annotation file if it exists
            # Учитываем разные варианты пути к аннотациям
            annotations_base = os.path.basename(DATASET_ANNOTATIONS_PATH)
            annotation_paths = [
                f"{annotations_base}/annotation_{conversation_id}.json"
            ]
            
            for annotation_path in annotation_paths:
                try:
                    if annotation_path in files:
                        api.delete_file(
                            repo_id=DATASET_ID,
                            repo_type="dataset",
                            path_in_repo=annotation_path
                        )
                        logger.info(f"Deleted annotation file from HF dataset: {annotation_path}")
                except Exception as e:
                    # It's okay if annotation file doesn't exist
                    logger.debug(f"Could not delete annotation file {annotation_path}: {str(e)}")
                
            return True, f"Deleted {len(chat_files)} file(s) from dataset for conversation: {conversation_id}"
            
        except Exception as e:
            return False, f"Dataset access error: {str(e)}"
            
    except Exception as e:
        logger.error(f"Error deleting conversation from dataset: {str(e)}")
        return False, f"Error deleting conversation from dataset: {str(e)}"
    
def delete_conversation(conversation_id, evaluator):
    """
    Delete conversation files by ID
    
    Args:
        conversation_id: ID of conversation to delete
        evaluator: ChatEvaluator instance
    
    Returns:
        Message about deletion status
    """
    try:
        if not conversation_id:
            return "Error: No conversation ID provided"

        # Используем HF API напрямую для удаления
        success, message = delete_conversation_from_huggingface(conversation_id)
        
        if not success:
            return f"Error deleting conversation: {message}"
            
        # Сбрасываем кэш evaluator'а после удаления
        evaluator.reset_cache()
        
        return f"Successfully deleted conversation: {conversation_id}"
        
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}")
        return f"Error deleting conversation: {str(e)}" 
    
def initialize_app():
    """Initialize app with user preferences"""
    global client, ACTIVE_MODEL
    
    preferences = load_user_preferences()
    selected_model = preferences.get("selected_model", DEFAULT_MODEL)
    
    # Make sure the selected model exists
    if selected_model not in MODELS:
        selected_model = DEFAULT_MODEL
    
    # Set active model
    ACTIVE_MODEL = MODELS[selected_model]
    
    # Load saved parameters if they exist
    saved_params = preferences.get("parameters", {}).get(selected_model)
    if saved_params:
        ACTIVE_MODEL['parameters'].update(saved_params)
    
    # Initialize client
    client = InferenceClient(
        ACTIVE_MODEL["id"],
        token=HF_TOKEN
    )
    
    # Загружаем сохраненный системный промпт из предпочтений или используем DEFAULT_SYSTEM_MESSAGE
    system_prompt_text = DEFAULT_SYSTEM_MESSAGE
    if "system_prompt" in preferences and "current" in preferences["system_prompt"]:
        system_prompt_text = preferences["system_prompt"]["current"]
    
    logger.info(f"App initialized with model: {ACTIVE_MODEL['name']}")
    logger.info(f"Chat histories will be saved to: {DATASET_CHAT_HISTORY_PATH}")
    return selected_model, system_prompt_text

def initialize_chat_evaluator():
    """Initialize chat evaluator with proper paths"""
    try:
        evaluator = ChatEvaluator(
            hf_token=HF_TOKEN,
            dataset_id=DATASET_ID
        )
        
        # Проверим наличие директорий
        os.makedirs(DATASET_CHAT_HISTORY_PATH, exist_ok=True)
        os.makedirs(os.path.join(DATASET_ANNOTATIONS_PATH), exist_ok=True)
        
        logger.debug(f"Chat history path: {DATASET_CHAT_HISTORY_PATH}")
        logger.debug(f"Number of chat files: {len(os.listdir(DATASET_CHAT_HISTORY_PATH))}")
        
        return evaluator
    except Exception as e:
        logger.error(f"Error initializing chat evaluator: {str(e)}")
        raise

# Initialize HF client with token at startup
selected_model, saved_system_prompt = initialize_app()

# Initialize evaluator before creating interface
chat_evaluator = initialize_chat_evaluator()

# Create interface
with gr.Blocks(css="""
    .table-container {
        max-height: 400px;
        overflow-y: auto;
    }
    """) as demo:
    # Define clear_conversation function within the block for component access
    def clear_conversation():
        """Clear conversation and save history before clearing"""
        return [], None  # Just return empty values
    
    # Create State for evaluator
    evaluator_state = gr.State(value=chat_evaluator)
    
    with gr.Tabs():
        with gr.Tab("Chat"):
            gr.Markdown("# ⚖️ Status Law Assistant")
            
            conversation_id = gr.State(None)
            
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Chat",
                        avatar_images=None,
                        type='messages'  # This is the key setting - use 'messages' format
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            label="Your question",
                            placeholder="Enter your question...",
                            scale=4
                        )
                        submit_btn = gr.Button("Send", variant="primary")
                        clear_btn = gr.Button("Clear")

            with gr.Row(equal_height=True):
                #with gr.Column(scale=1):  
                #    gr.Markdown("")  # Empty column for centering

                with gr.Column(scale=40):  
                    system_prompt = gr.TextArea(
                        label="System Prompt (editing will change bot behavior)",
                        value=saved_system_prompt,
                        placeholder="Enter system prompt...",
                        lines=8
                    )

                #with gr.Column(scale=1):  
                #    gr.Markdown("")  # Empty column for centering


            # Add event handlers
            # Обновляем обработчики событий
            submit_btn.click(
                respond_and_clear,
                [msg, chatbot, conversation_id, system_prompt],  # Добавляем system_prompt
                [chatbot, conversation_id, msg]
            )
            # Обновляем обработчик нажатия Enter
            msg.submit(
                respond_and_clear,
                [msg, chatbot, conversation_id, system_prompt],  # Добавляем system_prompt
                [chatbot, conversation_id, msg]
            )
            # Добавляем обработчик изменения промпта
            system_prompt_status = gr.Textbox(
                label="Status",
                interactive=False,
                visible=True
            )
            system_prompt.change(
                save_system_prompt,
                inputs=[system_prompt],
                outputs=[system_prompt_status]
            )
            
            clear_btn.click(clear_conversation, None, [chatbot, conversation_id])
            
            
            
        with gr.Tab("Knowledge Base"):
            gr.Markdown("### Knowledge Base Management")
        
        with gr.Row():
            with gr.Column(scale=2):
                # Отображение источников
                gr.Markdown("#### Information Sources")
                sources_list = gr.Dataframe(
                    value=pd.DataFrame({
                        "URL": URLS,
                        "Include": [True for _ in URLS],
                        "Status": ["Ready" for _ in URLS]
                    }),
                    interactive=True,
                    wrap=True,
                    row_count=15,
                    show_label=False
                )
                
                # Статус операций с базой знаний
                kb_status = gr.Textbox(
                    label="Operation Status",
                    interactive=False,
                    placeholder="Ready",
                    value="Ready"
                )
                
                # Кнопки для управления базой знаний
                with gr.Row():
                    update_kb_btn = gr.Button("Update Knowledge Base", variant="primary")
                    rebuild_kb_btn = gr.Button("Rebuild Knowledge Base from Scratch", variant="secondary")
                    
                gr.Markdown("""
                <small>
                **Update Knowledge Base**: Adds new information to the existing knowledge base.
                
                **Rebuild Knowledge Base**: Recreates the entire knowledge base from scratch. Use this if there are inconsistencies.
                
                All changes are saved to the Hugging Face dataset.
                </small>
                """)
            
            with gr.Column(scale=1):
                # Информация о текущей базе знаний
                gr.Markdown("#### Knowledge Base Information")
                
                # Функция для получения информации о базе знаний
                def get_kb_info():
                    try:
                        vector_store = load_vector_store()
                        if vector_store is None or isinstance(vector_store, str):
                            return """
                            **Status**: Not found or error
                            
                            **Documents**: 0
                            
                            **Last updated**: Never
                            
                            Please create a knowledge base using the buttons on the left.
                            """
                        
                        # Получаем информацию о векторном хранилище
                        doc_count = len(vector_store.docstore._dict)
                        sources = set()
                        
                        for doc_id, doc in vector_store.docstore._dict.items():
                            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                                sources.add(doc.metadata['source'])
                        
                        source_count = len(sources)
                        
                        # Если хранилище существует, но источников нет
                        if source_count == 0:
                            return """
                            **Status**: Created but empty
                            
                            **Documents**: 0
                            
                            **Last updated**: Unknown
                            
                            Please rebuild the knowledge base using the button on the left.
                            """
                        
                        # Получаем файл с датой последнего обновления
                        last_updated = "Unknown"
                        try:
                            from src.knowledge_base.dataset import DatasetManager
                            dataset = DatasetManager()
                            last_updated = dataset.get_last_update_date() or "Unknown"
                        except Exception as e:
                            logger.error(f"Error getting last update date: {str(e)}")
                        
                        return f"""
                        **Status**: Active
                        
                        **Documents**: {doc_count}
                        
                        **Sources**: {source_count}
                        
                        **Last updated**: {last_updated}
                        """
                        
                    except Exception as e:
                        return f"""
                        **Status**: Error
                        
                        **Details**: {str(e)}
                        
                        Please try rebuilding the knowledge base.
                        """
                
                kb_info = gr.Markdown(value=get_kb_info())
                refresh_kb_info_btn = gr.Button("Refresh Information")

    # 3. Добавим обработчики событий для кнопок в конце файла
    # Добавьте эти обработчики перед строкой "if __name__ == "__main__":"

        # Обработчики для Knowledge Base
        update_kb_btn.click(
            fn=update_kb_with_selected,
            inputs=[sources_list],
            outputs=[kb_status]
        )
        
        rebuild_kb_btn.click(
            fn=rebuild_kb_with_selected,
            inputs=[sources_list],
            outputs=[kb_status]
        )
        
        # Обновление информации о базе знаний
        refresh_kb_info_btn.click(
            fn=get_kb_info,
            inputs=[],
            outputs=[kb_info]
        )
        
        # Автоматическое обновление информации после операций с базой знаний
        update_kb_btn.click(
            fn=get_kb_info,
            inputs=[],
            outputs=[kb_info]
        )
        
        rebuild_kb_btn.click(
            fn=get_kb_info,
            inputs=[],
            outputs=[kb_info]
        )

        with gr.Tab("Model Settings"):
            gr.Markdown("### Model Configuration")
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Add model selector
                    model_selector = gr.Dropdown(
                        choices=list(MODELS.keys()),
                        value=selected_model,  # Use loaded model from preferences
                        label="Select Model",
                        interactive=True
                    )
                    
                    # Current model info display
                    model_info = gr.Markdown(value=update_model_info(selected_model))
                    
                    # Status indicator for model loading
                    model_loading = gr.Textbox(
                        label="Status",
                        placeholder="Model ready",
                        interactive=False,
                        value="Model ready"
                    )
                    
                    # Model Parameters - make them interactive
                    with gr.Row():
                        max_length = gr.Slider(
                            minimum=1,
                            maximum=4096,
                            value=ACTIVE_MODEL['parameters']['max_length'],
                            step=1,
                            label="Maximum Length",
                            interactive=True
                        )
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=ACTIVE_MODEL['parameters']['temperature'],
                            step=0.1,
                            label="Temperature",
                            interactive=True
                        )
                    with gr.Row():
                        top_p = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=ACTIVE_MODEL['parameters']['top_p'],
                            step=0.1,
                            label="Top-p",
                            interactive=True
                        )
                        rep_penalty = gr.Slider(
                            minimum=1.0,
                            maximum=2.0,
                            value=ACTIVE_MODEL['parameters']['repetition_penalty'],
                            step=0.1,
                            label="Repetition Penalty",
                            interactive=True
                        )
                    
                    # Button to save parameters
                    save_params_btn = gr.Button("Save Parameters", variant="primary")

                    gr.Markdown("""
                    <small>
                    **Parameters explanation:**
                    - **Maximum Length**: Maximum number of tokens in the generated response
                    - **Temperature**: Controls randomness (0.1 = very focused, 2.0 = very creative)
                    - **Top-p**: Controls diversity via nucleus sampling (lower = more focused)
                    - **Repetition Penalty**: Prevents word repetition (higher = less repetition)
                    </small>
                    """)
                
                with gr.Column(scale=1):
                    # Model details panel
                    model_details = gr.HTML(get_model_details_html(selected_model))
                    
                    gr.Markdown("### Training Configuration")
                    gr.Markdown(f"""
                    **Base Model Path:** 
                    ```
                    {ACTIVE_MODEL['training']['base_model_path']}
                    ```
                    
                    **Fine-tuned Model Path:**
                    ```
                    {ACTIVE_MODEL['training']['fine_tuned_path']}
                    ```
                    
                    **LoRA Configuration:**
                    - Rank (r): {ACTIVE_MODEL['training']['lora_config']['r']}
                    - Alpha: {ACTIVE_MODEL['training']['lora_config']['lora_alpha']}
                    - Dropout: {ACTIVE_MODEL['training']['lora_config']['lora_dropout']}
                    """)

        with gr.Tab("Model Training"):
            gr.Markdown("### Model Training Interface")
            
            with gr.Row():
                with gr.Column(scale=1):
                    training_tabs = gr.Tabs()
                    
                    with training_tabs:
                        with gr.TabItem("Regular Training"):
                            epochs = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Number of Epochs")
                            batch_size = gr.Slider(minimum=1, maximum=32, value=4, step=1, label="Batch Size")
                            learning_rate = gr.Slider(minimum=1e-6, maximum=1e-3, value=2e-4, label="Learning Rate")
                            
                            train_btn = gr.Button("Start Training", variant="primary")
                            training_output = gr.Textbox(label="Training Status", interactive=False)
                        
                        with gr.TabItem("Train from Annotations"):
                            annot_epochs = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Number of Epochs")
                            annot_batch_size = gr.Slider(minimum=1, maximum=32, value=4, step=1, label="Batch Size")
                            annot_learning_rate = gr.Slider(minimum=1e-6, maximum=1e-3, value=2e-4, label="Learning Rate")
                            annot_min_rating = gr.Slider(minimum=1, maximum=5, value=4, step=0.5, label="Minimum Rating for Training")
                            
                            annot_train_btn = gr.Button("Start Training from Annotations", variant="primary")
                            annot_training_output = gr.Textbox(label="Training Status", interactive=False)
                    
                    gr.Markdown("""
                    <small>
                                        
                    **Epochs:**   
                    Lower = Faster training  -> Higher = Model learns more thoroughly   
                    Best for small datasets: 3-5 -> Best for large datasets: 1-2
                                                           
                    **Batch Size:**  
                    Lower = Slower but more stable -> Higher = Faster but needs more RAM   
                    4 = Good for 16GB RAM -> 8 = Good for 32GB RAM
                            
                    **Learning Rate:**   
                    Lower = Learns slower but more reliable -> Higher = Learns faster but may be unstable   
                    2e-4 (0.0002) = Usually works best -> 1e-4 = Safer choice for fine-tuning
                    </small>
                    """)

                with gr.Column(scale=1):
                    analysis_btn = gr.Button("Generate Chat Analysis")
                    analysis_output = gr.Markdown()
            
            train_btn.click(
                start_finetune_action,
                inputs=[epochs, batch_size, learning_rate],
                outputs=[training_output]
            )
            
            # Function to handle training from annotations
            def start_annotation_finetune(epochs, batch_size, learning_rate, min_rating):
                """Wrapper function to start fine-tuning from annotations"""
                success, message = finetune_from_annotations(
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    min_rating=min_rating
                )
                return message
            
            annot_train_btn.click(
                start_annotation_finetune,
                inputs=[annot_epochs, annot_batch_size, annot_learning_rate, annot_min_rating],
                outputs=[annot_training_output]
            )
            
            analysis_btn.click(
                generate_chat_analysis,
                inputs=[],
                outputs=[analysis_output]
            )

        with gr.Tab("Chat Evaluation"):
            gr.Markdown("### Evaluation of Chat Responses")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Status section
                    evaluation_status = gr.Textbox(
                        label="Evaluation Status",
                        interactive=False,
                        show_label=True
                    )
                    refresh_status_btn = gr.Button("Refresh Status and Chat History")
                    
                    # Moved refresh status and evaluation report here
                    refresh_data_status = gr.Textbox(
                        label="Refresh Status",
                        interactive=False,
                        show_label=True
                    )
                    
                    evaluation_report = gr.HTML(label="Evaluation Report")
                    refresh_report_btn = gr.Button("Generate Report")
                    
                    # QA pairs table section
                    show_evaluated = gr.Checkbox(
                        label="Show Only Evaluated Pairs",
                        value=False
                    )
                    qa_table = gr.Dataframe(
                        value=pd.DataFrame(
                            columns=["Conversation ID", "Question", "Answer", "Evaluated"]
                        ),
                        interactive=True,
                        wrap=True,
                        row_count=15,  # Changed from height to row_count
                        show_label=True
                    )
                    
                    # Conversation selection section
                    gr.Markdown("### Select Conversation to Evaluate")
                    with gr.Row():
                        selected_conversation = gr.Textbox(
                            label="Conversation ID", 
                            placeholder="Select from table above",
                            interactive=True
                        )
                        load_btn = gr.Button("Load Conversation")
                        delete_btn = gr.Button("Delete Conversation", variant="stop")
                        
                        delete_status = gr.Textbox(label="Delete Status", interactive=False)
                    
                    # Conversation content section
                    gr.Markdown("### Evaluate Response")
                    question_display = gr.Textbox(label="User Question", interactive=False)
                    original_answer = gr.TextArea(label="Original Bot Answer", interactive=False)
                    improved_answer = gr.TextArea(label="Improved Answer (Gold Standard)", interactive=True)
                    
                    # Ratings section
                    gr.Markdown("### Quality Ratings (1-5)")
                    with gr.Row():
                        accuracy = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Factual Accuracy")
                        completeness = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Completeness")
                    with gr.Row():
                        relevance = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Relevance")
                        clarity = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Clarity")
                    legal_correctness = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Legal Correctness")
                    
                    # Notes and save section
                    notes = gr.TextArea(label="Evaluator Notes", placeholder="Add your notes about this response...")
                    save_btn = gr.Button("Save Evaluation", variant="primary")
                    evaluation_status_msg = gr.Textbox(label="Status", interactive=False)
                    
                    # Data export section
                    gr.Markdown("### Export Evaluation Data")
                    with gr.Row():
                        min_rating = gr.Slider(minimum=1, maximum=5, value=4, step=0.5, label="Minimum Rating for Export")
                        export_path = gr.Textbox(label="Export File Path", value="training_data.jsonl")
                    export_btn = gr.Button("Export Training Data")
                    export_status = gr.Textbox(label="Export Status", interactive=False)
            
            # Event handlers for Chat Evaluation
            refresh_status_btn.click(
                fn=lambda: get_evaluation_status(chat_evaluator, force_reload=True),
                inputs=[],
                outputs=[evaluation_status, qa_table, refresh_data_status]
            )
            
            refresh_report_btn.click(
                fn=lambda: generate_evaluation_report_html(chat_evaluator),
                inputs=[],
                outputs=[evaluation_report]
            )
            
            show_evaluated.change(
                fn=lambda x: get_qa_pairs_dataframe(chat_evaluator, x),
                inputs=[show_evaluated],
                outputs=[qa_table]
            )
            
            def on_table_select(evt: gr.SelectData, dataframe):
                """Handle table row selection using the dataframe input"""
                try:
                    # Get the selected row index
                    row_index = evt.index[0]
                    
                    # Access the dataframe passed as input parameter
                    if dataframe is not None and len(dataframe) > row_index:
                        # Get conversation ID from first column
                        conversation_id = str(dataframe.iloc[row_index, 0])
                        logger.info(f"Selected conversation ID: {conversation_id}")
                        return conversation_id
                    else:
                        logger.error("DataFrame is empty or row index out of bounds")
                        return ""
                except Exception as e:
                    logger.error(f"Error in table selection: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return ""

            # Update the table row selection handler to include the dataframe as input
            qa_table.select(
                fn=on_table_select,
                inputs=[qa_table],  # Pass the table itself as input
                outputs=[selected_conversation]
            )
            # Load conversation for evaluation
            load_btn.click(
                fn=lambda x: load_qa_pair_for_evaluation(conversation_id=x, evaluator=chat_evaluator),
                inputs=[selected_conversation],
                outputs=[question_display, original_answer, improved_answer, 
                        accuracy, completeness, relevance, clarity, legal_correctness, notes]
            )
            
            # Save evaluation
            save_btn.click(
                fn=lambda conv_id, q, orig_a, imp_a, acc, comp, rel, clar, legal, notes: 
                    save_evaluation(conv_id, q, orig_a, imp_a, acc, comp, rel, clar, legal, notes, evaluator=chat_evaluator),
                inputs=[
                    selected_conversation, question_display, original_answer, improved_answer,
                    accuracy, completeness, relevance, clarity, legal_correctness, notes
                ],
                outputs=[evaluation_status_msg]
            )
            
            # Export training data
            export_btn.click(
                fn=lambda min_r, path: export_training_data_action(chat_evaluator, min_r, path),
                inputs=[min_rating, export_path],
                outputs=[export_status]
            )
            
            # Обработчик для удаления чата
            delete_btn.click(
                fn=delete_conversation,
                inputs=[selected_conversation, evaluator_state],
                outputs=[delete_status]
            )

            # Обновление таблицы и статуса после удаления
            delete_btn.click(
                fn=lambda: get_evaluation_status(chat_evaluator, force_reload=True),
                inputs=[],
                outputs=[evaluation_status, qa_table, refresh_data_status]
            )

    # Model change handler - outside of Tabs but inside Blocks
    model_selector.change(
        fn=change_model,
        inputs=[model_selector],
        outputs=[model_info, max_length, temperature, top_p, rep_penalty, model_loading]
    )
    
    # Update model details panel when changing model
    model_selector.change(
        fn=get_model_details_html,
        inputs=[model_selector],
        outputs=[model_details]
    )
    
    # Parameter save handler
    save_params_btn.click(
        fn=save_parameters,
        inputs=[model_selector, max_length, temperature, top_p, rep_penalty],
        outputs=[model_loading]
    )

# Launch application
if __name__ == "__main__":
    # Проверяем knowledge base
    if not load_vector_store():
        logger.warning("Knowledge base not found. Please create it through the interface.")
    
    demo.launch(share=True)

# Add helper functions for URL selection:
def get_selected_urls(sources_df):
    """Get list of URLs selected for inclusion"""
    try:
        if not isinstance(sources_df, pd.DataFrame):
            sources_df = pd.DataFrame(sources_df)
        
        selected_urls = sources_df[sources_df["Include"] == True]["URL"].tolist()
        return selected_urls
    except Exception as e:
        logger.error(f"Error getting selected URLs: {str(e)}")
        return []

def update_kb_with_selected(sources_df):
    """Update knowledge base using only selected URLs"""
    try:
        selected_urls = get_selected_urls(sources_df)
        
        if not selected_urls:
            return "Error: No URLs selected for inclusion"
        
        from config import constants
        original_urls = constants.URLS
        constants.URLS = selected_urls
        
        try:
            success, message = create_vector_store(mode="update")
            
            if success:
                metadata = {
                    "last_updated": datetime.datetime.now().isoformat(),
                    "source_count": len(selected_urls),
                    "sources": selected_urls
                }
                
                json_content = json.dumps(metadata, indent=2).encode('utf-8')
                api = HfApi(token=HF_TOKEN)
                
                api.upload_file(
                    path_or_fileobj=json_content,
                    path_in_repo="vector_store/metadata.json",
                    repo_id=DATASET_ID,
                    repo_type="dataset"
                )
            
            return message
        finally:
            constants.URLS = original_urls
            
    except Exception as e:
        return f"Error updating knowledge base: {str(e)}"

def rebuild_kb_with_selected(sources_df):
    """Rebuild knowledge base from scratch using only selected URLs"""
    try:
        selected_urls = get_selected_urls(sources_df)
        
        if not selected_urls:
            return "Error: No URLs selected for inclusion"
        
        from config import constants
        original_urls = constants.URLS
        constants.URLS = selected_urls
        
        try:
            success, message = create_vector_store(mode="rebuild")
            
            if success:
                metadata = {
                    "last_updated": datetime.datetime.now().isoformat(),
                    "source_count": len(selected_urls),
                    "sources": selected_urls
                }
                
                json_content = json.dumps(metadata, indent=2).encode('utf-8')
                api = HfApi(token=HF_TOKEN)
                
                api.upload_file(
                    path_or_fileobj=json_content,
                    path_in_repo="vector_store/metadata.json",
                    repo_id=DATASET_ID,
                    repo_type="dataset"
                )
            
            return message
        finally:
            constants.URLS = original_urls
            
    except Exception as e:
        return f"Error rebuilding knowledge base: {str(e)}"

# Add new function for source status updates
def update_source_status(df):
    """Update status column based on Include selection"""
    try:
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        
        df["Status"] = df["Include"].apply(lambda x: "Selected" if x else "Excluded")
        selected_count = df["Include"].sum()
        
        return df, f"{selected_count} URLs selected for inclusion"
    except Exception as e:
        return df, f"Error updating status: {str(e)}"

# Update event handlers in the Knowledge Base tab section
with gr.Tab("Knowledge Base"):
    gr.Markdown("### Knowledge Base Management")
    
    with gr.Row():
        with gr.Column(scale=2):
            # Sources list with selection
            gr.Markdown("#### Information Sources")
            sources_list = gr.Dataframe(
                value=pd.DataFrame({
                    "URL": URLS,
                    "Include": [True for _ in URLS],
                    "Status": ["Ready" for _ in URLS]
                }),
                interactive=True,
                wrap=True,
                row_count=15,
                show_label=False
            )
            
            # Status display
            kb_status = gr.Textbox(
                label="Operation Status",
                interactive=False,
                placeholder="Ready",
                value="Ready"
            )
            
            # Control buttons
            with gr.Row():
                update_kb_btn = gr.Button("Update Knowledge Base", variant="primary")
                rebuild_kb_btn = gr.Button("Rebuild Knowledge Base from Scratch", variant="secondary")
            
            # Help text
            gr.Markdown("""
            <small>
            **Update Knowledge Base**: Adds new information to the existing knowledge base.
            
            **Rebuild Knowledge Base**: Recreates the entire knowledge base from scratch. Use this if there are inconsistencies.
            
            All changes are saved to the Hugging Face dataset.
            </small>
            """)
        
        with gr.Column(scale=1):
            # Knowledge base info display
            gr.Markdown("#### Knowledge Base Information")
            kb_info = gr.Markdown(value=get_kb_info())
            refresh_kb_info_btn = gr.Button("Refresh Information")

    # Event handlers for Knowledge Base operations
    update_kb_btn.click(
        fn=update_kb_with_selected,
        inputs=[sources_list],
        outputs=[kb_status]
    )
    
    rebuild_kb_btn.click(
        fn=rebuild_kb_with_selected,
        inputs=[sources_list],
        outputs=[kb_status]
    )
    
    # Auto-refresh knowledge base info after operations
    update_kb_btn.click(
        fn=get_kb_info,
        inputs=[],
        outputs=[kb_info]
    )
    
    rebuild_kb_btn.click(
        fn=get_kb_info,
        inputs=[],
        outputs=[kb_info]
    )
    
    # Refresh button handler
    refresh_kb_info_btn.click(
        fn=get_kb_info,
        inputs=[],
        outputs=[kb_info]
    )
    
    # Source selection status update handler
    sources_list.change(
        fn=update_source_status,
        inputs=[sources_list],
        outputs=[sources_list, kb_status]
    )

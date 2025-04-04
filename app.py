import gradio as gr
import os
import json
import datetime
from pathlib import Path
from src.analytics.chat_evaluator import ChatEvaluator
import sys
import logging
from langdetect import detect
from huggingface_hub import InferenceClient, HfApi
from config.constants import DEFAULT_SYSTEM_MESSAGE
from config.settings import (
    HF_TOKEN, 
    MODELS,
    ACTIVE_MODEL,
    EMBEDDING_MODEL,
    DATASET_ID,
    CHAT_HISTORY_PATH,
    VECTOR_STORE_PATH,
    DEFAULT_MODEL,
    API_CONFIG 
)
from src.knowledge_base.vector_store import create_vector_store, load_vector_store
from web.training_interface import (
    get_models_df,
    generate_chat_analysis,
    register_model_action,
    start_finetune_action
)
from web.evaluation_interface import (
    get_evaluation_status,
    get_qa_pairs_dataframe,
    load_qa_pair_for_evaluation,
    save_evaluation,
    generate_evaluation_report_html,
    export_training_data_action
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")

# Path for user preferences file
USER_PREFERENCES_PATH = os.path.join(os.path.dirname(__file__), "user_preferences.json")
ERROR_LOGS_PATH = os.path.join(os.path.dirname(__file__), "error_logs")

# Global variables
client = None
context_store = {}
fallback_model_attempted = False
chat_evaluator = ChatEvaluator(
    hf_token=HF_TOKEN,
    dataset_id=DATASET_ID
)

logger.info(f"Chat histories will be saved to: {CHAT_HISTORY_PATH}")

def load_user_preferences():
    """Load user preferences from file"""
    try:
        if os.path.exists(USER_PREFERENCES_PATH):
            with open(USER_PREFERENCES_PATH, 'r') as f:
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
    """Save user preferences to file"""
    try:
        preferences = load_user_preferences()
        preferences["selected_model"] = model_key
        
        # Update parameters if provided
        if parameters:
            if model_key not in preferences["parameters"]:
                preferences["parameters"][model_key] = {}
            
            preferences["parameters"][model_key] = parameters
        
        with open(USER_PREFERENCES_PATH, 'w') as f:
            json.dump(preferences, f, indent=2)
        
        logger.info("User preferences saved successfully!")
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
    """Translate text using the active LLM"""
    try:
        prompt = f"Translate this text to {target_lang}. Provide ONLY the translation, without any explanations or meta information:\n\n{text}"
        
        response = client.chat_completion(
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=ACTIVE_MODEL['parameters']['max_length'],
            temperature=0.3,
            top_p=0.9,
            stream=False
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return text

def post_process_response(user_message, bot_response):
    """Check if the response language matches the user's language and translate if needed"""
    try:
        user_lang = detect_language(user_message)
        bot_lang = detect_language(bot_response)
        
        if user_lang != bot_lang and len(bot_response.strip()) > 20:
            logger.warning(f"Language mismatch detected! User: {user_lang}, Bot: {bot_lang}")
            
            translated_response = translate_with_llm(bot_response, user_lang)
            translated_lang = detect_language(translated_response)
            
            if translated_lang == user_lang:
                logger.info(f"Response automatically translated from {bot_lang} to {user_lang}")
                return translated_response
            else:
                logger.error(f"Translation failed: got {translated_lang} instead of {user_lang}")
            
        return bot_response
        
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
    """Detect language with fallback"""
    try:
        if len(text.strip()) < 5:
            logger.debug(f"Text too short for reliable detection: '{text}'")
            return "en"
            
        return detect(text.strip())
        
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
    """Generate response with proper error handling"""
    try:
        # Reset and determine user language for new request
        user_lang = detect_language(message)
        logger.debug(f"Detected user language: {user_lang}")
        
        # Create clean history without system messages
        clean_history = [
            msg for msg in history 
            if msg["role"] != "system"
        ]
        
        # Add fresh system message with current language instruction
        language_instruction = f"\nIMPORTANT: You MUST respond in {user_lang} language ONLY."
        full_system_message = system_message + language_instruction
        
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
        
        # Post-process response to check language
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
    """Log API errors to a separate file for monitoring"""
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
        success, message = create_vector_store(mode="update")
        return message
    except Exception as e:
        return f"Error updating knowledge base: {str(e)}"

def rebuild_kb():
    """Function to create knowledge base from scratch"""
    try:
        success, message = create_vector_store(mode="rebuild")
        return message
    except Exception as e:
        return f"Error creating knowledge base: {str(e)}"

def save_chat_history(history, conversation_id):
    """Save chat history to a file and to HuggingFace dataset"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(CHAT_HISTORY_PATH, exist_ok=True)
        
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
        filepath = os.path.join(CHAT_HISTORY_PATH, filename)
        
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
            
            # Extract just the directory name from CHAT_HISTORY_PATH
            dir_name = os.path.basename(CHAT_HISTORY_PATH)
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

def respond_and_clear(message, history, conversation_id):
    """Wrapper function with proper output handling"""
    try:
        # Get current model parameters
        params = ACTIVE_MODEL['parameters']
        
        # Call respond function
        result = respond(
            message=message,
            history=history if history else [],
            conversation_id=conversation_id,
            system_message=DEFAULT_SYSTEM_MESSAGE,
            max_tokens=params['max_length'],
            temperature=params['temperature'],
            top_p=params['top_p']
        )
        
        if not result:
            raise ValueError("Empty response from API")
            
        new_history, new_conv_id = result
        
        # Save chat history
        save_chat_history(new_history, new_conv_id)
        
        return new_history, new_conv_id, ""  # Clear input
        
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
    
    logger.info(f"App initialized with model: {ACTIVE_MODEL['name']}")
    return selected_model

def initialize_chat_evaluator():
    """Initialize chat evaluator with proper paths"""
    try:
        evaluator = ChatEvaluator(
            hf_token=HF_TOKEN,
            dataset_id=DATASET_ID,
            chat_history_path=CHAT_HISTORY_PATH,
            annotations_dir=os.path.join(CHAT_HISTORY_PATH, 'evaluations')
        )
        
        # Проверим наличие директорий
        os.makedirs(CHAT_HISTORY_PATH, exist_ok=True)
        os.makedirs(os.path.join(CHAT_HISTORY_PATH, 'evaluations'), exist_ok=True)
        
        logger.debug(f"Chat history path: {CHAT_HISTORY_PATH}")
        logger.debug(f"Number of chat files: {len(os.listdir(CHAT_HISTORY_PATH))}")
        
        return evaluator
    except Exception as e:
        logger.error(f"Error initializing chat evaluator: {str(e)}")
        raise

# Initialize HF client with token at startup
selected_model = initialize_app()

# Create interface
with gr.Blocks() as demo:
    # Define clear_conversation function within the block for component access
    def clear_conversation():
        """Clear conversation and save history before clearing"""
        return [], None  # Just return empty values
    
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

            # Add event handlers
            submit_btn.click(
                respond_and_clear,
                [msg, chatbot, conversation_id],
                [chatbot, conversation_id, msg]
            )
            # Add Enter key handler
            msg.submit(
                respond_and_clear,
                [msg, chatbot, conversation_id],
                [chatbot, conversation_id, msg]
            )
            update_kb_btn.click(update_kb, None, kb_status)
            rebuild_kb_btn.click(rebuild_kb, None, kb_status)
            
            clear_btn.click(clear_conversation, None, [chatbot, conversation_id])

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
                with gr.Column(scale=2):
                    # Status and reports section
                    with gr.Row():
                        with gr.Column(scale=1):
                            evaluation_status = gr.Textbox(label="Evaluation Status", interactive=False)
                            refresh_status_btn = gr.Button("Refresh Status")
                        
                        with gr.Column(scale=1):
                            evaluation_report = gr.HTML(label="Evaluation Report")
                            refresh_report_btn = gr.Button("Generate Report")
                    
                    # QA pairs table section
                    show_evaluated = gr.Checkbox(label="Show Already Evaluated Pairs", value=False)
                    qa_table = gr.DataFrame(
                        get_qa_pairs_dataframe(chat_evaluator),
                        interactive=True,
                        wrap=True
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
                fn=lambda: get_evaluation_status(chat_evaluator),
                inputs=[],
                outputs=[evaluation_status]
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
            
            # Обработчик выбора строки в таблице
            def on_table_select(evt: gr.SelectData) -> str:
                """Handle table row selection"""
                try:
                    # evt.value содержит данные выбранной строки
                    # Возвращаем conversation_id из первой колонки
                    return evt.value[0]
                except Exception as e:
                    logger.error(f"Error in table selection: {str(e)}")
                    return ""

            # Table row selection handler
            qa_table.select(
                fn=on_table_select,
                inputs=[],  # Не нужны входные данные
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
                fn=lambda min_r, path: export_training_data_action(min_r, path, chat_evaluator),
                inputs=[min_rating, export_path],
                outputs=[export_status]
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
    # Create error logs directory
    os.makedirs(ERROR_LOGS_PATH, exist_ok=True)
    
    # Check knowledge base availability in dataset
    if not load_vector_store():
        logger.warning("Knowledge base not found. Please create it through the interface.")
    
    demo.launch(share=True)

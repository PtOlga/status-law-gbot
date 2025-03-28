import gradio as gr
import os
import json
import datetime
from pathlib import Path
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
    DEFAULT_MODEL
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
from src.analytics.chat_evaluator import ChatEvaluator

if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")

# Enhanced model details for UI
MODEL_DETAILS = {
    "llama-7b": {
        "full_name": "Meta Llama 2 7B Chat",
        "capabilities": [
            "Multilingual support ",
            "Good performance on legal texts",
            "Free model with open license",
            "Can run on computers with 16GB+ RAM"
        ],
        "limitations": [
            "Limited knowledge of specific legal terminology",
            "May provide incorrect answers to complex legal questions",
            "Knowledge is limited to training data"
        ],
        "use_cases": [
            "Legal document analysis",
            "Answering general legal questions",
            "Searching through legal knowledge base",
            "Assistance in document drafting"
        ],
        "documentation": "https://huggingface.co/meta-llama/Llama-2-7b-chat-hf"
    },
    "zephyr-7b": {
        "full_name": "HuggingFaceH4 Zephyr 7B Beta",
        "capabilities": [
            "High performance on instruction-following tasks",
            "Good response accuracy",
            "Advanced reasoning capabilities",
            "Excellent text generation quality"
        ],
        "limitations": [
            "May require paid API for usage",
            "Limited support for languages other than English",
            "Less optimization for legal topics compared to specialized models"
        ],
        "use_cases": [
            "Complex legal reasoning",
            "Case analysis",
            "Legal research",
            "Structured legal text generation"
        ],
        "documentation": "https://huggingface.co/HuggingFaceH4/zephyr-7b-beta"
    }
}

# Path for user preferences file
USER_PREFERENCES_PATH = os.path.join(os.path.dirname(__file__), "user_preferences.json")
ERROR_LOGS_PATH = os.path.join(os.path.dirname(__file__), "error_logs")

# Global variables
client = None
context_store = {}
fallback_model_attempted = False
chat_evaluator = ChatEvaluator(
    hf_token=HF_TOKEN,
    dataset_id=DATASET_ID,
    chat_history_path=CHAT_HISTORY_PATH
)

print(f"Chat histories will be saved to: {CHAT_HISTORY_PATH}")

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
        print(f"Error loading user preferences: {str(e)}")
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
        
        print(f"User preferences saved successfully!")
        return True
    except Exception as e:
        print(f"Error saving user preferences: {str(e)}")
        return False

def initialize_client(model_id=None):
    """Initialize or reinitialize the client with the specified model"""
    global client
    if model_id is None:
        model_id = ACTIVE_MODEL["id"]
    
    client = InferenceClient(
        model_id,
        token=HF_TOKEN
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
        
        print(f"Switched to model: {model_key}")
        return True
    except Exception as e:
        print(f"Error switching to model {model_key}: {str(e)}")
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
        print("Knowledge base not found or failed to load")
        return ""
    
    # Check if vector_store is a string (error message) instead of an actual store
    if isinstance(vector_store, str):
        print(f"Error with vector store: {vector_store}")
        return ""
    
    try:
        # Extract context
        context_docs = vector_store.similarity_search(message, k=3)
        context_text = "\n\n".join([f"From {doc.metadata.get('source', 'unknown')}: {doc.page_content}" for doc in context_docs])
        
        # Save context for this conversation
        context_store[conversation_id] = context_text
        
        return context_text
    except Exception as e:
        print(f"Error getting context: {str(e)}")
        return ""

def load_vector_store():
    """Load knowledge base from dataset"""
    try:
        from src.knowledge_base.dataset import DatasetManager
        
        print("Debug - Attempting to load vector store...")
        dataset = DatasetManager()
        success, result = dataset.download_vector_store()
        
        print(f"Debug - Download result: success={success}, result_type={type(result)}")
        
        if success:
            if isinstance(result, str):
                print(f"Debug - Error message received: {result}")
                return None
            return result
        else:
            print(f"Debug - Failed to load vector store: {result}")
            return None
            
    except Exception as e:
        import traceback
        print(f"Exception loading knowledge base: {str(e)}")
        print(traceback.format_exc())
        return None

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
    """Generate response using the current model with fallback option"""
    global fallback_model_attempted
    
    # Create ID for new conversation
    if not conversation_id:
        import uuid
        conversation_id = str(uuid.uuid4())
    
    # Get context from knowledge base
    context = get_context(message, conversation_id)
    
    # Convert history from Gradio format to OpenAI format
    messages = [{"role": "system", "content": system_message}]
    if context:
        messages[0]["content"] += f"\n\nContext for response:\n{context}"
    
    # Debug: print the history format
    print("Debug - Processing history format:", history)
    
    # Convert history to OpenAI format for API call
    if history:
        try:
            for entry in history:
                # Check if we have messages in the expected format
                if isinstance(entry, dict) and 'role' in entry and 'content' in entry:
                    messages.append(entry)
        except Exception as e:
            print(f"Error processing history: {str(e)}")
            # Continue with empty history if there was an error
    
    # Add current user message
    messages.append({"role": "user", "content": message})
    
    # Debug: print API messages
    print("Debug - API messages:", messages)
    
    try:
        # Non-streaming version for debugging
        full_response = client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=False,
            temperature=temperature,
            top_p=top_p,
        )
        
        response = full_response.choices[0].message.content
        print(f"Debug - Full response from API: {response}")
        
        # Reset fallback flag on successful API call
        fallback_model_attempted = False
        
        # Return complete response in the new format
        final_history = history.copy() if history else []
        # Add user message
        final_history.append({"role": "user", "content": message})
        # Add assistant response
        final_history.append({"role": "assistant", "content": response})
        
        yield final_history, conversation_id
            
    except Exception as e:
        print(f"Debug - Error during API call: {str(e)}")
        error_message = str(e)
        current_model_key = None
        
        # Find current model key
        for key, model in MODELS.items():
            if model["id"] == ACTIVE_MODEL["id"]:
                current_model_key = key
                break
        
        # Try fallback model if appropriate
        if attempt_fallback and ("402" in error_message or "429" in error_message) and not fallback_model_attempted:
            fallback_model_key = get_fallback_model(current_model_key)
            if fallback_model_key:
                fallback_model_attempted = True
                
                # Log fallback attempt
                print(f"Attempting to fallback from {current_model_key} to {fallback_model_key}")
                log_api_error(message, error_message, ACTIVE_MODEL["id"], is_fallback=True)
                
                # Switch model temporarily
                original_model = ACTIVE_MODEL.copy()
                if switch_to_model(fallback_model_key):
                    # Try with fallback model (but don't fallback again)
                    fallback_generator = respond(
                        message, 
                        history, 
                        conversation_id, 
                        system_message, 
                        max_tokens, 
                        temperature, 
                        top_p,
                        attempt_fallback=False
                    )
                    
                    yield from fallback_generator
                    
                    # Restore original model
                    ACTIVE_MODEL.update(original_model)
                    initialize_client(ACTIVE_MODEL["id"])
                    return
        
        # Format user-friendly error message
        if "402" in error_message and "Payment Required" in error_message:
            friendly_error = (
                "⚠️ API Error: Free request limit exceeded for this model.\n\n"
                "Solutions:\n"
                "1. Switch to another model in the 'Model Settings' tab\n"
                "2. Use a local model version\n"
                "3. Subscribe to Hugging Face PRO for higher limits"
            )
        elif "401" in error_message and "Unauthorized" in error_message:
            friendly_error = (
                "⚠️ API Error: Authentication problem. Please check your API key."
            )
        elif "429" in error_message and "Too Many Requests" in error_message:
            friendly_error = (
                "⚠️ API Error: Too many requests. Please try again later."
            )
        else:
            friendly_error = f"⚠️ API Error: There was an error accessing the model. Details: {error_message}"
        
        # Log the error
        log_api_error(message, error_message, ACTIVE_MODEL["id"])
        
        error_history = history.copy() if history else []
        # Add user message
        error_history.append({"role": "user", "content": message})
        # Add error message as assistant response
        error_history.append({"role": "assistant", "content": friendly_error})
        
        yield error_history, conversation_id

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
            
        print(f"API error logged to {log_path}")
    except Exception as e:
        print(f"Failed to log API error: {str(e)}")

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
        
        print(f"Debug - Chat history saved locally to {filepath}")
        
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
            
            print(f"Debug - Chat history uploaded to dataset at {target_path}")
            
        except Exception as e:
            print(f"Warning - Failed to upload chat history to dataset: {str(e)}")
            # Continue execution even if upload fails
        
        return True
    except Exception as e:
        print(f"Error saving chat history: {str(e)}")
        return False

def respond_and_clear(message, history, conversation_id):
    """Handle chat message and clear input"""
    # Get model parameters from config
    max_tokens = ACTIVE_MODEL['parameters']['max_length']
    temperature = ACTIVE_MODEL['parameters']['temperature']
    top_p = ACTIVE_MODEL['parameters']['top_p']
    
    # Print debug information to help diagnose the issue
    print("Debug - Message type:", type(message), "Content:", message)
    print("Debug - History type:", type(history), "Content:", history)
    
    try:
        # Get response generator
        response_generator = respond(
            message=message,
            history=history if history else [],
            conversation_id=conversation_id,
            system_message=DEFAULT_SYSTEM_MESSAGE,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # Get first response from generator
        new_history, conv_id = next(response_generator)
        
        # Debug the response
        print("Debug - Final history:", new_history)
        
        # Check if the history contains errors (by looking for error message pattern)
        last_message = new_history[-1] if new_history else None
        is_error = last_message and isinstance(last_message.get('content', ''), str) and "⚠️ API Error" in last_message.get('content', '')
        
        # Save chat history after response (even with errors)
        save_chat_history(new_history, conv_id)
        
        return new_history, conv_id, ""  # Clear message input
        
    except Exception as e:
        print(f"Error in respond_and_clear: {str(e)}")
        
        # Create a more readable error message
        if "incompatible with messages format" in str(e):
            error_message = (
                "⚠️ Message processing error: Problem with message format.\n\n"
                "Please try to clear the chat history using the 'Clear' button or "
                "switch to another model."
            )
        else:
            error_message = f"⚠️ Error: {str(e)}"
            
        # Create error history in the correct format
        error_history = history.copy() if history else []
        error_history.append({"role": "user", "content": message})
        error_history.append({"role": "assistant", "content": error_message})
        
        # Still try to save history with error
        if conversation_id:
            save_chat_history(error_history, conversation_id)
            
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
    if model_key not in MODEL_DETAILS:
        return "<p>Model information not available</p>"
    
    details = MODEL_DETAILS[model_key]
    
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
    
    print(f"App initialized with model: {ACTIVE_MODEL['name']}")
    return selected_model

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
                        clear_btn = gr.Button("Clear")  # Add clear button
                
                with gr.Column(scale=1):
                    gr.Markdown("### Knowledge Base Management")
                    gr.Markdown("""
                    - **Update**: Add new documents to existing base
                    - **Rebuild**: Create new base from scratch
                    """)
                    with gr.Row():
                        update_kb_btn = gr.Button("📝 Update Base", variant="secondary", scale=1)
                        rebuild_kb_btn = gr.Button("🔄 Rebuild Base", variant="primary", scale=1)
                    kb_status = gr.Textbox(
                        label="Status",
                        placeholder="Knowledge base status will appear here...",
                        interactive=False
                    )

            submit_btn.click(
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
                with gr.Column(scale=1):
                    evaluation_status = gr.Markdown(get_evaluation_status(chat_evaluator))
                    refresh_status_btn = gr.Button("Refresh Status")
                    
                    gr.Markdown("### Evaluation Metrics")
                    evaluation_report = gr.HTML(generate_evaluation_report_html(chat_evaluator))
                    refresh_report_btn = gr.Button("Refresh Report")
                    
                    gr.Markdown("### Export for Training")
                    with gr.Row():
                        min_rating = gr.Slider(
                            minimum=1, 
                            maximum=5, 
                            value=4, 
                            step=0.5, 
                            label="Minimum Average Rating"
                        )
                        export_path = gr.Textbox(
                            label="Export File Path (optional)",
                            placeholder="Leave empty for default path"
                        )
                    export_btn = gr.Button("Export Annotated Data", variant="primary")
                    export_status = gr.Textbox(label="Export Status", interactive=False)
                    
                with gr.Column(scale=2):
                    show_evaluated = gr.Checkbox(label="Show Already Evaluated Pairs", value=False)
                    qa_table = gr.DataFrame(get_qa_pairs_dataframe(chat_evaluator))
                    
                    gr.Markdown("### Select Conversation to Evaluate")
                    selected_conversation = gr.Textbox(label="Conversation ID", placeholder="Select from table above")
                    load_btn = gr.Button("Load Conversation", variant="primary")
                    
                    gr.Markdown("### Evaluate Response")
                    question_display = gr.Textbox(label="User Question", interactive=False)
                    original_answer = gr.TextArea(label="Original Bot Answer", interactive=False)
                    improved_answer = gr.TextArea(label="Improved Answer (Gold Standard)", interactive=True)
                    
                    gr.Markdown("### Quality Ratings (1-5)")
                    with gr.Row():
                        accuracy = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Factual Accuracy")
                        completeness = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Completeness")
                    with gr.Row():
                        relevance = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Relevance")
                        clarity = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Clarity")
                    legal_correctness = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Legal Correctness")
                    
                    notes = gr.TextArea(label="Evaluator Notes", placeholder="Add your notes about this response...")
                    save_btn = gr.Button("Save Evaluation", variant="primary")
                    evaluation_status_msg = gr.Textbox(label="Status", interactive=False)
            
            # Add event handlers
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
            
            # Table selection to conversation ID textbox
            qa_table.select(
                fn=lambda df, evt: evt.value[0] if evt and evt.value and len(evt.value) > 0 else "",
                inputs=[qa_table],
                outputs=[selected_conversation]
            )
            
            # Load conversation for evaluation
            load_btn.click(
                fn=lambda x: load_qa_pair_for_evaluation(x, chat_evaluator),
                inputs=[selected_conversation],
                outputs=[question_display, original_answer, improved_answer, 
                        accuracy, completeness, relevance, clarity, legal_correctness, notes]
            )
            
            # Save evaluation
            save_btn.click(
                fn=lambda *args: save_evaluation(*args, evaluator=chat_evaluator),
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
    
    # Model change handler
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
        print("Knowledge base not found. Please create it through the interface.")
    
    demo.launch()

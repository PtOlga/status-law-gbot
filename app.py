import gradio as gr
import os
import json
import datetime
from pathlib import Path
from huggingface_hub import InferenceClient
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

if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")

# Extended model information
MODEL_DETAILS = {
    "llama-7b": {
        "full_name": "Meta Llama 2 7B Chat",
        "capabilities": [
            "Multilingual (supports Russian, English and other languages)",
            "Good performance on legal texts",
            "Free model with open license",
            "Easy to run on computers with 16GB+ RAM"
        ],
        "limitations": [
            "Limited knowledge of specific legal terminology",
            "May give incorrect answers to complex legal questions",
            "Knowledge limited by training data"
        ],
        "use_cases": [
            "Legal document analysis",
            "Answering general legal questions",
            "Legal knowledge base search",
            "Document drafting assistance"
        ],
        "documentation": "https://huggingface.co/meta-llama/Llama-2-7b-chat-hf"
    },
    "zephyr-7b": {
        "full_name": "HuggingFaceH4 Zephyr 7B Beta",
        "capabilities": [
            "High performance on instruction tasks",
            "Good response accuracy",
            "Advanced reasoning",
            "Excellent text generation quality"
        ],
        "limitations": [
            "May require API payment for usage",
            "Limited support for languages other than English",
            "Less optimization for legal topics than specialized models"
        ],
        "use_cases": [
            "Complex legal reasoning",
            "Case law analysis",
            "Legislative research",
            "Structured legal text generation"
        ],
        "documentation": "https://huggingface.co/HuggingFaceH4/zephyr-7b-beta"
    }
}

# Путь к файлу с пользовательскими настройками
USER_PREFERENCES_PATH = os.path.join(os.path.dirname(__file__), "user_preferences.json")

# Глобальные переменные
client = None
context_store = {}

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
):
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
            for item in history:
                # Check if we have a pair of messages as expected
                if len(item) == 2:
                    user_msg, assistant_msg = item
                    
                    # Add user message
                    messages.append({"role": "user", "content": user_msg})
                    
                    # Add assistant message
                    messages.append({"role": "assistant", "content": assistant_msg})
        except Exception as e:
            print(f"Error processing history: {str(e)}")
            # Continue with empty history if there was an error
    
    # Add current user message
    messages.append({"role": "user", "content": message})
    
    # Debug: print API messages
    print("Debug - API messages:", messages)
    
    # Send API request and stream response
    response = ""
    is_complete = False
    
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
        
        # Return complete response immediately
        final_history = history.copy() if history else []
        final_history.append((message, response))
        yield final_history, conversation_id
            
    except Exception as e:
        print(f"Debug - Error during API call: {str(e)}")
        error_history = history.copy() if history else []
        error_history.append((message, f"An error occurred: {str(e)}"))
        yield error_history, conversation_id


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
    """Save chat history to a file"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(CHAT_HISTORY_PATH, exist_ok=True)
        
        # Format history for saving
        formatted_history = []
        for item in history:
            if len(item) == 2:
                user_msg, assistant_msg = item
                formatted_history.append({
                    "user": user_msg,
                    "assistant": assistant_msg,
                    "timestamp": datetime.datetime.now().isoformat()
                })
        
        # Create filename with conversation_id and timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{conversation_id}_{timestamp}.json"
        filepath = os.path.join(CHAT_HISTORY_PATH, filename)
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "conversation_id": conversation_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "history": formatted_history
            }, f, ensure_ascii=False, indent=2)
        
        print(f"Debug - Chat history saved to {filepath}")
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
        
        # Save chat history after response
        save_chat_history(new_history, conv_id)
        
        return new_history, conv_id, ""  # Clear message input
        
    except Exception as e:
        print(f"Error in respond_and_clear: {str(e)}")
        error_history = history + [(message, f"An error occurred: {str(e)}")]
        
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
        return "<p>Информация о модели недоступна</p>"
    
    details = MODEL_DETAILS[model_key]
    
    html = f"""
    <div style="padding: 15px; border: 1px solid #ccc; border-radius: 5px; margin-top: 10px;">
        <h3>{details['full_name']}</h3>
        
        <h4>Возможности:</h4>
        <ul>
            {"".join([f"<li>{cap}</li>" for cap in details['capabilities']])}
        </ul>
        
        <h4>Ограничения:</h4>
        <ul>
            {"".join([f"<li>{lim}</li>" for lim in details['limitations']])}
        </ul>
        
        <h4>Рекомендуемое использование:</h4>
        <ul>
            {"".join([f"<li>{use}</li>" for use in details['use_cases']])}
        </ul>
        
        <p><a href="{details['documentation']}" target="_blank">Документация модели</a></p>
    </div>
    """
    
    return html

def change_model(model_key):
    """Change active model and update parameters"""
    global client, ACTIVE_MODEL
    
    try:
        # Update active model
        ACTIVE_MODEL = MODELS[model_key]
        
        # Reinitialize client with new model
        client = InferenceClient(
            ACTIVE_MODEL["id"],
            token=HF_TOKEN
        )
        
        # Сохраняем выбранную модель в предпочтениях
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
        
        # Сохраняем параметры в предпочтениях
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

def initialize_app():
    """Initialize app with user preferences"""
    global client, ACTIVE_MODEL
    
    preferences = load_user_preferences()
    selected_model = preferences.get("selected_model", DEFAULT_MODEL)
    
    # Убедиться, что выбранная модель существует
    if selected_model not in MODELS:
        selected_model = DEFAULT_MODEL
    
    # Установить активную модель
    ACTIVE_MODEL = MODELS[selected_model]
    
    # Загрузить сохраненные параметры, если они есть
    saved_params = preferences.get("parameters", {}).get(selected_model)
    if saved_params:
        ACTIVE_MODEL['parameters'].update(saved_params)
    
    # Инициализировать клиент
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
    # Определяем функцию clear_conversation внутри блока для доступа к компонентам
    def clear_conversation():
        """Clear conversation and save history before clearing"""
        return [], None  # Просто возвращаем пустые значения
    
    with gr.Tabs():
        with gr.Tab("Chat"):
            gr.Markdown("# ⚖️ Status Law Assistant")
            
            conversation_id = gr.State(None)
            
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Chat",
                        avatar_images=None,
                        type='messages'
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
                    
                    # Save parameters button
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
                with gr.Column():
                    epochs = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Number of Epochs")
                    batch_size = gr.Slider(minimum=1, maximum=32, value=4, step=1, label="Batch Size")
                    learning_rate = gr.Slider(minimum=1e-6, maximum=1e-3, value=2e-4, label="Learning Rate")
                    train_btn = gr.Button("Start Training", variant="primary")
                    training_output = gr.Textbox(label="Training Status", interactive=False)

                with gr.Column():
                    analysis_btn = gr.Button("Generate Chat Analysis")
                    analysis_output = gr.Markdown()
            
            train_btn.click(
                start_finetune_action,
                inputs=[epochs, batch_size, learning_rate],
                outputs=[training_output]
            )
            analysis_btn.click(
                generate_chat_analysis,
                inputs=[],
                outputs=[analysis_output]
            )
    
    # Model change handler
    model_selector.change(
        fn=change_model,
        inputs=[model_selector],
        outputs=[model_info, max_length, temperature, top_p, rep_penalty, model_loading]
    )
    
    # Update model details panel when model changes
    model_selector.change(
        fn=get_model_details_html,
        inputs=[model_selector],
        outputs=[model_details]
    )
    
    # Parameters save handler
    save_params_btn.click(
        fn=save_parameters,
        inputs=[model_selector, max_length, temperature, top_p, rep_penalty],
        outputs=[model_loading]
    )

# Launch application
if __name__ == "__main__":
    # Check knowledge base availability in dataset
    if not load_vector_store():
        print("Knowledge base not found. Please create it through the interface.")
    
    demo.launch()

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

# Initialize HF client with token
client = InferenceClient(
    ACTIVE_MODEL["id"],
    token=HF_TOKEN
)

# State for storing context and chat history
context_store = {}

print(f"Chat histories will be saved to: {CHAT_HISTORY_PATH}")

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
    max_tokens = ACTIVE_MODEL['parameters']['max_length']  # используем ACTIVE_MODEL вместо MODEL_CONFIG
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

# Create interface
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.Tab("Chat"):
            gr.Markdown("# ⚖️ Status Law Assistant")
            
            conversation_id = gr.State(None)
            
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Chat",
                        avatar_images=None
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
            def clear_conversation():
                """Clear conversation and save history before clearing"""
                # Save current history if there's a conversation
                if chatbot and conversation_id:
                    save_chat_history(chatbot, conversation_id)
                return [], None
                
            clear_btn.click(clear_conversation, None, [chatbot, conversation_id])

        with gr.Tab("Model Settings"):
            gr.Markdown("### Model Configuration")
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Add model selector
                    model_selector = gr.Dropdown(
                        choices=list(MODELS.keys()),
                        value=DEFAULT_MODEL,
                        label="Select Model",
                        interactive=True
                    )
                    
                    # Current model info display
                    model_info = gr.Markdown()
                    
                    # Model Parameters
                    with gr.Row():
                        max_length = gr.Slider(
                            minimum=1,
                            maximum=4096,
                            value=ACTIVE_MODEL['parameters']['max_length'],
                            step=1,
                            label="Maximum Length",
                            interactive=False
                        )
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=ACTIVE_MODEL['parameters']['temperature'],
                            step=0.1,
                            label="Temperature",
                            interactive=False
                        )
                    with gr.Row():
                        top_p = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=ACTIVE_MODEL['parameters']['top_p'],
                            step=0.1,
                            label="Top-p",
                            interactive=False
                        )
                        rep_penalty = gr.Slider(
                            minimum=1.0,
                            maximum=2.0,
                            value=ACTIVE_MODEL['parameters']['repetition_penalty'],
                            step=0.1,
                            label="Repetition Penalty",
                            interactive=False
                        )

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
                    gr.Markdown("### Training Configuration")
                    gr.Markdown(f"""
                    **Base Model Path:** 
                    ```
                    {MODEL_CONFIG['training']['base_model_path']}
                    ```
                    
                    **Fine-tuned Model Path:**
                    ```
                    {MODEL_CONFIG['training']['fine_tuned_path']}
                    ```
                    
                    **LoRA Configuration:**
                    - Rank (r): {MODEL_CONFIG['training']['lora_config']['r']}
                    - Alpha: {MODEL_CONFIG['training']['lora_config']['lora_alpha']}
                    - Dropout: {MODEL_CONFIG['training']['lora_config']['lora_dropout']}
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

# Добавим функцию для обновления информации о модели
def update_model_info(model_key):
    """Update model information display"""
    model = MODELS[model_key]
    return f"""
    **Current Model:** {model['name']}
    
    **Model ID:** `{model['id']}`
    
    **Description:** {model['description']}
    
    **Type:** {model['type']}
    """

# Добавим функцию для смены модели
def change_model(model_key):
    """Change active model"""
    global client, ACTIVE_MODEL
    
    try:
        # Update active model
        ACTIVE_MODEL = MODELS[model_key]
        
        # Reinitialize client with new model
        client = InferenceClient(
            ACTIVE_MODEL["id"],
            token=HF_TOKEN
        )
        
        return update_model_info(model_key)
    except Exception as e:
        return f"Error changing model: {str(e)}"

# Добавим обработчик события изменения модели
model_selector.change(
    fn=change_model,
    inputs=[model_selector],
    outputs=[model_info]
)

# При инициализации установим информацию о текущей модели
model_info.value = update_model_info(DEFAULT_MODEL)

# Launch application
if __name__ == "__main__":
    # Check knowledge base availability in dataset
    if not load_vector_store():
        print("Knowledge base not found. Please create it through the interface.")
    
    demo.launch()

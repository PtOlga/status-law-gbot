import gradio as gr
import os
from huggingface_hub import InferenceClient
from config.constants import DEFAULT_SYSTEM_MESSAGE
from config.settings import HF_TOKEN, MODEL_CONFIG, EMBEDDING_MODEL
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
    MODEL_CONFIG["id"],
    token=HF_TOKEN
)

# State for storing context
context_store = {}

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
            if result is None:
                print("Debug - Vector store is None despite success=True")
                return None
                
            if isinstance(result, str):
                print(f"Debug - Vector store is a string: {result}")
                return None
                
            # Check if the result has a similarity_search method
            if hasattr(result, 'similarity_search'):
                print("Debug - Vector store loaded successfully with similarity_search method")
                return result
            else:
                print(f"Debug - Vector store object does not have similarity_search method: {type(result)}")
                return None
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
                if isinstance(item, list) and len(item) == 2:
                    user_msg, assistant_msg = item
                    
                    # Handle different formats of user_msg and assistant_msg
                    if isinstance(user_msg, dict) and "content" in user_msg:
                        messages.append({"role": "user", "content": user_msg["content"]})
                    elif isinstance(user_msg, str):
                        messages.append({"role": "user", "content": user_msg})
                    else:
                        messages.append({"role": "user", "content": str(user_msg)})
                    
                    if isinstance(assistant_msg, dict) and "content" in assistant_msg:
                        messages.append({"role": "assistant", "content": assistant_msg["content"]})
                    elif isinstance(assistant_msg, str):
                        messages.append({"role": "assistant", "content": assistant_msg})
                    else:
                        messages.append({"role": "assistant", "content": str(assistant_msg)})
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
        for chunk in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        ):
            if hasattr(chunk.choices[0], 'finish_reason') and chunk.choices[0].finish_reason is not None:
                is_complete = True
                break
                
            token = chunk.choices[0].delta.content
            if token:
                response += token
                
                # Create a properly formatted history for Gradio
                formatted_history = []
                
                # Include existing history (ensure proper format)
                if history:
                    for item in history:
                        if isinstance(item, list) and len(item) == 2:
                            user_item, assistant_item = item
                            
                            # Ensure user message has correct format
                            if not isinstance(user_item, dict) or "role" not in user_item or "content" not in user_item:
                                user_item = {"role": "user", "content": str(user_item) if not isinstance(user_item, dict) else user_item.get("content", "")}
                            
                            # Ensure assistant message has correct format
                            if not isinstance(assistant_item, dict) or "role" not in assistant_item or "content" not in assistant_item:
                                assistant_item = {"role": "assistant", "content": str(assistant_item) if not isinstance(assistant_item, dict) else assistant_item.get("content", "")}
                            
                            formatted_history.append([user_item, assistant_item])
                
                # Add the new message pair
                formatted_history.append([
                    {"role": "user", "content": message}, 
                    {"role": "assistant", "content": response}
                ])
                
                yield formatted_history, conversation_id

        # Save history if response is complete
        if is_complete or response:  # add response check as fallback
            messages.append({"role": "assistant", "content": response})
            try:
                from src.knowledge_base.dataset import DatasetManager
                from config.settings import HF_TOKEN
                
                dataset = DatasetManager(token=HF_TOKEN)
                success, msg = dataset.save_chat_history(conversation_id, messages)
                print(f"Chat history save attempt: {success}, Message: {msg}")
                if not success:
                    print(f"Failed to save chat history: {msg}")
            except Exception as e:
                import traceback
                print(f"Exception while saving chat history: {str(e)}")
                print(traceback.format_exc())
            
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        # Format error response in the way Gradio chatbot expects
        formatted_history = []
        
        # Copy existing history (ensure proper format)
        if history:
            for item in history:
                if isinstance(item, list) and len(item) == 2:
                    user_item, assistant_item = item
                    
                    # Ensure user message has correct format
                    if not isinstance(user_item, dict) or "role" not in user_item or "content" not in user_item:
                        user_item = {"role": "user", "content": str(user_item) if not isinstance(user_item, dict) else user_item.get("content", "")}
                    
                    # Ensure assistant message has correct format
                    if not isinstance(assistant_item, dict) or "role" not in assistant_item or "content" not in assistant_item:
                        assistant_item = {"role": "assistant", "content": str(assistant_item) if not isinstance(assistant_item, dict) else assistant_item.get("content", "")}
                    
                    formatted_history.append([user_item, assistant_item])
        
        # Add error message
        formatted_history.append([
            {"role": "user", "content": message}, 
            {"role": "assistant", "content": f"An error occurred while generating the response: {str(e)}"}
        ])
        
        yield formatted_history, conversation_id


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

def respond_and_clear(message, history, conversation_id):
    """Handle chat message and clear input"""
    # Get model parameters from config
    max_tokens = MODEL_CONFIG['parameters']['max_length']
    temperature = MODEL_CONFIG['parameters']['temperature']
    top_p = MODEL_CONFIG['parameters']['top_p']
    
    # Print debug information to help diagnose the issue
    print("Debug - Message type:", type(message), "Content:", message)
    print("Debug - History type:", type(history), "Content:", history)
    
    # Create user message in proper format
    user_message = {"role": "user", "content": message}
    
    # Use system message from constants
    response_generator = respond(
        message=message,
        history=history,
        conversation_id=conversation_id,
        system_message=DEFAULT_SYSTEM_MESSAGE,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )
    
    try:
        # Get first response from generator
        new_history, conv_id = next(response_generator)
        
        # Print debug information about the received history
        print("Debug - New history received:", new_history)
        
        # Ensure the final output is properly formatted for Gradio
        formatted_history = []
        for item in new_history:
            if isinstance(item, list) and len(item) == 2:
                user_msg, assistant_msg = item
                
                # Ensure user message is properly formatted
                if not isinstance(user_msg, dict) or "role" not in user_msg or "content" not in user_msg:
                    user_msg = {"role": "user", "content": str(user_msg) if not isinstance(user_msg, dict) else user_msg.get("content", "")}
                
                # Ensure assistant message is properly formatted
                if not isinstance(assistant_msg, dict) or "role" not in assistant_msg or "content" not in assistant_msg:
                    assistant_msg = {"role": "assistant", "content": str(assistant_msg) if not isinstance(assistant_msg, dict) else assistant_msg.get("content", "")}
                
                formatted_history.append([user_msg, assistant_msg])
        
        # Print the final formatted history
        print("Debug - Formatted history:", formatted_history)
        
        return formatted_history, conv_id, ""  # Clear message input
    except Exception as e:
        print(f"Error in respond_and_clear: {str(e)}")
        # Create a properly formatted error message
        error_history = []
        if history:
            # Copy existing history (ensuring proper format)
            for item in history:
                if isinstance(item, list) and len(item) == 2:
                    user_msg, assistant_msg = item
                    
                    # Ensure user message is properly formatted
                    if not isinstance(user_msg, dict) or "role" not in user_msg or "content" not in user_msg:
                        user_msg = {"role": "user", "content": str(user_msg) if not isinstance(user_msg, dict) else user_msg.get("content", "")}
                    
                    # Ensure assistant message is properly formatted
                    if not isinstance(assistant_msg, dict) or "role" not in assistant_msg or "content" not in assistant_msg:
                        assistant_msg = {"role": "assistant", "content": str(assistant_msg) if not isinstance(assistant_msg, dict) else assistant_msg.get("content", "")}
                    
                    error_history.append([user_msg, assistant_msg])
        
        # Add the error message
        error_history.append([
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"An error occurred: {str(e)}"}
        ])
        
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
                        type="messages",  # Use new messages format
                        avatar_images=["user.png", "assistant.png"]
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
            clear_btn.click(lambda: ([], None), None, [chatbot, conversation_id])

        with gr.Tab("Model Settings"):
            gr.Markdown("### Model Configuration")
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Model Information
                    gr.Markdown(f"""
                    **Current Model:** {MODEL_CONFIG['name']}
                    
                    **Model ID:** `{MODEL_CONFIG['id']}`
                    
                    **Description:** {MODEL_CONFIG['description']}
                    
                    **Type:** {MODEL_CONFIG['type']}
                    
                    **Embeddings Model:** `{EMBEDDING_MODEL}`
                    *Used for vector store creation and similarity search*
                    """)
                    
                    gr.Markdown("### Model Parameters")
                    with gr.Row():
                        max_length = gr.Slider(
                            minimum=1,
                            maximum=4096,
                            value=MODEL_CONFIG['parameters']['max_length'],
                            step=1,
                            label="Maximum Length",
                            interactive=False
                        )
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=MODEL_CONFIG['parameters']['temperature'],
                            step=0.1,
                            label="Temperature",
                            interactive=False
                        )
                    with gr.Row():
                        top_p = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=MODEL_CONFIG['parameters']['top_p'],
                            step=0.1,
                            label="Top-p",
                            interactive=False
                        )
                        rep_penalty = gr.Slider(
                            minimum=1.0,
                            maximum=2.0,
                            value=MODEL_CONFIG['parameters']['repetition_penalty'],
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

# Launch application
if __name__ == "__main__":
    # Check knowledge base availability in dataset
    if not load_vector_store():
        print("Knowledge base not found. Please create it through the interface.")
    
    demo.launch()

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
        dataset = DatasetManager()
        success, result = dataset.download_vector_store()
        
        if success and result is not None and not isinstance(result, str):
            return result
        
        # If result is a string, it's an error message
        if isinstance(result, str):
            print(f"Error loading knowledge base: {result}")
        else:
            print("Knowledge base is None or invalid format")
        
        return None
    except Exception as e:
        print(f"Exception loading knowledge base: {str(e)}")
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
    
    # Convert history to OpenAI format for API call
    # Handle different possible history formats more carefully
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
                    
                    if isinstance(assistant_msg, dict) and "content" in assistant_msg:
                        messages.append({"role": "assistant", "content": assistant_msg["content"]})
                    elif isinstance(assistant_msg, str):
                        messages.append({"role": "assistant", "content": assistant_msg})
        except Exception as e:
            print(f"Error processing history: {str(e)}")
            # Continue with empty history if there was an error
    
    # Add current user message
    messages.append({"role": "user", "content": message})
    
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
                # Format response in the way Gradio chatbot expects for type="messages"
                new_history = []
                if history:
                    new_history.extend(history)
                
                # Add the new message pair
                new_history.append([
                    {"role": "user", "content": message}, 
                    {"role": "assistant", "content": response}
                ])
                
                yield new_history, conversation_id

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
        new_history = []
        
        # Copy existing history
        if history:
            for item in history:
                if isinstance(item, list) and len(item) == 2:
                    new_history.append(item)
        
        # Add error message
        new_history.append([
            {"role": "user", "content": message}, 
            {"role": "assistant", "content": f"An error occurred while generating the response: {str(e)}"}
        ])
        
        yield new_history, conversation_id

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
        new_history, conv_id = next(response_generator)
        # Ensure each message in history is in correct format for Gradio messages type
        formatted_history = []
        for msg_pair in new_history:
            if isinstance(msg_pair, list) and len(msg_pair) == 2:
                # Format user message
                if isinstance(msg_pair[0], str):
                    user_msg = {"content": msg_pair[0], "role": "user"}
                elif isinstance(msg_pair[0], dict) and "content" in msg_pair[0]:
                    user_msg = msg_pair[0]
                else:
                    user_msg = {"content": str(msg_pair[0]), "role": "user"}

                # Format assistant message
                if isinstance(msg_pair[1], str):
                    assistant_msg = {"content": msg_pair[1], "role": "assistant"}
                elif isinstance(msg_pair[1], dict) and "content" in msg_pair[1]:
                    assistant_msg = msg_pair[1]
                else:
                    assistant_msg = {"content": str(msg_pair[1]), "role": "assistant"}

                formatted_history.append([user_msg, assistant_msg])
        
        return formatted_history, conv_id, ""  # Clear message input
    except Exception as e:
        print(f"Error in respond_and_clear: {str(e)}")
        error_msg = {"role": "assistant", "content": f"An error occurred: {str(e)}"}
        if history:
            return history + [[{"role": "user", "content": message}, error_msg]], conversation_id, ""
        else:
            return [[{"role": "user", "content": message}, error_msg]], conversation_id, ""

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
                    update_kb_btn = gr.Button("Update Knowledge Base", variant="secondary")
                    rebuild_kb_btn = gr.Button("Rebuild Knowledge Base", variant="primary")
                    kb_status = gr.Textbox(label="Knowledge Base Status", interactive=False)

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

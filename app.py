import gradio as gr
import os
from huggingface_hub import InferenceClient
from config.constants import DEFAULT_SYSTEM_MESSAGE
from config.settings import DEFAULT_MODEL, HF_TOKEN
from src.knowledge_base.vector_store import create_vector_store, load_vector_store

if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")

# Initialize HF client with token
client = InferenceClient(
    DEFAULT_MODEL,
    token=HF_TOKEN
)

# State for storing context
context_store = {}

def get_context(message, conversation_id):
    """Get context from knowledge base"""
    vector_store = load_vector_store()
    if vector_store is None:
        return "Knowledge base not found. Please create it first."
    
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
    
    # Convert history to OpenAI format
    for user_msg, assistant_msg in history:
        messages.extend([
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        ])
    
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
            # Check for finish_reason in chunk
            if hasattr(chunk.choices[0], 'finish_reason') and chunk.choices[0].finish_reason is not None:
                is_complete = True
                break
                
            token = chunk.choices[0].delta.content
            if token:
                response += token
                yield [(message, response)], conversation_id

        # Save history if response is complete
        if is_complete or response:  # add response check as fallback
            messages.append({"role": "assistant", "content": response})
            try:
                from src.knowledge_base.dataset import DatasetManager
                from config.settings import HF_TOKEN
                
                dataset = DatasetManager(token=HF_TOKEN)  # Explicitly pass the token
                success, msg = dataset.save_chat_history(conversation_id, messages)
                print(f"Chat history save attempt: {success}, Message: {msg}")  # Add debug log
                if not success:
                    print(f"Failed to save chat history: {msg}")
            except Exception as e:
                import traceback
                print(f"Exception while saving chat history: {str(e)}")
                print(traceback.format_exc())  # Print full traceback for debugging
            
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        yield [(message, "An error occurred while generating the response.")], conversation_id

def build_kb():
    """Function to create knowledge base"""
    try:
        success, message = create_vector_store()
        return message
    except Exception as e:
        return f"Error creating knowledge base: {str(e)}"

def load_vector_store():
    """Load knowledge base from dataset"""
    try:
        from src.knowledge_base.dataset import DatasetManager
        dataset = DatasetManager()
        success, store = dataset.download_vector_store()
        if success:
            return store
        print(f"Error loading knowledge base: {store}")
        return None
    except Exception as e:
        print(f"Error loading knowledge base: {str(e)}")
        return None

# Create interface
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Status Law Assistant")
    
    conversation_id = gr.State(None)
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Chat",
                bubble_full_width=False,
                avatar_images=["user.png", "assistant.png"]  # optional
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Your question",
                    placeholder="Enter your question...",
                    scale=4
                )
                submit_btn = gr.Button("Send", variant="primary")
        
        with gr.Column(scale=1):
            gr.Markdown("### Knowledge Base Management")
            build_kb_btn = gr.Button("Create/Update Knowledge Base", variant="primary")
            kb_status = gr.Textbox(label="Knowledge Base Status", interactive=False)
            
            gr.Markdown("### Generation Settings")
            max_tokens = gr.Slider(
                minimum=1, 
                maximum=2048, 
                value=512, 
                step=1, 
                label="Maximum Response Length",
                info="Limits the number of tokens in response. More tokens = longer response"
            )
            temperature = gr.Slider(
                minimum=0.1, 
                maximum=2.0, 
                value=0.7, 
                step=0.1, 
                label="Temperature",
                info="Controls creativity. Lower value = more predictable responses"
            )
            top_p = gr.Slider(
                minimum=0.1, 
                maximum=1.0, 
                value=0.95, 
                step=0.05, 
                label="Top-p",
                info="Controls diversity. Lower value = more focused responses"
            )
            
            clear_btn = gr.Button("Clear Chat History")

    def respond_and_clear(
        message,
        history,
        conversation_id,
        max_tokens,
        temperature,
        top_p,
    ):
        # Use existing respond function
        response_generator = respond(
            message,
            history,
            conversation_id,
            DEFAULT_SYSTEM_MESSAGE,
            max_tokens,
            temperature,
            top_p,
        )
        
        # Return result and empty string to clear input field
        for response in response_generator:
            yield response[0], response[1], ""  # chatbot, conversation_id, empty string for msg

    # Event handlers
    msg.submit(
        respond_and_clear,
        [msg, chatbot, conversation_id, max_tokens, temperature, top_p],
        [chatbot, conversation_id, msg]  # Add msg to output parameters
    )
    submit_btn.click(
        respond_and_clear,
        [msg, chatbot, conversation_id, max_tokens, temperature, top_p],
        [chatbot, conversation_id, msg]  # Add msg to output parameters
    )
    build_kb_btn.click(build_kb, None, kb_status)
    clear_btn.click(lambda: ([], None), None, [chatbot, conversation_id])

# Launch application
if __name__ == "__main__":
    # Check knowledge base availability in dataset
    if not load_vector_store():
        print("Knowledge base not found. Please create it through the interface.")
    
    demo.launch()

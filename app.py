import gradio as gr
import os
from huggingface_hub import InferenceClient
from config.constants import DEFAULT_SYSTEM_MESSAGE
from config.settings import DEFAULT_MODEL, HF_TOKEN
from src.knowledge_base.vector_store import create_vector_store, load_vector_store

# Создаем клиент для инференса с токеном
client = InferenceClient(
    DEFAULT_MODEL,
    token=HF_TOKEN
)

# Состояние для хранения контекста
context_store = {}

def get_context(message, conversation_id):
    """Получение контекста из базы знаний"""
    vector_store = load_vector_store()
    if vector_store is None:
        return "База знаний не найдена. Пожалуйста, создайте её сначала."
    
    try:
        # Извлечение контекста
        context_docs = vector_store.similarity_search(message, k=3)
        context_text = "\n\n".join([f"Из {doc.metadata.get('source', 'неизвестно')}: {doc.page_content}" for doc in context_docs])
        
        # Сохраняем контекст для этого разговора
        context_store[conversation_id] = context_text
        
        return context_text
    except Exception as e:
        print(f"Ошибка при получении контекста: {str(e)}")
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
    # Если это новый разговор, создаем ID
    if not conversation_id:
        import uuid
        conversation_id = str(uuid.uuid4())
    
    # Получаем контекст из базы знаний
    context = get_context(message, conversation_id)
    
    # Формируем полную системную инструкцию с контекстом
    full_system_message = system_message
    if context:
        full_system_message += f"\n\nКонтекст для ответа:\n{context}"
    
    # Формируем сообщения для LLM
    messages = [{"role": "system", "content": full_system_message}]
    
    # Преобразуем историю в формат для API
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    
    # Добавляем текущее сообщение пользователя
    messages.append({"role": "user", "content": message})
    
    # Отправляем запрос к API и стримим ответ
    response = ""
    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content
        if token:
            response += token
            yield response, conversation_id

def build_kb():
    """Функция для создания базы знаний"""
    try:
        success, message = create_vector_store()
        return message
    except Exception as e:
        return f"Ошибка при создании базы знаний: {str(e)}"

# Создаем интерфейс
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Status Law Assistant")
    
    conversation_id = gr.State(None)
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Чат")
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Ваш вопрос",
                    placeholder="Введите ваш вопрос...",
                    scale=4
                )
                submit_btn = gr.Button("Отправить", variant="primary")
        
        with gr.Column(scale=1):
            gr.Markdown("### Управление базой знаний")
            build_kb_btn = gr.Button("Создать/обновить базу знаний", variant="primary")
            kb_status = gr.Textbox(label="Статус базы знаний", interactive=False)
            
            gr.Markdown("### Настройки чата")
            system_message = gr.Textbox(
                label="Системное сообщение", 
                value=DEFAULT_SYSTEM_MESSAGE,
                lines=5
            )
            max_tokens = gr.Slider(
                minimum=1, 
                maximum=2048, 
                value=512, 
                step=1, 
                label="Максимальное количество токенов"
            )
            temperature = gr.Slider(
                minimum=0.1, 
                maximum=2.0, 
                value=0.7, 
                step=0.1, 
                label="Температура"
            )
            top_p = gr.Slider(
                minimum=0.1, 
                maximum=1.0, 
                value=0.95, 
                step=0.05, 
                label="Top-p (nucleus sampling)"
            )
            
            clear_btn = gr.Button("Очистить историю чата")
    
    # Обработчики событий
    msg.submit(
        respond, 
        [msg, chatbot, conversation_id, system_message, max_tokens, temperature, top_p], 
        [chatbot, conversation_id]
    )
    submit_btn.click(
        respond, 
        [msg, chatbot, conversation_id, system_message, max_tokens, temperature, top_p], 
        [chatbot, conversation_id]
    )
    build_kb_btn.click(build_kb, None, kb_status)
    clear_btn.click(lambda: ([], None), None, [chatbot, conversation_id])

# Запускаем приложение
if __name__ == "__main__":
    # Проверяем наличие базы знаний
    if not os.path.exists(os.path.join("data", "vector_store", "index.faiss")):
        print("База знаний не найдена. Создайте её через интерфейс.")
    
    demo.launch()

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
    
    # Преобразуем историю из формата Gradio в формат OpenAI
    messages = [{"role": "system", "content": system_message}]
    if context:
        messages[0]["content"] += f"\n\nКонтекст для ответа:\n{context}"
    
    # Конвертируем историю в формат OpenAI
    for user_msg, assistant_msg in history:
        messages.extend([
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        ])
    
    # Добавляем текущее сообщение пользователя
    messages.append({"role": "user", "content": message})
    
    # Отправляем запрос к API и стримим ответ
    response = ""
    last_token = ""
    sentence_end_chars = {'.', '!', '?', '\n'}
    
    try:
        for chunk in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        ):
            token = chunk.choices[0].delta.content
            if token:
                response += token
                last_token = token
                yield [(message, response)], conversation_id

        # Проверяем, завершено ли последнее предложение
        if last_token and last_token[-1] not in sentence_end_chars:
            # Добавляем точку, если предложение не завершено
            response += "."
            yield [(message, response)], conversation_id

        # Сохраняем историю после полного ответа
        messages.append({"role": "assistant", "content": response})
        try:
            from src.knowledge_base.dataset import DatasetManager
            dataset = DatasetManager()
            success, msg = dataset.save_chat_history(conversation_id, messages)
            if not success:
                print(f"Ошибка при сохранении истории чата: {msg}")
        except Exception as e:
            print(f"Ошибка при сохранении истории чата: {str(e)}")
            
    except Exception as e:
        print(f"Ошибка при генерации ответа: {str(e)}")
        yield [(message, "Произошла ошибка при генерации ответа.")], conversation_id

def build_kb():
    """Функция для создания базы знаний"""
    try:
        success, message = create_vector_store()
        return message
    except Exception as e:
        return f"Ошибка при создании базы знаний: {str(e)}"

def load_vector_store():
    """Загрузка базы знаний из датасета"""
    try:
        from src.knowledge_base.dataset import DatasetManager
        dataset = DatasetManager()
        success, store = dataset.download_vector_store()
        if success:
            return store
        print(f"Ошибка загрузки базы знаний: {store}")
        return None
    except Exception as e:
        print(f"Ошибка при загрузке базы знаний: {str(e)}")
        return None

# Создаем интерфейс
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Status Law Assistant")
    
    conversation_id = gr.State(None)
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Чат",
                bubble_full_width=False,
                avatar_images=["user.png", "assistant.png"]  # опционально
            )
            
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
            
            gr.Markdown("### Настройки генерации")
            max_tokens = gr.Slider(
                minimum=1, 
                maximum=2048, 
                value=512, 
                step=1, 
                label="Максимальная длина ответа",
                info="Ограничивает количество токенов в ответе. Больше токенов = длиннее ответ"
            )
            temperature = gr.Slider(
                minimum=0.1, 
                maximum=2.0, 
                value=0.7, 
                step=0.1, 
                label="Температура",
                info="Контролирует креативность. Ниже значение = более предсказуемые ответы"
            )
            top_p = gr.Slider(
                minimum=0.1, 
                maximum=1.0, 
                value=0.95, 
                step=0.05, 
                label="Top-p",
                info="Контролирует разнообразие. Ниже значение = более сфокусированные ответы"
            )
            
            clear_btn = gr.Button("Очистить историю чата")

    def respond_and_clear(
        message,
        history,
        conversation_id,
        max_tokens,
        temperature,
        top_p,
    ):
        # Используем существующую функцию respond
        response_generator = respond(
            message,
            history,
            conversation_id,
            DEFAULT_SYSTEM_MESSAGE,
            max_tokens,
            temperature,
            top_p,
        )
        
        # Возвращаем результат и пустую строку для очистки поля ввода
        for response in response_generator:
            yield response[0], response[1], ""  # chatbot, conversation_id, пустая строка для msg

    # Обработчики событий
    msg.submit(
        respond_and_clear,
        [msg, chatbot, conversation_id, max_tokens, temperature, top_p],
        [chatbot, conversation_id, msg]  # Добавляем msg в выходные параметры
    )
    submit_btn.click(
        respond_and_clear,
        [msg, chatbot, conversation_id, max_tokens, temperature, top_p],
        [chatbot, conversation_id, msg]  # Добавляем msg в выходные параметры
    )
    build_kb_btn.click(build_kb, None, kb_status)
    clear_btn.click(lambda: ([], None), None, [chatbot, conversation_id])

# Запускаем приложение
if __name__ == "__main__":
    # Проверяем доступность базы знаний в датасете
    if not load_vector_store():
        print("База знаний не найдена. Создайте её через интерфейс.")
    
    demo.launch()

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from src.knowledge_base.loader import load_documents
from config.settings import VECTOR_STORE_PATH, EMBEDDING_MODEL
from config.constants import CHUNK_SIZE, CHUNK_OVERLAP

def get_embeddings():
    """Получение модели эмбеддингов"""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )

def create_vector_store():
    """Создание или обновление векторного хранилища"""
    # Загрузка документов
    documents = load_documents()
    
    if not documents:
        return False, "Ошибка: документы не загружены"
    
    # Разделение на чанки
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    
    # Инициализация эмбеддингов
    embeddings = get_embeddings()
    
    # Создание векторного хранилища
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Сохранение
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    vector_store.save_local(folder_path=VECTOR_STORE_PATH)
    
    return True, f"База знаний создана успешно! Загружено {len(documents)} документов, создано {len(chunks)} чанков."

def load_vector_store():
    """Загрузка векторного хранилища"""
    embeddings = get_embeddings()
    
    if not os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
        return None
    
    try:
        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store
    except Exception as e:
        print(f"Ошибка загрузки векторного хранилища: {str(e)}")
        return None
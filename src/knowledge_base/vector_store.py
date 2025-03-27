import os
import tempfile
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from src.knowledge_base.loader import load_documents
from config.settings import VECTOR_STORE_PATH, EMBEDDING_MODEL, HF_TOKEN
from config.constants import CHUNK_SIZE, CHUNK_OVERLAP

def get_embeddings():
    """Get embeddings model"""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )

def create_vector_store(mode: str = "rebuild"):
    """Create or update vector store and upload to dataset"""
    # Load documents
    documents = load_documents()
    
    if not documents:
        return False, "Error: documents not loaded"
    
    print(f"Loaded {len(documents)} documents")
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    # Initialize embeddings
    embeddings = get_embeddings()
    
    try:
        # Always create new vector store in rebuild mode
        if mode == "rebuild":
            print("Creating new vector store...")
            vector_store = FAISS.from_documents(chunks, embeddings)
        else:
            # Try to load and update existing store
            from src.knowledge_base.dataset import DatasetManager
            dataset = DatasetManager(token=HF_TOKEN)
            success, result = dataset.download_vector_store()
            
            if success:
                print("Updating existing vector store...")
                vector_store = result
                vector_store.add_documents(chunks)
            else:
                return False, "Failed to load existing vector store for update"
        
        # Upload to dataset with force flag
        from src.knowledge_base.dataset import DatasetManager
        dataset = DatasetManager(token=HF_TOKEN)
        success, message = dataset.upload_vector_store(vector_store, force_update=True)
        
        if not success:
            return False, f"Error uploading to dataset: {message}"
        
        action = "updated" if mode == "update" else "created"
        return True, f"Knowledge base {action} successfully! Processed {len(documents)} documents, {len(chunks)} chunks."
        
    except Exception as e:
        return False, f"Error {mode}ing knowledge base: {str(e)}"

def load_vector_store():
    """Load vector store"""
    try:
        from src.knowledge_base.dataset import DatasetManager
        dataset = DatasetManager(token=HF_TOKEN)
        success, result = dataset.download_vector_store()
        
        if not success:
            print(f"Failed to download vector store: {result}")
            return None
            
        return result
        
    except Exception as e:
        print(f"Error loading vector store: {str(e)}")
        return None

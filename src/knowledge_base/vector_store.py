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

def create_vector_store():
    """Create vector store and upload to dataset"""
    # Load documents
    documents = load_documents()
    
    if not documents:
        return False, "Error: documents not loaded"
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    
    # Initialize embeddings
    embeddings = get_embeddings()
    
    # Create vector store in temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        vector_store = FAISS.from_documents(chunks, embeddings)
        # Save to temporary directory
        vector_store.save_local(folder_path=temp_dir)
        
        # Copy files to VECTOR_STORE_PATH for subsequent loading
        os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
        for file in ["index.faiss", "index.pkl"]:
            shutil.copy2(
                os.path.join(temp_dir, file),
                os.path.join(VECTOR_STORE_PATH, file)
            )
        
        # Upload to dataset with explicit token passing
        from src.knowledge_base.dataset import DatasetManager
        dataset = DatasetManager(token=HF_TOKEN)
        success, message = dataset.upload_vector_store()
        
        # Clean up local files after upload
        shutil.rmtree(VECTOR_STORE_PATH)
        
        if not success:
            return False, f"Error uploading to dataset: {message}"
    
    return True, f"Knowledge base created successfully! Loaded {len(documents)} documents, created {len(chunks)} chunks."

def load_vector_store():
    """Load vector store"""
    try:
        # First check if we need to download from dataset
        from src.knowledge_base.dataset import DatasetManager
        dataset = DatasetManager(token=HF_TOKEN)
        success, result = dataset.download_vector_store()
        
        if not success:
            print(f"Failed to download vector store: {result}")
            return None
            
        # Now try to load the local vector store
        embeddings = get_embeddings()
        
        if not os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
            print("Vector store files not found locally")
            return None
        
        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store
        
    except Exception as e:
        print(f"Error loading vector store: {str(e)}")
        return None

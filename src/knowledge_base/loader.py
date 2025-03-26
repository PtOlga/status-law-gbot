import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from config.constants import URLS

def load_documents():
    """Load documents from website"""
    documents = []
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }
    
    for url in URLS:
        try:
            loader = WebBaseLoader(
                web_paths=[url],
                header_template=headers
            )
            docs = loader.load()
            if docs:
                documents.extend(docs)
                print(f"Loaded {url}: {len(docs)} documents")
        except Exception as e:
            print(f"Error loading {url}: {str(e)}")
    
    return documents

---
language:
- en
pretty_name: status-law-knowledge-base-gbot
---

# Status Law Knowledge Base Dataset

This dataset serves as a storage for the Status Law Assistant chatbot, containing vector embeddings and chat history.

## 📁 Structure

```
status-law-knowledge-base/
├── vector_store/
│   ├── index.faiss     # FAISS vector store for document embeddings
│   └── index.pkl       # Metadata and configuration for the vector store
│
└── chat_history/
    └── logs.json       # Chat history logs
```

## 🔍 Description

- `vector_store/`: Contains FAISS embeddings of legal documents from status.law website
  - `index.faiss`: Vector embeddings for semantic search
  - `index.pkl`: Metadata and configuration information

- `chat_history/`: Stores conversation logs
  - `logs.json`: JSON file containing chat history and metadata

## 🚀 Usage

This dataset is used by the Status Law Assistant chatbot to:
1. Store and retrieve document embeddings for context-aware responses
2. Maintain chat history for conversation continuity
3. Track user interactions and improve response quality

## 🔗 Related Links

- [Status Law Website](https://status.law)
- [Status Law Assistant Repository](https://huggingface.co/spaces/Rulga/status-law-assistant)

## 📝 License

Private dataset for Status Law Assistant usage only.
---
title: Status Law Gbot
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.23.0
app_file: app.py
pinned: false
---

# Status Law Assistant

An intelligent chatbot based on Hugging Face and LangChain for legal consultations using information from the Status Law company website.

## 📝 Description

Status Law Assistant is a smart chatbot that answers user questions about Status Law company's legal services. The bot uses RAG (Retrieval-Augmented Generation) technology to find relevant information in a knowledge base created from the official website content and generates responses using a language model.

## ✨ Features

- Automatic creation and updating of knowledge base from status.law website content
- Relevant information search for user queries
- Context-aware response generation
- Multi-language query support (responds in the language of the question)
- Customizable text generation parameters (temperature, token count, etc.)

## 🚀 Technologies

- **LangChain**: For query processing chains and knowledge base management
- **Hugging Face**: For language model access and application hosting
- **FAISS**: For efficient vector search
- **Gradio**: For user interface creation
- **BeautifulSoup**: For web page information extraction

## 🏗️ Project Structure

```
status-law-gbot/
├── app.py                 # Main application file with interface and request handling logic
├── requirements.txt       # Project dependencies
├── config/               # Configuration files
│   ├── settings.py       # Application settings
│   └── constants.py      # Constants and default values
├── src/                  # Source code
│   ├── analytics/        # Analytics module
│   │   └── chat_analyzer.py
│   ├── knowledge_base/   # Knowledge base management
│   │   ├── loader.py
│   │   └── vector_store.py
│   ├── training/         # Model training module
│   │   ├── fine_tuner.py
│   │   └── model_manager.py
│   └── models/          # Model-related code
├── web/                 # Web interface components
│   └── training_interface.py
└── data/               # Data storage
    ├── vector_store/   # FAISS vector storage
    │   ├── index.faiss
    │   └── index.pkl
    └── chat_history/   # Conversation logs
        └── logs.json
```

## 💾 Data Storage

### Vector Store
- `data/vector_store/index.faiss`: FAISS vector store for document embeddings
- `data/vector_store/index.pkl`: Metadata and configuration for the vector store

### Chat History
- `data/chat_history/logs.json`: JSON file containing chat history and metadata

## 🚀 Usage

The Status Law Assistant chatbot uses this structure to:
1. Store and retrieve document embeddings for context-aware responses
2. Maintain chat history for conversation continuity
3. Track user interactions and improve response quality
4. Fine-tune models based on conversation history

## 🛠️ Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/status-law-gbot.git
cd status-law-gbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run the application:
```bash
python app.py
```

## 🔗 Related Links

- [Status Law Website](https://status.law)
- [Status Law Assistant on Hugging Face](https://huggingface.co/spaces/Rulga/status-law-assistant)

## 📝 License

Private repository for Status Law Assistant usage only.

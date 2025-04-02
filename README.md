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
- Model switching with fallback mechanism
- Fine-tuning capabilities based on chat history
- Multiple model support:
  - Llama 2 7B Chat: Meta's general-purpose model optimized for chat
  - Zephyr 7B: State-of-the-art model with strong performance
  - Mistral 7B Instruct v0.2: Advanced model with superior multilingual capabilities
  - XGLM 7.5B: Specialized model for cross-lingual generation with 30+ language support

## 🚀 Technologies

- **LangChain**: For query processing chains and knowledge base management
- **Hugging Face**: For language model access and application hosting
- **FAISS**: For efficient vector search
- **Gradio**: For user interface creation
- **BeautifulSoup**: For web page information extraction
- **PEFT**: For efficient fine-tuning using LoRA
- **SentencePiece**: For tokenization

## 🏗️ Project Structure

```
status-law-gbot/
├── app.py                 # Main application file with interface and request handling logic
├── requirements.txt       # Project dependencies
├── config/               # Configuration files
│   ├── settings.py       # Application settings and model configurations
│   └── constants.py      # Constants and default values
├── src/                  # Source code
│   ├── analytics/        # Analytics module
│   │   └── chat_analyzer.py
│   ├── knowledge_base/   # Knowledge base management
│   │   ├── loader.py
│   │   └── vector_store.py
│   ├── training/         # Model training module
│   │   ├── fine_tuner.py  # LoRA fine-tuning implementation
│   │   └── model_manager.py  # Model switching and management
│   └── models/          # Model storage
│       └── fine_tuned/  # Fine-tuned model storage
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

### Models
- `src/models/fine_tuned/`: Directory for storing fine-tuned models
- `src/models/registry.json`: Model registry and configuration

## 🚀 Usage

The Status Law Assistant chatbot uses this structure to:
1. Store and retrieve document embeddings for context-aware responses
2. Maintain chat history for conversation continuity
3. Track user interactions and improve response quality
4. Fine-tune models based on conversation history
5. Provide automatic model fallback in case of API errors
6. Support multiple language models with easy switching

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
# Edit .env with your configuration, including HUGGINGFACE_TOKEN
```

4. Run the application:
```bash
python app.py
```

## 🔧 Model Fine-tuning

To fine-tune the model on your chat history:

```python
from src.training.fine_tuner import finetune_from_chat_history

success, message = finetune_from_chat_history(epochs=3)
print(message)
```

The fine-tuning process uses LoRA (Low-Rank Adaptation) for efficient training with minimal resource requirements.

## 🔄 Model Switching

The application supports multiple models with automatic fallback:

- Llama 2 7B Chat (default)
- Zephyr 7B
- Custom fine-tuned versions

Models can be switched dynamically through the interface or programmatically:

```python
from src.training.model_manager import switch_to_model

switch_to_model("llama-7b")  # or "zephyr-7b"
```

## 🔗 Related Links

- [Status Law Website](https://status.law)
- [Status Law Assistant on Hugging Face](https://huggingface.co/spaces/Rulga/status-law-assistant)

## 📝 License

Private repository for Status Law Assistant usage only.

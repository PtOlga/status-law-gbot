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

Status Law Assistant is a smart chatbot that answers user questions about Status Law company's legal services. The bot uses RAG (Retrieval-Augmented Generation) technology to find relevant information in a knowledge base created from the official website content.

## ✨ Key Features

- Knowledge Base Management:
  - Dynamic URL management for knowledge base sources
  - Ability to add custom URLs for information extraction
  - Selective source inclusion/exclusion
  - Two modes of knowledge base updates:
    - Update Mode: Adds new information while preserving existing knowledge
    - Rebuild Mode: Complete recreation of knowledge base from selected sources
  - Real-time status tracking for knowledge base operations
  - Automatic metadata management and versioning
- Automatic creation and updating of knowledge base from status.law website content
- Intelligent information retrieval for query responses
- Context-aware response generation
- Advanced multilingual support:
  - Automatic language detection
  - Native language response generation
  - Built-in translation system with fallback mechanism
  - Support for 30+ languages
- Customizable text generation parameters
- Model switching system with automatic fallback
- Fine-tuning capabilities based on chat history
- Multiple model support:
  - Llama 2 7B Chat (primary): Optimized for dialogues
  - Zephyr 7B: Enhanced performance and response quality
  - Mistral 7B Instruct v0.2: Superior multilingual capabilities
  - XGLM 7.5B: Specialized cross-lingual generation model (requires paid API access)

## 🚀 Technologies

- **LangChain**: Query processing chains and knowledge base management
- **Hugging Face**: Language model access and hosting
- **FAISS**: Efficient vector search
- **Gradio**: User interface creation
- **BeautifulSoup**: Web page information extraction
- **PEFT**: Efficient fine-tuning using LoRA
- **SentencePiece**: Tokenization

## 🏗️ Project Structure

```
status-law-gbot/
├── app.py                 # Main application file
├── requirements.txt       # Project dependencies
├── config/               # Configuration files
│   ├── settings.py       # Application and model settings
│   └── constants.py      # Constants and default values
├── src/                  # Source code
│   ├── analytics/        # Analytics module
│   │   └── chat_analyzer.py
│   ├── knowledge_base/   # Knowledge base management
│   │   ├── loader.py
│   │   └── vector_store.py
│   └── training/         # Training module
│       ├── fine_tuner.py
│       └── model_manager.py
└── dataset/             # HuggingFace dataset structure
    ├── annotations/     # Conversation annotations
    ├── chat_history/    # Chat logs and conversations
    ├── fine_tuned_models/ # Fine-tuned model storage
    ├── preferences/     # User preferences
    ├── training_data/   # Processed training data
    ├── training_logs/   # Training process logs
    └── vector_store/    # FAISS vector storage
```

## 💾 Data Storage

### Dataset Organization
- `annotations/`: Conversation quality metrics and annotations
- `chat_history/`: JSON files containing chat conversations
- `fine_tuned_models/`: Storage for LoRA adapters and model checkpoints
- `preferences/`: User preferences and settings
- `training_data/`: Processed data ready for model training
- `training_logs/`: Detailed training process logs
- `vector_store/`: FAISS indexes for semantic search

## 🛠️ Setup

1. Clone the repository:
```bash
git clone https://github.com/PtOlga/status-law-gbot.git
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

- Llama 2 7B Chat (default): Optimized for dialogues
- Zephyr 7B: Enhanced performance and response quality
- Mistral 7B Instruct v0.2: Superior multilingual capabilities
- XGLM 7.5B: Specialized cross-lingual generation model (requires paid API access)

Models can be switched dynamically through the interface or programmatically:

```python
from src.training.model_manager import switch_to_model

switch_to_model("llama-7b")  # or "zephyr-7b", "mistral-7b", "xglm-7b"
```

## 🔄 Knowledge Base Management

The application provides a flexible interface for managing knowledge sources:

1. **Source Management**:
   - View and edit the list of source URLs
   - Enable/disable specific sources
   - Add custom URLs for information extraction
   - Monitor source status and availability

2. **Update Operations**:
   - **Update Knowledge Base**: Incrementally add new information while preserving existing knowledge
   - **Rebuild Knowledge Base**: Completely recreate the knowledge base using only selected sources
   - Real-time operation status tracking
   - Automatic backup of previous versions

3. **Usage**:
   ```python
   # Add new URL to knowledge base
   sources_df.append({"URL": "https://example.com", "Include": True, "Status": "Ready"})
   
   # Update knowledge base with selected sources
   update_kb_with_selected(sources_df)
   
   # Rebuild knowledge base from scratch
   rebuild_kb_with_selected(sources_df)
   ```

All changes to the knowledge base are automatically synchronized with the Hugging Face dataset, ensuring data persistence and version control.

## 🔗 Related Links

- [Status Law Website](https://status.law)
- [Status Law Assistant on Hugging Face](https://huggingface.co/spaces/Rulga/status-law-gbot)

## 📝 License

Public repository for Status Law Assistant.

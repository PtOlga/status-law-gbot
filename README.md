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

An example chatbot using [Gradio](https://gradio.app), [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/v0.22.2/en/index), and the [Hugging Face Inference API](https://huggingface.co/docs/api-inference/index).

# Status Law Assistant

Чат-бот на базе Hugging Face и LangChain для юридической консультации на основе информации с сайта компании Status Law.

## 📝 Описание

Status Law Assistant — это интеллектуальный чат-бот, который отвечает на вопросы пользователей о юридических услугах компании Status Law. Бот использует технологию RAG (Retrieval-Augmented Generation), чтобы находить релевантную информацию в базе знаний, созданной на основе содержимого официального сайта компании, и генерировать на её основе ответы с помощью языковой модели.

## ✨ Возможности

- Автоматическое создание и обновление базы знаний на основе контента сайта status.law
- Поиск релевантной информации для ответа на вопросы пользователей
- Генерация ответов с использованием контекстно-ориентированного подхода
- Поддержка многоязычных запросов (отвечает на языке вопроса)
- Настраиваемые параметры генерации текста (температура, количество токенов и т.д.)

## 🚀 Технологии

- **LangChain**: для создания цепочек обработки запросов и управления базой знаний
- **Hugging Face**: для доступа к языковым моделям и хостинга приложения
- **FAISS**: для эффективного векторного поиска
- **Gradio**: для создания пользовательского интерфейса
- **BeautifulSoup**: для извлечения информации с веб-страниц

## 🏗️ Структура проекта

- `app.py`: основной файл приложения, в котором определен интерфейс и логика обработки запросов
- `config/`: директория с конфигурационными файлами
- `src/`: директория с исходным кодом
  - `knowledge_base/`: модуль для работы с базой знаний
  - `models/`: модуль для работы с моделями
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

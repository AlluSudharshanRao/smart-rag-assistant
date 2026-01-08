# 🚀 Smart RAG Document Assistant

A production-ready Retrieval-Augmented Generation (RAG) system that enables intelligent Q&A over documents using modern LLMs and vector databases.

## ✨ Features

- 📄 **Multi-Format Support**: PDF, DOCX, TXT, MD file uploads with batch processing
- 🧠 **Intelligent Chunking**: Smart document splitting with configurable overlap
- 🔍 **Vector Search**: Semantic similarity search using embeddings
- 💬 **Conversational Q&A**: Multi-turn conversations with context preservation
- 📚 **Source Citations**: Traceable answers with source documents and metadata
- 📁 **Multi-Document Collections**: Organize documents into separate collections
- 📊 **Analytics Dashboard**: Track query performance, relevance, and response times
- 📈 **Evaluation Metrics**: Monitor precision, recall, F1 score, and answer quality
- 💾 **Export/Import**: Export chat history, evaluation results, and collection data
- 🗑️ **Document Management**: View and delete documents by source file
- 🎨 **Modern UI**: Beautiful Streamlit interface with tabbed navigation
- 🐳 **Docker Ready**: Containerized for easy local deployment
- ☁️ **Cloud Deployed**: Live on Streamlit Cloud

## 🛠️ Tech Stack

- **Frontend**: Streamlit 1.28.1
- **LLM Framework**: LangChain 0.1.20
- **Vector Database**: ChromaDB 0.4.18
- **Embeddings**: Sentence Transformers (local, free) / OpenAI (optional)
- **LLM**: OpenAI GPT-3.5-turbo / GPT-4
- **Document Processing**: PyPDF, python-docx
- **Containerization**: Docker & Docker Compose
- **Deployment**: Streamlit Cloud

## 📋 Prerequisites

- Python 3.9+
- Docker & Docker Compose (optional, for containerized deployment)
- OpenAI API key (optional, for GPT models)

## 🚀 Quick Start

### Option 1: Local Development

```bash
# Clone or navigate to project
cd smart-rag-assistant

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="api-key"  # Optional
export EMBEDDINGS_MODEL="sentence-transformers"  # or "openai"

# Run the application
streamlit run app.py
```

### Option 2: Docker

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access application at http://localhost:8501
```

## 📖 Usage

1. **Upload Documents**: Use the sidebar to upload PDF/DOCX/TXT/MD files (single or batch)
2. **Process Documents**: Click "Process & Index Document" to chunk and embed documents
3. **Manage Collections**: Create separate collections to organize different document sets
4. **Ask Questions**: Type questions in the chat interface at the bottom of the page
5. **View Sources**: Expand source citations to see the retrieved document chunks
6. **Monitor Performance**: Check the Analytics tab for query metrics and evaluation results
7. **Export Data**: Download chat history, evaluation results, and collection info from the Export/Import tab

## 🏗️ Architecture

```
User Upload → Document Parser → Text Chunker → Embeddings → Vector DB
                                                              ↓
User Question → Embedding → Vector Search → Context Retrieval → LLM → Answer
```

## 📁 Project Structure

```
smart-rag-assistant/
├── app.py                      # Main Streamlit application
├── core/
│   ├── __init__.py
│   ├── document_processor.py   # Document parsing & chunking
│   ├── embeddings.py           # Embedding generation (OpenAI/Sentence Transformers)
│   └── rag_chain.py            # RAG pipeline implementation
├── features/
│   ├── __init__.py
│   ├── multi_document.py       # Multi-collection management
│   └── evaluation.py           # RAG evaluation metrics
├── utils/
│   ├── __init__.py
│   └── config.py               # Configuration management
├── requirements.txt            # Python dependencies
├── runtime.txt                 # Python version for Streamlit Cloud
├── Dockerfile                  # Docker container configuration
├── docker-compose.yml          # Docker Compose setup
├── env_template.txt            # Environment variables template
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```

## 🚀 Deployment

### Streamlit Cloud (Recommended)

The application is currently deployed on Streamlit Cloud:

1. Push the repository to GitHub
2. Connect your GitHub account to [Streamlit Cloud](https://streamlit.io/cloud)
3. Select the repository and set the main file to `app.py`
4. Configure secrets (if needed) in Streamlit Cloud dashboard:
   - `OPENAI_API_KEY`: Your OpenAI API key (optional)
5. Deploy!

### Local Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access application at http://localhost:8501
```

### Other Cloud Platforms

For AWS/GCP/Azure deployment, adapt the Docker configuration or deploy directly using their container services.

## 📊 Key Features Details

- **Document Processing**: Supports PDF, DOCX, TXT, and Markdown files with automatic chunking
- **Embedding Options**: 
  - Sentence Transformers (local, free) - default
  - OpenAI embeddings (requires API key)
- **RAG Pipeline**: Uses LangChain with ChromaDB for efficient vector storage and retrieval
- **Evaluation Metrics**: Automatic tracking of relevance scores, answer quality, precision, recall, and response times
- **Data Persistence**: Chat history and evaluation results are saved locally (or in-memory for cloud deployments)
- **Collection Management**: Create multiple collections to organize documents by topic or project

## 🔧 Environment Variables

Create a `.env` file based on `env_template.txt`:

```bash
# For local embeddings (FREE - no API key needed)
EMBEDDINGS_MODEL=sentence-transformers

# OR for OpenAI embeddings (requires API key)
# OPENAI_API_KEY=your-api-key-here
# EMBEDDINGS_MODEL=openai
# OPENAI_MODEL=gpt-3.5-turbo

# Vector Database
CHROMA_PERSIST_DIRECTORY=./chroma_db

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# RAG Settings
TOP_K_RETRIEVAL=3
TEMPERATURE=0.7
MAX_TOKENS=500
```

## 🔗 Live Demo

🌐 **Live Application**: [https://smart-rag-assistant.streamlit.app/](https://smart-rag-assistant.streamlit.app/)

Try the application online - upload documents, ask questions, and explore the analytics dashboard!


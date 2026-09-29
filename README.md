# District RAG Chatbot 🏛️

A production-style RAG (Retrieval-Augmented Generation) chatbot that answers questions using uploaded district documents.

## Tech Stack

- **Backend:** Python 3.11, FastAPI, LangChain, FAISS, SentenceTransformers
- **Frontend:** React, Vite, Tailwind CSS
- **LLM:** Mistral 7B Instruct via Ollama (FREE, local)
- **Embeddings:** all-MiniLM-L6-v2

## Quick Start

### 1. Ollama Setup
```bash
# Install from https://ollama.com
ollama pull mistral
ollama serve
```

### 2. Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### 3. Frontend
```bash
cd frontend
npm install
npm run dev
```

### 4. Open Browser
- Frontend: http://localhost:5173
- API Docs: http://localhost:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /upload | Upload PDF/DOCX/TXT documents |
| POST | /chat | Ask questions about uploaded documents |
| GET | /health | Health check with index stats |

## Features

- 📄 Document upload (PDF, DOCX, TXT)
- 🔍 Semantic search with FAISS
- 🤖 Mistral 7B answers via Ollama
- 📎 Source references in answers
- 🎨 Premium dark-themed UI
- 📱 Responsive design

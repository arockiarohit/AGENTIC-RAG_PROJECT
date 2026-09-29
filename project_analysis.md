# Comprehensive Project File Workflow

This document details the exact purpose and workflow of **every single file** in the District RAG Chatbot project across the backend, services, data loading, and frontend.

---

## 1. Root Directory (Data & Setup)

*   **`requirements.txt`**
    *   **Purpose**: The dependency blueprint.
    *   **Workflow**: When you run `pip install -r requirements.txt`, it downloads FastAPI (for the server), LangChain (for RAG), FAISS (for vector storage), Groq (for LLM), and Rank-BM25 (for hybrid search) to build the environment.
*   **`load_dataset.py`**
    *   **Purpose**: The ingestion script.
    *   **Workflow**: You run this script once to populate the database. It reads a large JSON file of district facts, passes it to the `DocumentLoader`, splits it using `Chunker`, converts text to vectors via `Embeddings`, and saves it all to disk using `VectorStore`.

---

## 2. Core Backend (`app/`)

*   **`main.py`**
    *   **Purpose**: The API Server Entrypoint.
    *   **Workflow**: It creates the FastAPI application. It loads the FAISS index into memory when it starts up. It defines `/health` (returns database stats) and `/chat` (takes user input, runs the RAG chain, and streams the answer). It also mounts the `static/` folder to serve the frontend UI.
*   **`config.py`**
    *   **Purpose**: The Central Brain for Settings.
    *   **Workflow**: Imported everywhere. If you need to change the LLM model, the maximum conversation length, the chunk size, or edit the AI's exact system prompt, you do it here. No hardcoding in logic files.
*   **`rag_chain.py`**
    *   **Purpose**: The Master Pipeline Orchestrator.
    *   **Workflow**: When `main.py` calls `ask_question()`, this file triggers the services in exact order: `SessionManager` → `QueryClassifier` → `QuestionRewriter` → `HybridRetriever` → `Reranker` (if enabled) → `PromptBuilder` → `LLMService`. It stitches the isolated services together into one pipeline.
*   **`vector_store.py`**
    *   **Purpose**: The Database Manager.
    *   **Workflow**: Wraps `faiss-cpu`. It takes text chunks, uses `embeddings.py` to turn them into numbers, and saves them to the `faiss_index` folder. When asked, it performs mathematical similarity searches to find the closest text to a given question.
*   **`embeddings.py`**
    *   **Purpose**: The Text-to-Number Translator.
    *   **Workflow**: Downloads and uses HuggingFace's `all-MiniLM-L6-v2` model locally on your CPU. It converts sentences into a 384-dimensional vector so `FAISS` can do math on them.
*   **`chunker.py`**
    *   **Purpose**: The Text Splitter.
    *   **Workflow**: Documents are too big for the LLM. This file takes a huge document and safely slices it into 250-word chunks (with a 30-word overlap) without cutting sentences in half.
*   **`document_loader.py`**
    *   **Purpose**: The File Reader.
    *   **Workflow**: Capable of reading raw text from PDFs, Docx, and JSON files and extracting the pure string data for the chunker.
*   **`mcp_client.py` & `mcp_server.py`**
    *   **Purpose**: The Booking Tool Handlers (External Protocol).
    *   **Workflow**: A separate micro-protocol. If a user asks to "book a ticket", the logic bypasses standard RAG and uses these files to connect to a mocked booking interface.
*   **`llm.py` & `memory_service.py` (Legacy Wrappers)**
    *   **Purpose**: Backward Compatibility.
    *   **Workflow**: Old files that just say `from app.services.llm_service import ...`. They exist so older imports don't break.

---

## 3. Deep-Dive: Enterprise Services (`app/services/`)
*This directory contains the highly modular "brain" of the chatbot. Each file handles exactly one responsibility, ensuring the system never crashes if a single step fails.*

### `session_manager.py`
*   **Purpose**: Coordinates Memory and API.
*   **Workflow**: It acts as the bridge between `rag_chain.py` and the database. When a user starts a chat (`session_id`), it asks `memory_service.py` for the history. It exposes clean methods like `add_user_message()` and `add_assistant_message()`. If the memory service crashes for any reason, this file intercepts the error and returns a clean, empty history so the chat can still continue.

### `memory_service.py`
*   **Purpose**: The Conversation Storage.
*   **Workflow**: It maintains an in-memory dictionary of `session_id` mapping to an array of chat messages (e.g., `[{"role": "user", "content": "hi"}, ...]`). 
*   **Key Logic**: It enforces the `HISTORY_LIMIT`. If the user has sent 10 messages but the limit is 6, it slices the array to only keep the most recent 6. It also tracks a `summary` string. When the threshold is reached, it passes the oldest messages to the `Summarizer` and deletes them, replacing them with the summary.

### `query_classifier.py`
*   **Purpose**: The Fast Traffic Cop (0ms Latency).
*   **Workflow**: Before running any heavy AI searches, it scans the user's string using Python Lists and regex (e.g., looking for "hi", "thanks", or "book ticket"). 
*   **Key Logic**: If it detects a Greeting or Small Talk, it skips the entire FAISS database and LLM generation, instantly returning a hardcoded friendly response. If it detects a Booking, it routes to `main.py`'s booking logic. Otherwise, it returns `QueryType.DOCUMENT`, signaling `rag_chain.py` to proceed with full AI search.

### `question_rewriter.py`
*   **Purpose**: The Pronoun Resolver.
*   **Workflow**: If the user asks a follow-up question like "Who is the collector there?", the AI doesn't know where "there" is. This service takes the chat history (e.g., "Tell me about Theni district") and the new question, and asks the Groq LLM to rewrite it.
*   **Key Logic**: It creates a prompt: `"History: Tell me about Theni... Question: Who is the collector there?"`. Groq returns: `"Who is the collector of Theni district?"`. This fully-contextualized string is passed forward. If Groq fails or the question is too short, it falls back gracefully to the original string.

### `retriever.py`
*   **Purpose**: The Hybrid Search Engine.
*   **Workflow**: This is the heart of the RAG system. It doesn't rely on just one search method.
*   **Key Logic**: It takes the rewritten question and asks two systems:
    1. **FAISS**: "Find me paragraphs that have the same semantic meaning." (Dense Search)
    2. **BM25**: "Find me paragraphs that contain these exact keywords." (Sparse Search)
    It merges the results from both, removes any duplicate paragraphs, and returns a high-quality list of the top ~10 document chunks.

### `reranker.py`
*   **Purpose**: The Precision Scorer.
*   **Workflow**: If enabled in `config.py`, it takes the top 10 chunks from the Retriever and passes them through a HuggingFace CrossEncoder model. 
*   **Key Logic**: FAISS is fast but sometimes inaccurate. A CrossEncoder is slow but highly accurate. This file re-scores the 10 chunks against the user's question, sorts them by true relevance, and cuts off the bad ones, ensuring the LLM only reads the absolute best context.

### `prompt_builder.py`
*   **Purpose**: The Template Architect.
*   **Workflow**: It acts as a massive string formatter. It reads `SYSTEM_PROMPT` from `config.py` and injects variables into it.
*   **Key Logic**: It weaves together:
    - `{conversation_summary}` (The compressed old memory)
    - `{conversation_history}` (The recent chat bubbles)
    - `{context}` (The 5 paragraphs found by the Retriever)
    - `{question}` (The rewritten question)
    It returns one massive, structured prompt ready for Groq to read.

### `llm_service.py`
*   **Purpose**: The Groq Communicator.
*   **Workflow**: The ONLY file in the project that makes network calls to `api.groq.com`. 
*   **Key Logic**: It initializes the Groq client and has two main methods:
    1. `generate()`: Used by the Rewriter and Summarizer to get a fast, complete text string back.
    2. `stream()`: Used by `rag_chain.py` to yield words chunk-by-chunk so the React frontend can display the typing animation.

### `summarizer.py`
*   **Purpose**: The Token Saver.
*   **Workflow**: When `memory_service.py` detects a conversation is getting too long, it sends an array of old messages to this file.
*   **Key Logic**: It places the old messages into a `SUMMARY_PROMPT` and asks Groq to compress them into a short 1-2 sentence paragraph (e.g., "The user asked about Theni's climate and the assistant explained the rainfall patterns."). It then returns this string to be stored forever as the session's background context.

### `__init__.py`
*   **Purpose**: Package definition.
*   **Workflow**: Tells Python that `app/services` is a module, allowing the rest of the application to import these files cleanly (e.g., `from app.services.llm_service import llm_service`).

---

## 4. Frontend Application (`frontend/`)

*   **`package.json` & `vite.config.js`**
    *   **Purpose**: Node setup files.
    *   **Workflow**: Handles npm dependencies (React, Axios, Tailwind) and instructions for how to build the source code into the final `dist` folder.
*   **`src/main.jsx`**
    *   **Purpose**: React Entry.
    *   **Workflow**: Grabs the `<div id="root">` in `index.html` and renders the entire App inside it.
*   **`src/App.jsx`**
    *   **Purpose**: The Main Container.
    *   **Workflow**: Holds the layout (Sidebar on left, Chat on right). It manages the `messages` state array. When a user sends a message, it adds a temporary "loading" bubble, calls the API, and appends the streaming words as they arrive.
*   **`src/api/api.js`**
    *   **Purpose**: The Network Layer.
    *   **Workflow**: Generates a random `session_id` when the page loads. When a user asks a question, it uses `fetch()` to hit `http://localhost:8000/chat`, reads the streaming response chunk-by-chunk using a `TextDecoder`, and sends the chunks back to `App.jsx`.
*   **`src/components/Sidebar.jsx`**
    *   **Purpose**: The Dashboard.
    *   **Workflow**: Pings the `/health` API endpoint every 15 seconds. If it gets a response, it turns the "Backend API" dot green and displays the total number of FAISS vectors loaded.
*   **`src/components/ChatInput.jsx`**
    *   **Purpose**: The Text Box.
    *   **Workflow**: Captures user typing. Prevents sending empty messages. Binds the "Enter" key to send the payload up to `App.jsx`.
*   **`src/components/ChatWindow.jsx`**
    *   **Purpose**: The Message Renderer.
    *   **Workflow**: Loops through the `messages` state array and renders either a blue "user" bubble or a dark "assistant" bubble. It uses a React Ref (`useRef`) to automatically scroll to the bottom of the screen when new text streams in.
*   **`src/index.css`**
    *   **Purpose**: The Styling Engine.
    *   **Workflow**: Imports Tailwind CSS, defines root CSS variables, and creates all the custom animations (like `.animate-fade-in` and `.glass-light`) that make the UI look premium.

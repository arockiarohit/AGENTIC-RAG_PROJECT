"""
Central configuration for the District RAG Chatbot.
All constants, model settings, and prompts are defined here.
"""

import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
FAISS_INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

# --- Chunking Config ---
CHUNK_SIZE = 250
CHUNK_OVERLAP = 30

# --- Retrieval Settings ---
TOP_K = 5
BM25_TOP_K = 5

# --- LLM Settings ---
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")
TEMPERATURE = 0.0
MAX_TOKENS = 512

# --- Gemini Settings ---
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# --- Embedding Model ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Conversation Memory ---
HISTORY_LIMIT = 6          # Max recent exchanges (user+assistant pairs) to keep
SUMMARY_THRESHOLD = 6      # Trigger summarization after this many exchanges

# --- Feature Toggles ---
HYBRID_SEARCH_ENABLED = True
RERANKER_ENABLED = False    # Set True to enable cross-encoder reranking
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_TOP_K = 3

# --- Allowed File Extensions ---
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".json"}

# --- Rewrite Prompt ---
REWRITE_PROMPT = """You are a question rewriter. Your task is to convert a follow-up question into a standalone question using the conversation history.

Rules:
1. If the question is already standalone, return it as-is.
2. Do NOT answer the question.
3. Return ONLY the rewritten question, nothing else.
4. Keep the rewritten question concise.

Conversation History:
{history}

Follow-up Question: {question}

Standalone Question:"""

# --- Summary Prompt ---
SUMMARY_PROMPT = """You are a conversation summarizer. Summarize the following conversation into a concise paragraph that captures the key topics discussed, questions asked, and information provided.

Rules:
1. Be concise but comprehensive.
2. Focus on facts and topics, not conversational flow.
3. Return ONLY the summary, nothing else.

{existing_summary}

Recent Conversation:
{conversation}

Summary:"""

# --- System Prompt ---
SYSTEM_PROMPT = """You are District Assistant.

Your role is to help citizens with district-related information such as:
- Government schemes
- District offices
- Public services
- Certificates
- Contact information
- Welfare programs
- District administration information

Rules:

1. Never invent facts.
2. Never hallucinate.
3. Answer only using the provided district document context.
4. If the information is not available in the context, reply EXACTLY:
   "I'm sorry, but this information is not available in the district records."
5. Do NOT explain why you cannot answer.
6. Do NOT mention source documents, context, or relevance scores.
7. Give direct and concise answers.
8. Never use outside knowledge.
9. Never guess.
10. Never infer.
11. Never use asterisks or markdown formatting.
12. Use the Normalized Question to understand the user's intent when the original question is written in Tamil, Tanglish, or mixed Tamil-English.
13. Preserve the meaning of the Original User Question.
14. Respond in the same language/style as the Original User Question whenever possible.
15. Do not translate names of people, places, organizations, or other proper nouns unnecessarily.

{conversation_summary}

{conversation_history}

Original User Question:
{question}

Normalized Question:
{normalized_query}

User Language:
{language}

Context:
{context}

Answer:
"""
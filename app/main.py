"""
FastAPI Main Application.
District RAG Chatbot — API endpoints for upload, chat, and health check.
"""

import os
import re
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()
from app.vector_store import vector_store
from app.rag_chain import ask_question

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- FastAPI App ---
app = FastAPI(
    title="District RAG Chatbot API",
    description="A RAG-powered chatbot for answering questions from district documents.",
    version="2.0.0"
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static Files (Frontend) ---
STATIC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "static"
)

if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

GENERATED_IMAGE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "generated_images"
)

if not os.path.exists(GENERATED_IMAGE_DIR):
    os.makedirs(GENERATED_IMAGE_DIR)

app.mount(
    "/generated_images",
    StaticFiles(directory=GENERATED_IMAGE_DIR),
    name="generated_images"
)


# --- Request/Response Models ---
class ChatRequest(BaseModel):
    question: str
    session_id: str



class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    context_used: int


class HealthResponse(BaseModel):
    status: str
    total_vectors: int
    total_documents: int


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check():
    stats = vector_store.get_stats()

    return HealthResponse(
        status="healthy",
        total_vectors=stats["total_vectors"],
        total_documents=stats["total_documents"]
    )

@app.get("/python-info")
async def python_info():
    import sys

    return {
        "python": sys.executable,
        "path": sys.path
    }

@app.get("/")
async def root():
    index_path = os.path.join(STATIC_DIR, "index.html")

    if os.path.exists(index_path):
        return FileResponse(index_path)

    return {
        "message": "District RAG Chatbot API is running",
        "docs": "http://127.0.0.1:8000/docs",
        "warning": "index.html not found in static folder"
    }


@app.post("/chat")
async def chat(request: ChatRequest):

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    # Check whether RAG documents are available
    stats = vector_store.get_stats()

    if stats["total_vectors"] == 0:
        return {
            "answer": (
                "No documents have been uploaded yet. "
                "Please upload district documents first."
            ),
            "sources": [],
            "route": "general",
            "image_url": None
        }

    try:
        # Import LangGraph
        from app.graph.workflow import graph

        # Execute LangGraph
        result = graph.invoke({
            "question": request.question,
            "session_id": request.session_id,
            "route": "",
            "context": "",
            "answer": "",
            "sources": [],
            "image_path": None
        })

        answer = result.get(
            "answer",
            "Error generating response."
        )

        image_path = result.get("image_path")

        # Build response
        response_data = {
            "answer": answer,
            "sources": result.get("sources", []),
            "route": result.get("route", "general"),
            "image_url": None
        }

        # If image was generated
        if image_path:

            filename = os.path.basename(image_path)

            response_data["image_url"] = (
                f"/generated_images/{filename}"
            )

            logger.info(
                f"[FastAPI] Generated image URL: "
                f"{response_data['image_url']}"
            )

        return response_data

    except Exception as e:

        logger.exception(
            f"[FastAPI] LangGraph execution failed: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to process the question."
        )

@app.on_event("startup")
async def startup_event():
    """
    Pre-load the embedding model and dataset on startup.
    """

    logger.info("=" * 50)
    logger.info("District RAG Chatbot API Starting...")
    logger.info("=" * 50)

    from app.embeddings import EmbeddingService

    EmbeddingService.get_model()

    stats = vector_store.get_stats()

    if stats["total_vectors"] == 0:
        logger.info("Index is empty. Loading project dataset...")

        try:
            from load_dataset import load_dataset

            load_dataset()

            stats = vector_store.get_stats()

        except Exception as e:
            logger.error("Failed to load dataset: %s", str(e))

    logger.info(
        "Index loaded: %d vectors, %d document chunks",
        stats['total_vectors'],
        stats['total_documents']
    )

    logger.info("=" * 50)
    logger.info("API is ready! Docs at: http://localhost:8000/docs")
    logger.info("=" * 50)

    
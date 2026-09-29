"""
RAG Chain module.
Orchestrates the full Enterprise Conversational RAG pipeline.

Pipeline:
    Load Memory → Classify Query → Rewrite Question → Hybrid Retrieval →
    Rerank → Build Prompt → Stream LLM Response → Update Memory → Summarize
"""

import time
import logging
from typing import Generator

from app.services.session_manager import session_manager
from app.services.query_classifier import query_classifier, QueryType
from app.services.question_rewriter import question_rewriter
from app.services.retriever import hybrid_retriever
from app.services.reranker import reranker
from app.services.prompt_builder import prompt_builder
from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)


def extract_sources(results: list[dict]) -> list[dict]:
    """
    Extract source information from search results.

    Args:
        results: List of search results.

    Returns:
        List of source references.
    """
    sources: list[dict] = []
    seen: set[str] = set()

    for result in results:
        source: str = result.get("metadata", {}).get("source", "Unknown")
        chunk_idx: int = result.get("metadata", {}).get("chunk_index", 0)
        score: float = result.get("score", 0)

        key: str = f"{source}_{chunk_idx}"

        if key not in seen:
            seen.add(key)
            sources.append({
                "filename": source,
                "chunk": chunk_idx,
                "relevance": round(score, 3),
            })

    return sources


def ask_question(session_id: str, question: str) -> Generator[str, None, None]:
    """
    Full Enterprise RAG pipeline:
        1. Load Memory
        2. Classify Query
        3. Rewrite Question
        4. Hybrid Retrieval
        5. Rerank
        6. Build Prompt
        7. Stream LLM Response
        8. Update Memory
        9. Summarize if needed

    Args:
        session_id: Unique session identifier.
        question: The user's question.

    Yields:
        Text chunks of the LLM response.
    """
    total_start: float = time.time()

    # ------------------------------------------------------------------
    # Step 1: Load Memory
    # ------------------------------------------------------------------
    try:
        context_data: dict = session_manager.get_session_context(session_id)
        history: list[dict] = context_data["history"]
        summary: str = context_data["summary"]
    except Exception as e:
        logger.error("[RAG] Failed to load memory: %s", str(e))
        history = []
        summary = ""

    logger.info(
        "[RAG] Session %s: %d history messages, %d char summary",
        session_id, len(history), len(summary)
    )

    # Save current user message
    session_manager.add_user_message(session_id, question)

    # ------------------------------------------------------------------
    # Step 2: Classify Query
    # ------------------------------------------------------------------
    try:
        query_type: QueryType = query_classifier.classify(question)
    except Exception as e:
        logger.error("[RAG] Classification failed: %s", str(e))
        query_type = QueryType.DOCUMENT

    # Handle non-document queries with canned responses
    if query_type in (QueryType.GREETING, QueryType.SMALL_TALK, QueryType.IDENTITY):
        canned: str | None = query_classifier.get_canned_response(query_type)
        if canned:
            # Save assistant response
            session_manager.add_assistant_message(session_id, canned)
            logger.info(
                "[RAG] Handled %s query in %.2f seconds",
                query_type.value, time.time() - total_start
            )
            yield canned
            return

    # ------------------------------------------------------------------
    # Step 3: Rewrite Question
    # ------------------------------------------------------------------
    rewrite_start: float = time.time()
    try:
        rewritten_question: str = question_rewriter.rewrite_question(
            question=question,
            history=history,
        )
    except Exception as e:
        logger.error("[RAG] Question rewriting failed: %s", str(e))
        rewritten_question = question

    rewrite_time: float = time.time() - rewrite_start
    logger.info(
        "[PERFORMANCE] Rewrite Time: %.2f seconds | '%s' -> '%s'",
        rewrite_time, question, rewritten_question
    )

    # ------------------------------------------------------------------
    # Step 4: Hybrid Retrieval
    # ------------------------------------------------------------------
    retrieval_start: float = time.time()
    try:
        results: list[dict] = hybrid_retriever.retrieve(rewritten_question)
    except Exception as e:
        logger.error("[RAG] Retrieval failed: %s", str(e))
        results = []

    retrieval_time: float = time.time() - retrieval_start
    logger.info(
        "[PERFORMANCE] Retrieval Time: %.2f seconds | %d results",
        retrieval_time, len(results)
    )

    # ------------------------------------------------------------------
    # Step 5: Rerank
    # ------------------------------------------------------------------
    rerank_start: float = time.time()
    try:
        results = reranker.rerank(
            query=rewritten_question,
            results=results,
        )
    except Exception as e:
        logger.error("[RAG] Reranking failed: %s", str(e))

    rerank_time: float = time.time() - rerank_start
    logger.info(
        "[PERFORMANCE] Rerank Time: %.2f seconds | %d results after rerank",
        rerank_time, len(results)
    )

    # ------------------------------------------------------------------
    # Step 6: Build Prompt
    # ------------------------------------------------------------------
    context: str = prompt_builder.build_context(results)

    prompt: str = prompt_builder.build_prompt(
        question=rewritten_question,
        context=context,
        history=history,
        summary=summary,
    )

    logger.info(
        "[RAG] Context: %d chars from %d chunks | Prompt: %d chars",
        len(context), len(results), len(prompt)
    )

    # ------------------------------------------------------------------
    # Step 7: Stream LLM Response
    # ------------------------------------------------------------------
    llm_start: float = time.time()
    answer: str = ""

    try:
        for chunk in llm_service.stream(prompt):
            answer += chunk
            yield chunk
    except Exception as e:
        error_msg: str = f"Error generating response: {str(e)}"
        logger.error("[RAG] LLM streaming failed: %s", str(e))
        answer = error_msg
        yield error_msg

    llm_time: float = time.time() - llm_start
    logger.info("[PERFORMANCE] LLM Time: %.2f seconds", llm_time)

    # ------------------------------------------------------------------
    # Step 8: Update Memory (saves assistant message + summarizes)
    # ------------------------------------------------------------------
    try:
        session_manager.add_assistant_message(session_id, answer)
    except Exception as e:
        logger.error("[RAG] Memory update failed: %s", str(e))

    # ------------------------------------------------------------------
    # Performance Summary
    # ------------------------------------------------------------------
    total_time: float = time.time() - total_start
    logger.info(
        "[PERFORMANCE] Total Pipeline: %.2f seconds "
        "(rewrite=%.2f, retrieval=%.2f, rerank=%.2f, llm=%.2f)",
        total_time, rewrite_time, retrieval_time, rerank_time, llm_time
    )
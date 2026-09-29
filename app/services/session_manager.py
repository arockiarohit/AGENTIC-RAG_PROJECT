"""
Session Manager Service.
Orchestrates session lifecycle — memory retrieval, message tracking,
and automatic summarization.
"""

import logging

from app.services.memory_service import memory_service

logger = logging.getLogger(__name__)


class SessionManagerService:
    """
    Thin orchestration layer for session lifecycle.

    Provides a clean interface for the RAG chain to interact with
    session state without knowing the internal details of memory
    management and summarization.
    """

    def get_session_context(self, session_id: str) -> dict:
        """
        Get the full session context including history and summary.

        Args:
            session_id: Unique session identifier.

        Returns:
            Dict with 'history' (list[dict]) and 'summary' (str) keys.
        """
        history: list[dict] = memory_service.get_history(session_id)
        summary: str = memory_service.get_summary(session_id)

        logger.info(
            "[SESSION] Loaded context for %s: %d messages, "
            "summary=%d chars",
            session_id, len(history), len(summary)
        )

        return {
            "history": history,
            "summary": summary,
        }

    def add_user_message(self, session_id: str, question: str) -> None:
        """
        Record a user message in the session.

        Args:
            session_id: Unique session identifier.
            question: The user's question.
        """
        memory_service.add_message(
            session_id=session_id,
            role="user",
            content=question,
        )

    def add_assistant_message(self, session_id: str, answer: str) -> None:
        """
        Record an assistant response and trigger summarization if needed.

        Args:
            session_id: Unique session identifier.
            answer: The assistant's response.
        """
        memory_service.add_message(
            session_id=session_id,
            role="assistant",
            content=answer,
        )

        # Trigger summarization if history exceeds threshold
        try:
            memory_service.summarize_old_messages(session_id)
        except Exception as e:
            logger.error(
                "[SESSION] Post-response summarization failed for %s: %s",
                session_id, str(e)
            )

    def update_session(
        self,
        session_id: str,
        question: str,
        answer: str,
    ) -> None:
        """
        Convenience method to record both user question and assistant answer.

        Args:
            session_id: Unique session identifier.
            question: The user's question.
            answer: The assistant's response.
        """
        self.add_user_message(session_id, question)
        self.add_assistant_message(session_id, answer)

    def clear_session(self, session_id: str) -> None:
        """
        Clear all data for a session.

        Args:
            session_id: Unique session identifier.
        """
        memory_service.clear(session_id)
        logger.info("[SESSION] Session %s cleared.", session_id)


# --------------------------------------------------------------------------
# Global singleton instance
# --------------------------------------------------------------------------
session_manager: SessionManagerService = SessionManagerService()

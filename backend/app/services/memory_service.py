"""
Memory Service.
Manages per-session conversation history with automatic summarization.
"""

import logging
from collections import defaultdict
from typing import Optional

from app.config import HISTORY_LIMIT, SUMMARY_THRESHOLD

logger = logging.getLogger(__name__)


class MemoryService:
    """
    Per-session conversation memory with automatic summarization.

    Stores recent messages and a running conversation summary.
    When the message count exceeds SUMMARY_THRESHOLD, older messages
    are summarized and replaced with a compact summary.
    """

    def __init__(self) -> None:
        """Initialize session storage."""
        self._sessions: dict[str, list[dict]] = defaultdict(list)
        self._summaries: dict[str, str] = {}

    def get_history(self, session_id: str) -> list[dict]:
        """
        Get the conversation history for a session.

        Args:
            session_id: Unique session identifier.

        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        return list(self._sessions[session_id])

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str
    ) -> None:
        """
        Add a message to the session history.

        Args:
            session_id: Unique session identifier.
            role: Message role ('user' or 'assistant').
            content: Message content.
        """
        self._sessions[session_id].append({
            "role": role,
            "content": content,
        })
        logger.debug(
            "[MEMORY] Added %s message to session %s", role, session_id
        )

    def clear(self, session_id: str) -> None:
        """
        Clear all data for a session.

        Args:
            session_id: Unique session identifier.
        """
        self._sessions.pop(session_id, None)
        self._summaries.pop(session_id, None)
        logger.info("[MEMORY] Cleared session %s", session_id)

    def get_summary(self, session_id: str) -> str:
        """
        Get the conversation summary for a session.

        Args:
            session_id: Unique session identifier.

        Returns:
            The stored summary string, or empty string if none.
        """
        return self._summaries.get(session_id, "")
    
    def set_summary(self, session_id: str, summary: str) -> None:
        """
        Store a conversation summary for a session.

        Args:
            session_id: Unique session identifier.
            summary: The summary text to store.
        """
        self._summaries[session_id] = summary
        logger.debug(
            "[MEMORY] Updated summary for session %s (%d chars)",
            session_id, len(summary)
        )

    def limit_history(self, session_id: str) -> None:
        """
        Trim the session history to the configured HISTORY_LIMIT.
        Keeps only the most recent messages (HISTORY_LIMIT * 2 individual
        messages, since each exchange is a user + assistant pair).

        Args:
            session_id: Unique session identifier.
        """
        messages: list[dict] = self._sessions[session_id]
        max_messages: int = HISTORY_LIMIT * 2  # Each exchange = 2 messages

        if len(messages) > max_messages:
            self._sessions[session_id] = messages[-max_messages:]
            logger.info(
                "[MEMORY] Trimmed session %s from %d to %d messages",
                session_id, len(messages), max_messages
            )

    def summarize_old_messages(self, session_id: str) -> None:
        """
        Summarize older messages when history exceeds SUMMARY_THRESHOLD.

        Keeps the most recent HISTORY_LIMIT exchanges and summarizes
        everything older into the running summary.

        Args:
            session_id: Unique session identifier.
        """
        messages: list[dict] = self._sessions[session_id]
        max_messages: int = SUMMARY_THRESHOLD * 2

        if len(messages) <= max_messages:
            return

        try:
            # Late import to avoid circular dependency
            from app.services.summarizer import summarizer

            # Split: older messages to summarize, recent to keep
            keep_count: int = HISTORY_LIMIT * 2
            messages_to_summarize: list[dict] = messages[:-keep_count]
            messages_to_keep: list[dict] = messages[-keep_count:]

            # Get existing summary
            existing_summary: str = self.get_summary(session_id)

            # Generate new summary
            new_summary: str = summarizer.summarize_conversation(
                messages=messages_to_summarize,
                existing_summary=existing_summary,
            )

            # Update state
            self.set_summary(session_id, new_summary)
            self._sessions[session_id] = messages_to_keep

            logger.info(
                "[MEMORY] Summarized %d old messages for session %s",
                len(messages_to_summarize), session_id
            )

        except Exception as e:
            logger.error(
                "[MEMORY] Summarization failed for session %s: %s",
                session_id, str(e)
            )
            # Fallback: just trim the history
            self.limit_history(session_id)

    def get_recent_history_text(self, session_id: str) -> str:
        """
        Format recent history as a readable text block.

        Args:
            session_id: Unique session identifier.

        Returns:
            Formatted conversation history string.
        """
        messages: list[dict] = self.get_history(session_id)

        if not messages:
            return ""

        parts: list[str] = []
        for msg in messages:
            role: str = msg.get("role", "unknown").capitalize()
            content: str = msg.get("content", "")
            parts.append(f"{role}: {content}")

        return "\n".join(parts)


# --------------------------------------------------------------------------
# Global singleton instance
# --------------------------------------------------------------------------
memory_service: MemoryService = MemoryService()

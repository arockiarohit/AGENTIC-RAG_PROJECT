"""
Summarizer Service.
Generates conversation summaries using LLM when history exceeds threshold.
"""

import logging
from typing import Optional

from app.config import SUMMARY_PROMPT

logger = logging.getLogger(__name__)


class SummarizerService:
    """Generates conversation summaries from message history."""

    def summarize_conversation(
        self,
        messages: list[dict],
        existing_summary: str = ""
    ) -> str:
        """
        Summarize a list of conversation messages into a concise paragraph.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            existing_summary: Previously generated summary to incorporate.

        Returns:
            A concise summary string. Returns existing_summary on failure.
        """
        if not messages:
            return existing_summary

        try:
            # Late import to avoid circular dependency
            from app.services.llm_service import llm_service

            # Format conversation for the prompt
            conversation_text: str = self._format_messages(messages)

            # Build the existing summary section
            summary_section: str = ""
            if existing_summary:
                summary_section = (
                    f"Previous Summary:\n{existing_summary}\n\n"
                )

            prompt: str = SUMMARY_PROMPT.format(
                existing_summary=summary_section,
                conversation=conversation_text,
            )

            logger.info(
                "[SUMMARIZER] Summarizing %d messages", len(messages)
            )

            summary: str = llm_service.summarize(prompt)

            if summary and len(summary.strip()) > 10:
                logger.info(
                    "[SUMMARIZER] Generated summary of %d chars",
                    len(summary)
                )
                return summary.strip()

            logger.warning(
                "[SUMMARIZER] Summary too short, keeping existing."
            )
            return existing_summary

        except Exception as e:
            logger.error("[SUMMARIZER] Failed to summarize: %s", str(e))
            return existing_summary

    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        """
        Format messages into a readable conversation string.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.

        Returns:
            Formatted conversation string.
        """
        parts: list[str] = []

        for msg in messages:
            role: str = msg.get("role", "unknown").capitalize()
            content: str = msg.get("content", "")
            parts.append(f"{role}: {content}")

        return "\n".join(parts)


# --------------------------------------------------------------------------
# Global singleton instance
# --------------------------------------------------------------------------
summarizer: SummarizerService = SummarizerService()

"""
Question Rewriter Service.
Converts follow-up questions into standalone questions using conversation history.
"""

import logging
import time

from app.config import REWRITE_PROMPT

logger = logging.getLogger(__name__)


class QuestionRewriterService:
    """
    Rewrites follow-up questions into standalone questions.

    Uses the conversation history to resolve pronouns and references,
    producing a question that makes sense without prior context.

    Example:
        History: "Tell me about Theni district." -> "..."
        Question: "Who is the collector?"
        Rewritten: "Who is the collector of Theni district?"
    """

    def rewrite_question(
        self,
        question: str,
        history: list[dict],
    ) -> str:
        """
        Rewrite a follow-up question into a standalone question.

        Args:
            question: The current user question.
            history: List of message dicts with 'role' and 'content' keys.

        Returns:
            The rewritten standalone question, or the original question
            if rewriting fails or is not needed.
        """
        # No history means no context for rewriting
        if not history:
            logger.debug("[REWRITER] No history, returning original question.")
            return question

        # Short questions that are clearly standalone
        if len(question.split()) <= 2:
            logger.debug(
                "[REWRITER] Question too short to rewrite: '%s'", question
            )
            return question

        try:
            # Late import to avoid circular dependency
            from app.services.llm_service import llm_service

            start_time: float = time.time()

            # Format the history for the prompt
            history_text: str = self._format_history(history)

            prompt: str = REWRITE_PROMPT.format(
                history=history_text,
                question=question,
            )

            rewritten: str = llm_service.rewrite(prompt)

            elapsed: float = time.time() - start_time
            logger.info(
                "[REWRITER] Rewrote in %.2f seconds: '%s' -> '%s'",
                elapsed, question, rewritten.strip()
            )

            # Validate the rewritten question
            rewritten = rewritten.strip()
            if rewritten and len(rewritten) > 3:
                return rewritten

            logger.warning(
                "[REWRITER] Rewritten question too short, using original."
            )
            return question

        except Exception as e:
            logger.error("[REWRITER] Failed to rewrite: %s", str(e))
            return question

    @staticmethod
    def _format_history(history: list[dict]) -> str:
        """
        Format the last few messages for the rewrite prompt.

        Args:
            history: Full conversation history.

        Returns:
            Formatted string of recent conversation exchanges.
        """
        # Use only the last 4 messages for rewriting context
        recent: list[dict] = history[-4:]
        parts: list[str] = []

        for msg in recent:
            role: str = msg.get("role", "unknown").capitalize()
            content: str = msg.get("content", "")
            # Truncate long messages for the rewrite prompt
            if len(content) > 200:
                content = content[:200] + "..."
            parts.append(f"{role}: {content}")

        return "\n".join(parts)


# --------------------------------------------------------------------------
# Global singleton instance
# --------------------------------------------------------------------------
question_rewriter: QuestionRewriterService = QuestionRewriterService()

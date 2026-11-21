"""
Prompt Builder Service.
Assembles production-quality prompts from system prompt, conversation context,
retrieved documents, and the current question.
"""

import logging

from app.config import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class PromptBuilderService:
    """
    Builds structured prompts for the LLM.

    Assembles:
    1. System Prompt (with rules and persona)
    2. Conversation Summary (compressed older history)
    3. Recent Conversation History
    4. Retrieved Document Context
    5. Current Question
    """

    def build_prompt(
        self,
        question: str,
        context: str,
        history: list[dict] | None = None,
        summary: str = "",
        normalized_query: str = "",
        language: str = "english",

    ) -> str:
        """
        Build a complete prompt for the LLM.

        Args:
            question: The current user question.
            context: Retrieved document context.
            history: Recent conversation history (list of message dicts).
            summary: Compressed summary of older conversation.
            normalized_query: Normalized English version of the user's query.
            language: Detected language/style of the user's original query.

        Returns:
            A formatted prompt string ready for LLM consumption.
        """
        # Build conversation summary section
        summary_section: str = self._build_summary_section(summary)

        # Build recent history section
        history_section: str = self._build_history_section(history)

        # Assemble the full prompt
        prompt: str = SYSTEM_PROMPT.format(
            conversation_summary=summary_section,
            conversation_history=history_section,
            context=context,
            question=question,
            normalized_query=normalized_query or question,
            language=language,
        )

        logger.info(
            "[PROMPT] Built prompt: question=%d chars, context=%d chars, "
            "history=%d msgs, summary=%d chars, total=%d chars",
            len(question),
            len(context),
            len(history) if history else 0,
            len(summary),
            len(prompt),
        )

        return prompt

    @staticmethod
    def _build_summary_section(summary: str) -> str:
        """
        Format the conversation summary section.

        Args:
            summary: The conversation summary text.

        Returns:
            Formatted summary section, or empty string if no summary.
        """
        if not summary or not summary.strip():
            return ""

        return (
            f"Conversation Summary (earlier topics discussed):\n"
            f"{summary.strip()}\n"
        )

    @staticmethod
    def _build_history_section(history: list[dict] | None) -> str:
        """
        Format the recent conversation history section.

        Args:
            history: List of message dicts with 'role' and 'content' keys.

        Returns:
            Formatted history section, or empty string if no history.
        """
        if not history:
            return ""

        parts: list[str] = ["Recent Conversation:"]

        for msg in history:
            role: str = msg.get("role", "unknown").capitalize()
            content: str = msg.get("content", "")

            # Truncate very long messages in history display
            if len(content) > 300:
                content = content[:300] + "..."

            parts.append(f"{role}: {content}")

        return "\n".join(parts) + "\n"

    @staticmethod
    def build_context(results: list[dict]) -> str:
        """
        Build a context string from retrieved document chunks.

        Args:
            results: List of search results with 'content' and 'metadata'.

        Returns:
            Formatted context string.
        """
        if not results:
            return "No relevant context found in the uploaded documents."

        context_parts: list[str] = []

        for result in results:
            content: str = result.get("content", "")
            if content.strip():
                context_parts.append(content)

        return "\n\n".join(context_parts)


# --------------------------------------------------------------------------
# Global singleton instance
# --------------------------------------------------------------------------
prompt_builder: PromptBuilderService = PromptBuilderService()

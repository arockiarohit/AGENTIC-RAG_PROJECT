"""
LLM module — backward-compatible wrapper.
Delegates to app.services.llm_service for all LLM interactions.
"""

from typing import Generator

from app.services.llm_service import get_llm_response as _get_llm_response


def get_llm_response(prompt: str) -> Generator[str, None, None]:
    """
    Send prompt to Groq and yield response chunks.

    This is a backward-compatible wrapper. All new code should
    use app.services.llm_service.llm_service directly.

    Args:
        prompt: The prompt to send to the LLM.

    Yields:
        Text chunks from the LLM response.
    """
    yield from _get_llm_response(prompt)

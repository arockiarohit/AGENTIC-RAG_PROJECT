"""
Centralized LLM Service.
All Groq API interactions go through this module.
Provides generate, stream, rewrite, and summarize methods.
"""

import os
import time
import logging
from typing import Generator

from groq import Groq
from dotenv import load_dotenv

from app.config import LLM_MODEL, TEMPERATURE, MAX_TOKENS

load_dotenv()

logger = logging.getLogger(__name__)


class LLMService:
    """Centralized service for all LLM interactions via Groq."""

    def __init__(self) -> None:
        """Initialize the Groq client."""
        api_key: str | None = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY not found in environment variables.")
        self._client: Groq = Groq(api_key=api_key)
        self._model: str = LLM_MODEL

    def generate(self, prompt: str, temperature: float | None = None,
                 max_tokens: int | None = None) -> str:
        """
        Single-shot LLM completion.

        Args:
            prompt: The prompt to send to the LLM.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.

        Returns:
            The generated text response.
        """
        try:
            start_time: float = time.time()

            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature if temperature is not None else TEMPERATURE,
                max_tokens=max_tokens or MAX_TOKENS,
            )

            elapsed: float = time.time() - start_time
            logger.info("[LLM] Generate completed in %.2f seconds", elapsed)

            return response.choices[0].message.content or ""

        except Exception as e:
            logger.error("[LLM] Generate error: %s", str(e))
            return f"Error generating response: {str(e)}"

    def stream(self, prompt: str, temperature: float | None = None,
               max_tokens: int | None = None) -> Generator[str, None, None]:
        """
        Streaming LLM completion — yields chunks as they arrive.

        Args:
            prompt: The prompt to send to the LLM.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.

        Yields:
            Text chunks from the LLM response.
        """
        try:
            start_time: float = time.time()

            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature if temperature is not None else TEMPERATURE,
                max_tokens=max_tokens or MAX_TOKENS,
                stream=True,
            )

            for chunk in response:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content

            elapsed: float = time.time() - start_time
            logger.info("[LLM] Stream completed in %.2f seconds", elapsed)

        except Exception as e:
            logger.error("[LLM] Stream error: %s", str(e))
            yield f"Error generating response: {str(e)}"

    def rewrite(self, prompt: str) -> str:
        """
        Low-temperature rewrite call for question rewriting.

        Args:
            prompt: The rewrite prompt with question and history.

        Returns:
            The rewritten question.
        """
        return self.generate(prompt, temperature=0.0, max_tokens=150)

    def summarize(self, prompt: str) -> str:
        """
        Summarization call for conversation summarization.

        Args:
            prompt: The summarization prompt with conversation history.

        Returns:
            The conversation summary.
        """
        return self.generate(prompt, temperature=0.2, max_tokens=300)


# --------------------------------------------------------------------------
# Global singleton instance
# --------------------------------------------------------------------------
llm_service: LLMService = LLMService()


# --------------------------------------------------------------------------
# Backward-compatible wrapper
# --------------------------------------------------------------------------
def get_llm_response(prompt: str) -> Generator[str, None, None]:
    """
    Backward-compatible wrapper for existing code that calls get_llm_response.

    Args:
        prompt: The prompt to send to the LLM.

    Yields:
        Text chunks from the LLM response.
    """
    yield from llm_service.stream(prompt)
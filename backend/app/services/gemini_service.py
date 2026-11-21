"""
Gemini Search Service.
Leverages Google Gemini with live Google search intelligence to provide
accurate, real-time, and up-to-date answers with verified source citations.
"""

import os
import warnings
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv

# Suppress deprecation notice warnings from legacy Google SDK
warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv()

logger = logging.getLogger(__name__)

# Active, supported Gemini models in order of priority
GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest"
]


class GeminiSearchService:
    """Service for querying Google Gemini with live Google Search."""

    def __init__(self) -> None:
        self.api_key: str | None = os.getenv("GEMINI_API_KEY")
        self.primary_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self._genai_client = None
        self._genai_legacy = None
        self._sdk_type: str = "none"
        self._is_available: bool = False
        self._init_sdk()

    def _init_sdk(self) -> None:
        """Initialize the Gemini client using available SDK."""
        if not self.api_key or not self.api_key.strip():
            logger.info("[GeminiService] GEMINI_API_KEY not configured.")
            self._is_available = False
            return

        # 1. Try google-genai (modern SDK)
        try:
            from google import genai
            self._genai_client = genai.Client(api_key=self.api_key)
            self._sdk_type = "genai"
            self._is_available = True
            logger.info("[GeminiService] Initialized with modern google-genai SDK.")
            return
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"[GeminiService] Failed to init google-genai client: {e}")

        # 2. Try google-generativeai (installed SDK)
        try:
            import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=self.api_key)
            self._genai_legacy = genai_legacy
            self._sdk_type = "generativeai"
            self._is_available = True
            logger.info("[GeminiService] Initialized with google-generativeai SDK.")
            return
        except ImportError:
            logger.warning("[GeminiService] Neither google-genai nor google-generativeai is installed.")
            self._is_available = False
        except Exception as e:
            logger.warning(f"[GeminiService] Failed to configure google-generativeai: {e}")
            self._is_available = False

    @property
    def is_available(self) -> bool:
        """Return True if Gemini API key and SDK are available."""
        current_key = os.getenv("GEMINI_API_KEY")
        if current_key and (not self.api_key or not self._is_available):
            self.api_key = current_key
            self._init_sdk()
        return self._is_available

    def search_and_answer(self, query: str, user_language: str = "english") -> Dict[str, Any]:
        """
        Query Google Gemini with live Google Search results.

        Args:
            query: User's question or search query.
            user_language: Preferred response language.

        Returns:
            Dict containing:
                - answer (str): Up-to-date grounded answer.
                - sources (list[dict]): [{"title": ..., "url": ...}]
                - success (bool): Whether the search succeeded.
                - model (str): Name of the Gemini model used.
        """
        if not self.is_available:
            return {
                "answer": "",
                "sources": [],
                "success": False,
                "error": "Gemini API key is not configured or SDK unavailable."
            }

        models_to_try = [self.primary_model] + [m for m in GEMINI_MODELS if m != self.primary_model]

        for model_name in models_to_try:
            try:
                logger.info(f"[GeminiService] Querying Gemini model '{model_name}'...")
                
                if self._sdk_type == "genai":
                    res = self._call_modern_genai(model_name, query)
                else:
                    res = self._call_legacy_generativeai(model_name, query)

                if res.get("success") and res.get("answer"):
                    return res

            except Exception as e:
                logger.warning(f"[GeminiService] Error with model '{model_name}': {e}. Trying next model...")
                continue

        return {
            "answer": "",
            "sources": [],
            "success": False,
            "error": "All Gemini models failed to process query."
        }

    def _call_modern_genai(self, model_name: str, query: str) -> Dict[str, Any]:
        """Call Gemini using the modern google-genai SDK."""
        from google.genai import types

        # 1. Try native Google Search Tool
        try:
            config = types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.2,
                system_instruction=(
                    "You are an intelligent, helpful AI assistant. "
                    "Answer the user's question accurately, directly, and comprehensively "
                    "using real-time Google Search grounding. "
                    "Provide up-to-date, verified facts. "
                    "Respond in the same language and style as the user's question."
                )
            )

            response = self._genai_client.models.generate_content(
                model=model_name,
                contents=query,
                config=config
            )

            answer_text = response.text or ""
            sources: List[Dict[str, str]] = []
            seen_urls = set()

            if response.candidates:
                cand = response.candidates[0]
                gm = getattr(cand, "grounding_metadata", None)
                if gm:
                    chunks = getattr(gm, "grounding_chunks", []) or []
                    for chunk in chunks:
                        web = getattr(chunk, "web", None)
                        if web:
                            uri = getattr(web, "uri", "")
                            title = getattr(web, "title", "Google Search Result") or "Google Search Result"
                            if uri and uri not in seen_urls:
                                seen_urls.add(uri)
                                sources.append({"title": title, "url": uri})

            if answer_text.strip():
                return {
                    "answer": answer_text,
                    "sources": sources,
                    "success": True,
                    "model": model_name
                }
        except Exception as tool_err:
            logger.warning(f"[GeminiService] modern google-genai search tool failed: {tool_err}")

        # 2. Hybrid fallback: SerpApi Google search + Gemini synthesis
        return self._synthesize_with_serpapi(model_name, query, use_modern=True)

    def _call_legacy_generativeai(self, model_name: str, query: str) -> Dict[str, Any]:
        """Call Gemini using the google-generativeai SDK with search grounding or hybrid fallback."""
        # 1. Try native Google Search Retrieval tool
        try:
            model = self._genai_legacy.GenerativeModel(
                model_name=model_name,
                tools="google_search_retrieval",
                system_instruction=(
                    "You are an intelligent, helpful AI assistant. "
                    "Answer the user's question accurately, directly, and comprehensively "
                    "using real-time Google Search grounding. "
                    "Provide up-to-date, verified facts. "
                    "Respond in the same language and style as the user's question."
                )
            )

            response = model.generate_content(query)
            answer_text = getattr(response, "text", "") or ""
            sources: List[Dict[str, str]] = []
            seen_urls = set()

            if hasattr(response, "candidates") and response.candidates:
                cand = response.candidates[0]
                gm = getattr(cand, "grounding_metadata", None)
                if gm:
                    chunks = getattr(gm, "grounding_chunks", []) or []
                    for chunk in chunks:
                        web = getattr(chunk, "web", None)
                        if web:
                            uri = getattr(web, "uri", "")
                            title = getattr(web, "title", "Google Search Result") or "Google Search Result"
                            if uri and uri not in seen_urls:
                                seen_urls.add(uri)
                                sources.append({"title": title, "url": uri})

            if answer_text.strip():
                return {
                    "answer": answer_text,
                    "sources": sources,
                    "success": True,
                    "model": model_name
                }
        except Exception as tool_err:
            logger.warning(f"[GeminiService] Native Google Search tool not available or quota limit hit: {tool_err}. Using live web search + Gemini...")

        # 2. Hybrid fallback: SerpApi Google search + Gemini synthesis
        return self._synthesize_with_serpapi(model_name, query, use_modern=False)

    def _synthesize_with_serpapi(self, model_name: str, query: str, use_modern: bool = False) -> Dict[str, Any]:
        """Fetch live Google results via SerpApi and synthesize the final answer with Gemini."""
        search_snippets = []
        sources = []
        try:
            from app.tools.web_search import web_search
            raw_results = web_search(query)
            for r in raw_results[:5]:
                search_snippets.append(f"Title: {r['title']}\nSource: {r.get('url')}\nContent: {r['content']}")
                sources.append({"title": r["title"], "url": r["url"]})
        except Exception as search_err:
            logger.warning(f"[GeminiService] SerpApi search error: {search_err}")

        context_block = "\n\n".join(search_snippets) if search_snippets else "No live search results available."
        prompt = (
            f"User Question:\n{query}\n\n"
            f"Live Google Search Results:\n{context_block}\n\n"
            f"Instructions:\n"
            f"1. Answer the user's question directly, accurately, and factually based on the live Google Search results and verified facts.\n"
            f"2. Do not say 'Based on the search results' or 'According to the context'.\n"
            f"3. Provide clear, concise, and helpful information.\n"
            f"4. Respond in the same language and style as the user's question whenever possible."
        )

        answer_text = ""
        if use_modern and self._genai_client:
            res = self._genai_client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            answer_text = res.text or ""
        elif self._genai_legacy:
            m = self._genai_legacy.GenerativeModel(
                model_name=model_name,
                system_instruction="You are a helpful, accurate, and professional AI assistant."
            )
            res = m.generate_content(prompt)
            answer_text = getattr(res, "text", "") or ""

        return {
            "answer": answer_text,
            "sources": sources,
            "success": bool(answer_text.strip()),
            "model": model_name
        }


# Global singleton instance
gemini_service = GeminiSearchService()

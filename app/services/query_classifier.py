"""
Query Classifier Service.
Classifies user queries to determine the appropriate handling path.
Uses keyword matching for speed — no LLM call required.
"""

import logging
from enum import Enum

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Classification types for user queries."""
    GREETING = "greeting"
    SMALL_TALK = "small_talk"
    IDENTITY = "identity"
    DOCUMENT = "document"


# --- Keyword Sets ---
GREETING_KEYWORDS: set[str] = {
    "hi", "hello", "hey", "good morning", "good evening",
    "good afternoon", "good night", "greetings", "namaste",
    "vanakkam", "howdy",
}

SMALL_TALK_KEYWORDS: set[str] = {
    "thank you", "thanks", "thx", "good", "nice", "great",
    "awesome", "ok", "okay", "bye", "goodbye", "see you",
    "take care",
}

IDENTITY_KEYWORDS: set[str] = {
    "who are you", "what are you", "what can you do",
    "your name", "introduce yourself", "what is your purpose",
    "what do you do",
}

# --- Canned Responses ---
CANNED_RESPONSES: dict[QueryType, str] = {
    QueryType.GREETING: (
        "Hello! I am the District Information Assistant. "
        "How may I help you today?"
    ),
    QueryType.SMALL_TALK: (
        "You're welcome! Is there anything else "
        "I can help you with?"
    ),
    QueryType.IDENTITY: (
        "I am the District Information Assistant. "
        "I help users find information from district records "
        "such as government schemes, district offices, "
        "public services, certificates, and contact information."
    ),
}


class QueryClassifierService:
    """
    Classifies user queries using keyword matching.

    Categories:
    - greeting: Hi, hello, etc. → canned response, no retrieval
    - small_talk: Thanks, bye, etc. → canned response, no retrieval
    - identity: Who are you, etc. → canned response, no retrieval
    - booking: Book, ticket, etc. → MCP flow (handled in main.py)
    - document: Everything else → full RAG pipeline
    """

    def classify(self, question: str) -> QueryType:
        """
        Classify a user question into a QueryType.

        Args:
            question: The user's question.

        Returns:
            The classified QueryType.
        """
        question_lower: str = question.lower().strip()

        # Check greetings (exact match on full question)
        if question_lower in GREETING_KEYWORDS:
            logger.info("[CLASSIFIER] Classified as GREETING: '%s'", question)
            return QueryType.GREETING

        # Check identity (substring match)
        for keyword in IDENTITY_KEYWORDS:
            if keyword in question_lower:
                logger.info(
                    "[CLASSIFIER] Classified as IDENTITY: '%s'", question
                )
                return QueryType.IDENTITY

        # Check small talk (substring match)
        for keyword in SMALL_TALK_KEYWORDS:
            if keyword in question_lower:
                logger.info(
                    "[CLASSIFIER] Classified as SMALL_TALK: '%s'", question
                )
                return QueryType.SMALL_TALK

        # Default: document question requiring RAG
        logger.info("[CLASSIFIER] Classified as DOCUMENT: '%s'", question)
        return QueryType.DOCUMENT

    def get_canned_response(self, query_type: QueryType) -> str | None:
        """
        Get the canned response for a query type, if applicable.

        Args:
            query_type: The classified query type.

        Returns:
            The canned response string, or None if no canned response exists.
        """
        return CANNED_RESPONSES.get(query_type)


# --------------------------------------------------------------------------
# Global singleton instance
# --------------------------------------------------------------------------
query_classifier: QueryClassifierService = QueryClassifierService()

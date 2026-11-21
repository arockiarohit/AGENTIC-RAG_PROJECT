import logging
import json
import re

from app.graph.state import GraphState
from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)


def classify_question(state: GraphState) -> GraphState:

    question = state["question"]

    logger.info(
        f"[LangGraph] Classifying question: {question}"
    )

    q_lower = question.strip().lower()

    # ---------------------------------------------------------
    # 1. Conversational / basic intents
    # ---------------------------------------------------------

    is_greeting = bool(
        re.search(
            r"^\s*(hi|hello|hey|sup|greetings|good morning|good afternoon|good evening)\b",
            q_lower
        )
    )

    is_farewell = bool(
        re.search(
            r"^\s*(bye|goodbye|thank you|thanks|thx|see you|cya|tq)\b",
            q_lower
        )
    )

    is_identity = bool(
        re.search(
            r"^\s*(who are you|what are you|what can you do)\b",
            q_lower
        )
    )

    is_frustration = any(
        word in q_lower
        for word in [
            "stupid",
            "useless",
            "dumb",
            "idiot"
        ]
    )

    if (
        is_greeting
        or is_farewell
        or is_identity
        or is_frustration
    ):
        logger.info(
            "[LangGraph] Conversational intent detected. Route selected: general"
        )

        return {
            **state,
            "route": "general",
            "location": None,
            "time_range": None
        }

    # ---------------------------------------------------------
    # 2. LLM Query Classification & Entity Extraction
    # ---------------------------------------------------------

    prompt = f"""
You are an intent classifier and entity extractor for an AI Assistant.

Classify the user's question into exactly ONE of these categories:

1. "current_news": ONLY when the user explicitly asks for general news headlines, daily news feeds, or breaking news digests for a location (e.g., "today's news", "news today in Chennai", "latest headlines", "breaking news").
   CRITICAL: If the user asks a specific factual question about a person, politician, death, party, score, event, or incident (e.g., "which MLA died recently in TVK?", "who won the match?", "is X alive?", "what happened to Y?"), DO NOT classify as "current_news". Classify as "current" or "general".
2. "theni": Questions specifically about static district administration, offices, schemes, tourist spots, or static government records contained in the local knowledge base.
3. "current": Real-time web facts, current events, specific questions about people, deaths, politics, product prices, weather, sports scores.
4. "datetime": Questions about today's date, current time, or calculating what day of the week a date falls on (e.g., "what day is 28/10/2026?").
5. "arithmetic": Math calculations.
6. "image": Requests to create/generate an image.
7. "general": General knowledge or conversational questions.

Also extract:
- "location": The target location mentioned (e.g., "Theni", "Chennai", "India", "Global", or null if none).
- "time_range": "today" (if asking for today/current date headlines), "recent" (if asking for recent/latest news), "historical" (if asking for specific past year e.g. 2019/2026 election results), or "any".

JSON OUTPUT FORMAT:
{{
  "route": "current_news",
  "location": "Theni",
  "time_range": "today"
}}

User Question:
{question}
"""

    route = "general"
    location = None
    time_range = "any"

    try:
        response = llm_service.generate(prompt, temperature=0.0)
        logger.info(f"[LangGraph] Classifier response: {response}")

        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            route = data.get("route", "general")
            location = data.get("location")
            time_range = data.get("time_range", "any")

    except Exception as e:
        logger.error(f"[LangGraph] Classification failed: {str(e)}")

    # Deterministic fallback rules for News intent
    is_news_kw = any(kw in q_lower for kw in ["news", "headline", "headlines", "breaking news"])
    is_today_kw = any(kw in q_lower for kw in ["today", "today's", "current headlines"])
    is_recent_kw = any(kw in q_lower for kw in ["recent", "latest"])

    # Check if user asked a specific interrogative question (who, which, what, why, when, how, is, was, did, etc.)
    is_specific_question = bool(re.search(r"^\s*(which|who|what|where|when|why|how|is|was|did|has|have|can|will|tell me about)\b", q_lower))
    is_explicit_news_headline = any(kw in q_lower for kw in ["today's news", "news headlines", "headlines of", "breaking news", "daily news", "news today", "top headlines"])

    # Exclude commercial product queries like "Toyota Rumion price" from current_news
    is_price_product = any(kw in q_lower for kw in ["price", "on road price", "cost", "buy", "hotel booking"])

    if is_news_kw and not is_price_product and (is_explicit_news_headline or not is_specific_question):
        route = "current_news"

    # If classified as current_news but it's actually a specific question, redirect to 'current' for live web search
    if route == "current_news" and is_specific_question and not is_explicit_news_headline:
        logger.info("[LangGraph] Redirecting specific question from 'current_news' to 'current' for live web search.")
        route = "current"

    if is_today_kw:
        time_range = "today"
    elif is_recent_kw and time_range != "today":
        time_range = "recent"

    # Extract location fallback if missing
    if not location or location.lower() in ["none", "null"]:
        if "theni" in q_lower:
            location = "Theni"
        elif "chennai" in q_lower:
            location = "Chennai"
        elif "india" in q_lower:
            location = "India"
        else:
            location = "Global"

    allowed_routes = ["theni", "current_news", "current", "datetime", "arithmetic", "image", "general"]
    if route not in allowed_routes:
        route = "general"

    logger.info(
        f"[LangGraph] Route selected: {route} | Location: {location} | TimeRange: {time_range}"
    )

    return {
        **state,
        "route": route,
        "location": location,
        "time_range": time_range
    }


def route_question(state: GraphState) -> str:

    route = state.get(
        "route",
        "general"
    )

    if route == "theni":
        return "theni"

    elif route == "current_news":
        return "current_news"

    elif route == "current":
        return "current"

    elif route == "datetime":
        return "datetime"

    elif route == "arithmetic":
        return "arithmetic"

    elif route == "image":
        return "image"

    else:
        return "general"
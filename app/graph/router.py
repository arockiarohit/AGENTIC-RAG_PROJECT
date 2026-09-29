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
            "[LangGraph] Conversational intent detected. "
            "Route selected: theni"
        )

        return {
            **state,
            "route": "theni"
        }

    # ---------------------------------------------------------
    # 2. LLM Query Classification
    # ---------------------------------------------------------

    prompt = f"""
You are a query classifier for a District Information Assistant.

Your job is to classify the user's question into exactly ONE
of these four categories:

1. "theni"
2. "general"
3. "current"
4. "arithmetic"
5. "image"

---------------------------------------------------------
CATEGORY 1: THENI
---------------------------------------------------------

Use "theni" when the question is related to:

- Theni District
- Theni tourist places
- Theni geography
- Theni waterfalls
- Theni temples
- Theni district offices
- Theni government services
- Theni government schemes
- Theni agriculture
- Theni infrastructure
- Any information contained in the private Theni knowledge base

Examples:

"What are the tourist places in Theni?"
→ theni

"Which waterfalls are popular in the district?"
→ theni

"What government offices are available in Theni?"
→ theni


---------------------------------------------------------
CATEGORY 2: GENERAL
---------------------------------------------------------

Use "general" for general knowledge questions that are:

- Not related to Theni
- Not time-sensitive
- Not arithmetic

Examples:

"What is LangGraph?"
→ general

"What is Python?"
→ general

"Who invented the telephone?"
→ general


---------------------------------------------------------
CATEGORY 3: CURRENT
---------------------------------------------------------

Use "current" when the question requires
up-to-date or real-time information.

Examples:

"What is the current population of India?"
→ current

"What is today's weather?"
→ current

"What is the latest version of Python?"
→ current

"What are today's news headlines?"
→ current


---------------------------------------------------------
CATEGORY 4: ARITHMETIC
---------------------------------------------------------

Use "arithmetic" when the user asks for a mathematical
calculation.

This includes:

- Addition
- Subtraction
- Multiplication
- Division
- Percentage
- Average
- Power
- Modulo
- Mathematical expressions

Examples:

"What is 25 + 75?"
→ arithmetic

"What is 125 multiplied by 48?"
→ arithmetic

"Calculate 1000 / 25"
→ arithmetic

"What is 20% of 500?"
→ arithmetic

"What is 2 raised to the power of 10?"
→ arithmetic

"Calculate 500 minus 125"
→ arithmetic

5. "image":
   Use this when the user explicitly asks to create,
   generate, draw, design, visualize, or produce an image.

Examples:
"Create an image of a cat"
"Generate a picture of a dog"
"Draw a bird"
"Visualize a sunset"

---------------------------------------------------------
IMPORTANT RULES
---------------------------------------------------------

If the question requires mathematical calculation,
choose "arithmetic".

If the question requires current information,
choose "current".

If the question is about the Theni knowledge base,
choose "theni".

Otherwise choose "general".

Return ONLY valid JSON.

Valid responses are:

{{"route": "theni"}}

{{"route": "general"}}

{{"route": "current"}}

{{"route": "arithmetic"}}

{{"route": "image"}}


Question:
{question}
"""

    try:

        response = llm_service.generate(
            prompt,
            temperature=0.0
        )

        logger.info(
            f"[LangGraph] Classifier response: {response}"
        )

        # -----------------------------------------------------
        # Extract JSON from LLM response
        # -----------------------------------------------------

        json_match = re.search(
            r'\{.*\}',
            response,
            re.DOTALL
        )

        if json_match:

            data = json.loads(
                json_match.group(0)
            )

            route = data.get(
                "route",
                "general"
            )

        else:

            route = "general"

        # -----------------------------------------------------
        # Validate route
        # -----------------------------------------------------

        allowed_routes = [
            "theni",
            "general",
            "current",
            "arithmetic",
            "image"
        ]

        if route not in allowed_routes:
            logger.warning(
                f"[LangGraph] Invalid route returned: {route}"
            )

            route = "general"

    except Exception as e:

        logger.error(
            f"[LangGraph] Classification failed: {str(e)}"
        )

        route = "general"

    logger.info(
        f"[LangGraph] Route selected: {route}"
    )

    return {
        **state,
        "route": route
    }


def route_question(state: GraphState) -> str:

    route = state.get(
        "route",
        "general"
    )

    if route == "theni":
        return "theni"

    elif route == "current":
        return "current"

    elif route == "arithmetic":
        return "arithmetic"

    elif route == "image":
        return "image"

    else:
        return "general"
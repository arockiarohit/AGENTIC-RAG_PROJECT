# app/graph/nodes/query_normalizer.py

import requests
import json


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "qwen2:0.5b"


def normalize_query(query: str) -> dict:

    prompt = f"""
You are a multilingual query normalization system.

The user may write:
- English
- Tamil
- Tanglish (Tamil written using English letters)
- Tamil + English mixed language

Your task is to understand the user's intent and convert the query
into clear English.

Rules:
1. Do NOT answer the question.
2. Only normalize the query.
3. Preserve names of people, places, organizations and products.
4. Preserve dates and time expressions.
5. Understand Tanglish as Tamil written using English characters.
6. Do not invent information.
7. Return ONLY valid JSON.

Return exactly:

{{
    "normalized_query": "...",
    "language": "english|tamil|tanglish|mixed"
}}

User query:
{query}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        },
        timeout=60
    )

    response.raise_for_status()

    result = response.json()

    return json.loads(result["response"])
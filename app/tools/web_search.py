import logging
import os
from typing import List, Dict, Any

import serpapi

logger = logging.getLogger(__name__)


def web_search(query: str) -> List[Dict[str, Any]]:
    """
    Perform a live Google search using SerpApi.
    """

    try:
        api_key = os.getenv("SERPAPI_API_KEY")

        if not api_key:
            logger.error(
                "[Web Search] SERPAPI_API_KEY is not configured."
            )
            return []

        client = serpapi.Client(
            api_key=api_key
        )

        results = client.search({
            "engine": "google",
            "q": query,
            "gl": "in",
            "hl": "en"
        })

        organic_results = results.get(
            "organic_results",
            []
        )

        formatted_results = []

        for result in organic_results[:5]:

            formatted_results.append({
                "title": result.get("title", ""),
                "url": result.get("link", ""),
                "content": result.get("snippet", "")
            })

        logger.info(
            "[Web Search] Retrieved %d Google results.",
            len(formatted_results)
        )

        return formatted_results

    except Exception as e:
        logger.error(
            "[Web Search] Google search failed: %s",
            str(e)
        )
        return []
import logging
import os
import re
from typing import List, Dict, Any

import serpapi

logger = logging.getLogger(__name__)


def is_spam_url(url: str) -> bool:
    """Check if the URL matches known SEO spam or hijacked domain patterns."""
    if not url:
        return True
    spam_pattern = r"/aviator/|/apps-|\.gov\.in/(apps|aviator|pdf|slot|bet|latest-)|/slot/|/casino/|/betting/"
    return bool(re.search(spam_pattern, url, re.IGNORECASE))


def web_search(query: str) -> List[Dict[str, Any]]:
    """
    Perform a live Google search using SerpApi.
    Features:
    - Automatic News mode for current events/headlines.
    - Extract Answer Box and Knowledge Graph for direct factual queries.
    - Filter out fake SEO spam / hijacked domain URLs.
    - Trusted fallback search for factual/record questions.
    """

    try:
        api_key = os.getenv("SERPAPI_API_KEY")

        if not api_key:
            logger.error(
                "[Web Search] SERPAPI_API_KEY is not configured."
            )
            return []

        q_lower = query.lower()
        is_news_query = any(
            kw in q_lower for kw in [
                "news", "headline", "headlines", "today",
                "latest", "recent", "breaking", "update"
            ]
        )

        formatted_results = []
        seen_urls = set()

        if is_news_query:
            # 1. Primary news search using Google News tab
            news_params: Dict[str, Any] = {
                "engine": "google",
                "q": query,
                "tbm": "nws",
                "gl": "in",
                "hl": "en",
                "api_key": api_key
            }
            res_news = serpapi.GoogleSearch(news_params).get_dict()
            for result in res_news.get("news_results", []):
                url = result.get("link", "")
                if url and url not in seen_urls and not is_spam_url(url):
                    seen_urls.add(url)
                    source = result.get("source", "News")
                    date_str = result.get("date", "")
                    snippet = result.get("snippet", "")
                    content_text = f"Source: {source} ({date_str})\nSummary: {snippet}" if snippet else f"Source: {source} ({date_str})"
                    formatted_results.append({
                        "title": result.get("title", ""),
                        "url": url,
                        "content": content_text
                    })

            # 2. If fewer than 5 news items, supplement with broader news search
            if len(formatted_results) < 5:
                q_clean = query.lower()
                for phrase in ["what is the", "tell me", "what are", "today", "headlines of", "headline of"]:
                    q_clean = q_clean.replace(phrase, "").strip()
                fallback_query = f"{q_clean} news headlines".strip()
                
                res_fallback = serpapi.GoogleSearch({
                    "engine": "google",
                    "q": fallback_query,
                    "tbm": "nws",
                    "gl": "in",
                    "hl": "en",
                    "api_key": api_key
                }).get_dict()
                for result in res_fallback.get("news_results", []):
                    url = result.get("link", "")
                    if url and url not in seen_urls and not is_spam_url(url):
                        seen_urls.add(url)
                        source = result.get("source", "News")
                        date_str = result.get("date", "")
                        snippet = result.get("snippet", "")
                        content_text = f"Source: {source} ({date_str})\nSummary: {snippet}" if snippet else f"Source: {source} ({date_str})"
                        formatted_results.append({
                            "title": result.get("title", ""),
                            "url": url,
                            "content": content_text
                        })

        # 3. Standard Google Organic Search & Answer Box
        if not formatted_results:
            results = serpapi.GoogleSearch({
                "engine": "google",
                "q": query,
                "gl": "in",
                "hl": "en",
                "api_key": api_key
            }).get_dict()

            # Answer Box
            ab = results.get("answer_box")
            if ab:
                ans = ab.get("snippet") or ab.get("answer") or ab.get("result")
                link = ab.get("link", "")
                if ans:
                    formatted_results.append({
                        "title": "Google Direct Answer",
                        "url": link or "https://google.com",
                        "content": str(ans)
                    })

            # Knowledge Graph
            kg = results.get("knowledge_graph")
            if kg and kg.get("description"):
                formatted_results.append({
                    "title": kg.get("title", "Knowledge Graph"),
                    "url": kg.get("source", {}).get("link", "https://google.com"),
                    "content": kg.get("description")
                })

            # Organic results
            for result in results.get("organic_results", []):
                url = result.get("link", "")
                if url and url not in seen_urls and not is_spam_url(url):
                    seen_urls.add(url)
                    formatted_results.append({
                        "title": result.get("title", ""),
                        "url": url,
                        "content": result.get("snippet", "")
                    })

        # 4. Fallback search with trusted domain hints for sparse or factual queries
        if len(formatted_results) < 3 and not is_news_query:
            fallback_res = serpapi.GoogleSearch({
                "engine": "google",
                "q": f"{query} Wikipedia ESPNcricinfo stats",
                "gl": "in",
                "hl": "en",
                "api_key": api_key
            }).get_dict()
            for result in fallback_res.get("organic_results", []):
                url = result.get("link", "")
                if url and url not in seen_urls and not is_spam_url(url):
                    seen_urls.add(url)
                    formatted_results.append({
                        "title": result.get("title", ""),
                        "url": url,
                        "content": result.get("snippet", "")
                    })

        logger.info(
            "[Web Search] Retrieved %d clean Google results (News Mode: %s).",
            len(formatted_results),
            is_news_query
        )

        return formatted_results[:8]

    except Exception as e:
        logger.error(
            "[Web Search] Google search failed: %s",
            str(e)
        )
        return []
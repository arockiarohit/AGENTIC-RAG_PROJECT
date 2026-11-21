import os
import requests
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
GNEWS_URL = "https://gnews.io/api/v4/search"


def normalize_location_query(location: str) -> Dict[str, str]:
    """
    Canonical location normalization for Step 2.
    """
    loc_clean = location.strip()
    loc_lower = loc_clean.lower()

    # Common canonical mappings
    mapping = {
        "nyc": ("New York", "New York City"),
        "new york city": ("New York", "New York City"),
        "uk": ("United Kingdom", "UK"),
        "usa": ("United States", "USA"),
        "us": ("United States", "US"),
        "la": ("Los Angeles", "Los Angeles"),
        "los angeles": ("Los Angeles", "Los Angeles"),
        "sf": ("San Francisco", "San Francisco"),
        "san francisco": ("San Francisco", "San Francisco"),
        "tn": ("Tamil Nadu", "Tamil Nadu"),
        "karala": ("Kerala", "Kerala"),
        "kerla": ("Kerala", "Kerala"),
        "chenai": ("Chennai", "Chennai"),
        "madrai": ("Madurai", "Madurai"),
    }

    if loc_lower in mapping:
        canonical, variant = mapping[loc_lower]
        return {"canonical": canonical, "search_term": variant}

    return {"canonical": loc_clean, "search_term": loc_clean}


def get_news(location: str, max_results: int = 15) -> List[Dict[str, Any]]:
    """
    Search GNews API using multi-query discovery for any location globally (Step 5).
    Does NOT hard-code country='in' or any specific location.
    """
    if not GNEWS_API_KEY:
        logger.warning("[GNews] GNEWS_API_KEY is not configured in .env")
        return []

    norm_info = normalize_location_query(location)
    canonical_loc = norm_info["canonical"]
    search_term = norm_info["search_term"]

    # Generate targeted multi-queries dynamically based on location (Step 5)
    queries = [
        search_term,
        f"{search_term} news today",
        f"{search_term} local news",
        f"{search_term} politics business safety"
    ]

    candidate_articles = []
    seen_urls = set()
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=7)

    for q in queries:
        if len(candidate_articles) >= max_results:
            break

        params = {
            "q": q,
            "lang": "en",
            "max": 10,
            "sortby": "publishedAt",
            "apikey": GNEWS_API_KEY,
        }

        try:
            response = requests.get(GNEWS_URL, params=params, timeout=12)
            if not response.ok:
                continue
            data = response.json()
            raw_items = data.get("articles", [])
        except Exception as e:
            logger.error(f"[GNews] Error querying for '{q}': {e}")
            continue

        for article in raw_items:
            url = article.get("url")
            published_at = article.get("publishedAt")
            title = article.get("title")

            if not url or not published_at or not title:
                continue
            if url in seen_urls:
                continue

            try:
                published_date = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                if published_date < cutoff:
                    continue
            except ValueError:
                continue

            seen_urls.add(url)
            candidate_articles.append({
                "title": title,
                "description": article.get("description", ""),
                "content": article.get("content", ""),
                "url": url,
                "published_at": published_at,
                "source": article.get("source", {}).get("name", "News Source"),
                "canonical_location": canonical_loc
            })

    logger.info(f"[GNews] Retrieved {len(candidate_articles)} unique candidates across multi-queries for '{canonical_loc}'")
    return candidate_articles[:max_results]


def is_relevant_to_location(article: dict, location: str) -> bool:
    """
    Basic sanity check for location keyword relevance.
    """
    if not location or not article:
        return True

    loc_lower = location.lower().strip()
    title = (article.get("title") or "").lower()
    description = (article.get("description") or "").lower()
    content = (article.get("content") or "").lower()

    combined_text = f"{title} {description} {content}"
    loc_parts = loc_lower.split()

    if len(loc_parts) > 1:
        return loc_lower in combined_text or all(part in combined_text for part in loc_parts)

    return loc_lower in combined_text

import logging
import re
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import List, Dict, Any, Optional, Tuple

from app.services.article_scraper import fetch_article

logger = logging.getLogger(__name__)

# Tier 1 & Tier 2 Credible Publishers
TIER_1_SOURCES = [
    "reuters", "associated press", "ap news", "bbc", "cnn", "nbc", "cbs", "abc news",
    "the hindu", "times of india", "indian express", "new york times", "washington post",
    "the guardian", "afp", "bloomberg", "wall street journal", "npr", "pbs"
]

TIER_2_SOURCES = [
    "deccan chronicle", "tribune india", "los angeles times", "chicago tribune",
    "sydney morning herald", "japan times", "manila times", "economic times",
    "marketwatch", "cnbc", "usa today", "fox news", "hindustan times", "business standard"
]


def get_current_datetime() -> datetime:
    """
    Get current system datetime dynamically in India timezone (Asia/Kolkata).
    """
    try:
        return datetime.now(ZoneInfo("Asia/Kolkata"))
    except Exception:
        return datetime.now(timezone(timedelta(hours=5, minutes=30)))


def parse_published_date(published_at: Optional[str]) -> Optional[datetime]:
    """
    Extract and parse actual publication date from ISO or formatted timestamp.
    Returns datetime or None if unverified.
    """
    if not published_at or not str(published_at).strip():
        return None
    
    clean_str = str(published_at).strip()
    try:
        dt = datetime.fromisoformat(clean_str.replace("Z", "+00:00"))
        return dt
    except ValueError:
        pass

    formats = [
        "%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(clean_str, fmt)
        except ValueError:
            pass

    return None


def format_published_date_string(dt: Optional[datetime], raw_fallback: str = "") -> Tuple[Optional[str], bool]:
    """
    Returns (formatted_date_str, date_verified).
    Preserves exact real publication date (e.g. October 6, 2026).
    """
    if dt:
        return (dt.strftime("%B %d, %Y"), True)
    if raw_fallback and not any(kw in raw_fallback.lower() for kw in ["today", "recent", "unknown"]):
        return (raw_fallback, True)
    return (None, False)


def determine_event_date_and_status(
    title: str,
    content: str,
    pub_dt: Optional[datetime],
    now_dt: datetime
) -> Dict[str, Any]:
    """
    Determines event_date and event_status:
    CURRENT, ONGOING, UPCOMING, RECENT, HISTORICAL
    """
    text_combined = f"{title} {content}".lower()
    lead_text = text_combined[:500]
    now_date = now_dt.date()
    pub_date = pub_dt.date() if pub_dt else None

    # 1. Search for explicit event date references in content text
    event_date_str = None
    event_dt_obj = None

    months_pat = r'(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)'
    match = re.search(r'\b' + months_pat + r'\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?\b', text_combined)
    if match:
        m_str, d_str, y_str = match.group(1), match.group(2), match.group(3)
        year_val = int(y_str) if y_str else now_date.year
        months_dict = {
            'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
            'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
            'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'sept': 9,
            'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
        }
        if m_str in months_dict:
            try:
                candidate_dt = datetime(year_val, months_dict[m_str], int(d_str))
                event_dt_obj = candidate_dt
                event_date_str = candidate_dt.strftime("%B %d, %Y")
            except ValueError:
                pass

    if not event_date_str:
        if pub_dt:
            event_date_str = pub_dt.strftime("%B %d, %Y")
            event_dt_obj = pub_dt
        else:
            event_date_str = now_dt.strftime("%B %d, %Y")
            event_dt_obj = now_dt

    # 2. Status Classification
    pub_diff = (now_date - pub_date).days if pub_date else 0
    event_diff = (now_date - event_dt_obj.date()).days if event_dt_obj else pub_diff

    historical_keywords = [
        "in 2015", "in 2010", "in 2018", "years ago", "decade ago", "historical back", "bought in 20"
    ]
    is_historical = (
        any(kw in lead_text for kw in historical_keywords) or
        (pub_date and pub_diff > 7) or
        (event_dt_obj and event_diff > 7)
    )

    ongoing_keywords = [
        "continues", "ongoing", "enter day", "enters day", "entering day", "still active",
        "hearing continues", "floods continue", "winter schedule", "rescue operation",
        "protest continues", "investigation under way", "underway", "remains active",
        "voter deletion", "added 84 weekly", "flights for winter"
    ]
    is_ongoing = any(kw in lead_text for kw in ongoing_keywords) or any(kw in title.lower() for kw in ongoing_keywords)

    upcoming_keywords = [
        "begin", "begins", "starting", "scheduled for", "will take place", "to be held",
        "upcoming", "winter season", "starts on", "expected to start", "from october", "from november"
    ]
    is_upcoming = any(kw in lead_text for kw in upcoming_keywords) or (event_dt_obj and event_dt_obj.date() > now_date)

    if is_historical and not is_ongoing and pub_diff > 3:
        status = "HISTORICAL"
    elif is_upcoming or (event_dt_obj and event_dt_obj.date() > now_date):
        status = "UPCOMING"
    elif is_ongoing:
        status = "ONGOING"
    elif pub_date and pub_diff == 0:
        status = "CURRENT"
    elif pub_date and pub_diff in (1, 2):
        status = "RECENT"
    else:
        status = "HISTORICAL" if pub_diff > 3 else "RECENT"

    return {
        "event_date": event_date_str,
        "event_status": status
    }


def evaluate_source_reliability(source_name: str, url: str) -> float:
    """
    Evaluate Source Quality.
    Returns reliability score from 0.0 to 1.0.
    """
    src_lower = (source_name or "").lower()
    url_lower = (url or "").lower()

    if any(t1 in src_lower or t1 in url_lower for t1 in TIER_1_SOURCES):
        return 1.0
    if any(t2 in src_lower or t2 in url_lower for t2 in TIER_2_SOURCES):
        return 0.85
    if ".gov" in url_lower or ".edu" in url_lower or "official" in src_lower:
        return 0.95

    return 0.60


def evaluate_location_relevance(title: str, content: str, location: str) -> float:
    """
    Evaluate Location Relevance.
    Returns score from 0.0 to 1.0.
    """
    if not location or not location.strip():
        return 0.50

    loc_lower = location.strip().lower()
    title_lower = (title or "").lower()
    content_lower = (content or "").lower()
    lead_content = content_lower[:350]

    # Distinguish HQ / Incidental mention phrases
    hq_incidental_patterns = [
        r"headquartered in\s+" + re.escape(loc_lower),
        r"based in\s+" + re.escape(loc_lower),
        r"formerly of\s+" + re.escape(loc_lower),
        r"native of\s+" + re.escape(loc_lower),
        r"born in\s+" + re.escape(loc_lower),
        r"owner of\s+.*" + re.escape(loc_lower),
        r"traded to\s+.*" + re.escape(loc_lower)
    ]
    for pattern in hq_incidental_patterns:
        if re.search(pattern, title_lower + " " + lead_content):
            logger.info(f"[NewsProcessor] Detected HQ/Incidental mention for location '{location}'. Setting relevance to 0.50")
            return 0.50

    # Local Event Indicators
    local_event_keywords = [
        "police", "arrest", "court", "protest", "accident", "killed", "injured",
        "fire", "traffic", "highway", "mayor", "governor", "election", "council",
        "district", "weather", "rain", "flood", "storm", "hospital", "school", "flight", "flights"
    ]

    in_title = loc_lower in title_lower
    in_lead = loc_lower in lead_content
    in_body = loc_lower in content_lower
    has_local_kw = any(kw in (title_lower + " " + lead_content) for kw in local_event_keywords)

    if in_title and (in_lead or has_local_kw):
        return 1.0
    if in_title:
        return 0.85
    if in_lead and has_local_kw:
        return 0.80
    if in_lead:
        return 0.70
    if in_body:
        return 0.40

    return 0.0


def classify_news_category(title: str, content: str) -> str:
    text = f"{title} {content}".lower()
    categories = {
        "crime": ["police", "arrested", "murder", "robbery", "stolen", "theft", "accident", "killed", "injured", "electrocuted"],
        "politics": ["election", "minister", "mla", "mp", "political", "party", "protest", "governor", "parliament", "senate"],
        "weather": ["rain", "thunderstorm", "weather", "imd", "cyclone", "flood", "heatwave", "snow"],
        "business": ["business", "company", "investment", "market", "crore", "industry", "stocks", "economy"],
        "entertainment": ["film", "movie", "cinema", "actor", "actress", "box office", "hollywood", "bollywood"],
        "government": ["government", "collector", "district administration", "scheme", "high court", "court", "nhai"]
    }
    category_scores = {cat: sum(1 for kw in kws if kw in text) for cat, kws in categories.items()}
    if not any(category_scores.values()):
        return "general"
    return max(category_scores, key=category_scores.get)


def process_news_articles(
    articles: List[Dict[str, Any]],
    location: str,
    days: int = 7,
    min_location_relevance: float = 0.60,
    min_score: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Process candidate articles:
    - Scrape original webpage content
    - Preserve exact real publication date (e.g. October 6, 2026)
    - Extract event date & classify event status (CURRENT, ONGOING, UPCOMING, RECENT, HISTORICAL)
    - Score location relevance, source reliability, and recency
    - Exclude historical & location-irrelevant stories
    - Deduplicate & Rank by composite score
    """
    if min_score is not None:
        if min_score > 1.0:
            min_location_relevance = min_score / 100.0
        else:
            min_location_relevance = min_score

    now_dt = get_current_datetime()
    cutoff_date = now_dt - timedelta(days=days)

    candidates = []
    seen_headline_keys = set()

    for article in articles:
        title = (article.get("title") or "").strip()
        url = article.get("url")
        published_at = article.get("published_at", "")

        if not title or not url:
            continue

        # Headline deduplication key
        title_key = re.sub(r"[^a-zA-Z0-9]", "", title.lower())
        if title_key in seen_headline_keys:
            continue
        seen_headline_keys.add(title_key)

        # 1. Parse Publication Date
        pub_dt = parse_published_date(published_at)
        if pub_dt and pub_dt < cutoff_date:
            logger.info(f"[NewsProcessor] Article '{title[:40]}' published prior to cutoff {cutoff_date.date()}. Discarding.")
            continue

        pub_date_display, date_verified = format_published_date_string(pub_dt, raw_fallback=published_at)

        recency_score = 0.50
        if pub_dt:
            diff_days = (now_dt.date() - pub_dt.date()).days
            if diff_days == 0:
                recency_score = 1.0
            elif diff_days == 1:
                recency_score = 0.85
            elif diff_days <= 3:
                recency_score = 0.70
            else:
                recency_score = 0.40

        # 2. Fetch Original Webpage Text
        logger.info(f"[NewsProcessor] Scraping webpage text for: '{title[:50]}...'")
        article_content = fetch_article(url)

        if not article_content or len(article_content.strip()) < 50:
            logger.warning(f"[NewsProcessor] Failed to scrape webpage text from {url}. Discarding unverified candidate.")
            continue

        # 3. Location Relevance Scoring
        loc_relevance = evaluate_location_relevance(title, article_content, location)
        if loc_relevance < min_location_relevance:
            logger.info(f"[NewsProcessor] Rejecting article '{title[:40]}' - Location relevance {loc_relevance:.2f} < {min_location_relevance}")
            continue

        # 4. Source Reliability
        source_name = article.get("source", "News Source")
        src_reliability = evaluate_source_reliability(source_name, url)

        # 5. Determine Event Date & Event Status
        event_info = determine_event_date_and_status(title, article_content, pub_dt, now_dt)
        event_date_str = event_info["event_date"]
        event_status = event_info["event_status"]

        if event_status == "HISTORICAL":
            logger.info(f"[NewsProcessor] Rejecting article '{title[:40]}' classified as HISTORICAL.")
            continue

        news_category = classify_news_category(title, article_content)

        candidates.append({
            "title": title,
            "source": source_name,
            "url": url,
            "published_at_raw": published_at,
            "published_date": pub_date_display or "Unknown",
            "date_verified": date_verified,
            "event_date": event_date_str,
            "event_status": event_status,
            "location": location,
            "category": news_category,
            "summary": article_content[:300].strip() + "...",
            "content": article_content,
            "location_relevance": loc_relevance,
            "source_reliability": src_reliability,
            "recency_score": recency_score,
            "cross_source_boost": 0.0,
            "verification_status": "verified"
        })

    # Cross-Source Verification Boost
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            words_i = set(re.findall(r'\w{4,}', candidates[i]["title"].lower()))
            words_j = set(re.findall(r'\w{4,}', candidates[j]["title"].lower()))
            overlap = words_i.intersection(words_j)
            if len(overlap) >= 3:
                candidates[i]["cross_source_boost"] = 0.15
                candidates[j]["cross_source_boost"] = 0.15
                candidates[i]["verification_status"] = "confirmed_multi_source"
                candidates[j]["verification_status"] = "confirmed_multi_source"

    # Status scoring map
    status_score_map = {
        "CURRENT": 1.0,
        "ONGOING": 0.90,
        "RECENT": 0.75,
        "UPCOMING": 0.70,
        "HISTORICAL": 0.0
    }

    # Calculate Composite Ranking Score
    for c in candidates:
        status_score = status_score_map.get(c["event_status"], 0.50)
        comp = (
            (c["location_relevance"] * 0.35) +
            (status_score * 0.30) +
            (c["source_reliability"] * 0.20) +
            (c["recency_score"] * 0.10) +
            (c["cross_source_boost"] * 0.05)
        )
        c["composite_score"] = round(comp, 2)
        c["location_score"] = int(c["location_relevance"] * 100)

    # Sort by composite score descending
    candidates.sort(key=lambda x: x["composite_score"], reverse=True)

    logger.info(f"[NewsProcessor] Successfully verified and ranked {len(candidates)} articles for '{location}'")
    return candidates


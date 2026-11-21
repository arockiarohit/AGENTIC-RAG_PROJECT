import logging
import re
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

MONTH_MAP = {
    'january': 1, 'jan': 1,
    'february': 2, 'feb': 2,
    'march': 3, 'mar': 3,
    'april': 4, 'apr': 4,
    'may': 5,
    'june': 6, 'jun': 6,
    'july': 7, 'jul': 7,
    'august': 8, 'aug': 8,
    'september': 9, 'sep': 9, 'sept': 9,
    'october': 10, 'oct': 10,
    'november': 11, 'nov': 11,
    'december': 12, 'dec': 12
}

# Known regional sub-locations for specific districts (e.g. Theni sub-districts/towns)
LOCATION_ALIASES = {
    "theni": ["theni", "cumbum", "bodinayakanur", "bodi", "meghamalai", "megamali", "periyakulam", "andipatti", "suruli", "vaigai", "kottakudi"],
    "chennai": ["chennai", "madras", "tambaram", "velachery", "egmore"],
    "dindigul": ["dindigul", "kodaikanal", "palani"],
    "madurai": ["madurai", "usilampatti", "melur"]
}

# Non-news / commercial keywords to filter out from general news queries
NON_NEWS_KEYWORDS = [
    "on-road price", "on road price", "ex-showroom", "mileage", "specifications",
    "buy online", "amazon", "flipkart", "booking", "hotel", "resort", "real estate",
    "tourist places", "places to visit", "how to reach", "distance between",
    "wikipedia", "profile", "biography", "buy now"
]


def parse_publication_date(date_str: str, text_content: str, reference_date: date) -> Optional[date]:
    """
    Parses publication date from SerpApi date metadata or article snippet text.
    Returns datetime.date or None if unknown.
    """
    raw_str = (date_str or "").strip().lower()
    content_lower = (text_content or "").lower()

    if not raw_str and content_lower:
        # Check snippet for explicit relative timestamps
        for rel_kw in ["min ago", "mins ago", "hour ago", "hours ago", "today"]:
            if rel_kw in content_lower:
                return reference_date

    if not raw_str:
        return None

    # Relative time matching
    if any(k in raw_str for k in ["min ago", "mins ago", "hour ago", "hours ago", "today", "just now"]):
        return reference_date

    if "yesterday" in raw_str or "1 day ago" in raw_str:
        return reference_date - timedelta(days=1)

    m_days = re.search(r'(\d+)\s+days?\s+ago', raw_str)
    if m_days:
        num_days = int(m_days.group(1))
        return reference_date - timedelta(days=num_days)

    m_weeks = re.search(r'(\d+)\s+weeks?\s+ago', raw_str)
    if m_weeks:
        num_weeks = int(m_weeks.group(1))
        return reference_date - timedelta(days=num_weeks * 7)

    m_months = re.search(r'(\d+)\s+months?\s+ago', raw_str)
    if m_months:
        num_months = int(m_months.group(1))
        return reference_date - timedelta(days=num_months * 30)

    # Absolute date matching: e.g. "30 Sep 2026", "Sep 30, 2026", "2026-09-30", "30/09/2026"
    # Format: YYYY-MM-DD
    m_iso = re.search(r'\b(\d{4})[/\.\-](\d{1,2})[/\.\-](\d{1,2})\b', raw_str)
    if m_iso:
        try:
            return date(int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3)))
        except ValueError:
            pass

    # Format: DD/MM/YYYY
    m_ddmmyyyy = re.search(r'\b(\d{1,2})[/\.\-](\d{1,2})[/\.\-](\d{4})\b', raw_str)
    if m_ddmmyyyy:
        try:
            d_val, m_val, y_val = int(m_ddmmyyyy.group(1)), int(m_ddmmyyyy.group(2)), int(m_ddmmyyyy.group(3))
            if d_val > 12:
                return date(y_val, m_val, d_val)
            return date(y_val, d_val, m_val)
        except ValueError:
            pass

    # Format: "30 Sep 2026" or "30th September 2026"
    m_text1 = re.search(r'\b(\d{1,2})(?:st|nd|rd|th)?\s+([a-zA-Z]+)\s+(\d{4})\b', raw_str)
    if m_text1:
        d_val = int(m_text1.group(1))
        month_name = m_text1.group(2).lower()
        y_val = int(m_text1.group(3))
        if month_name in MONTH_MAP:
            try:
                return date(y_val, MONTH_MAP[month_name], d_val)
            except ValueError:
                pass

    # Format: "September 30, 2026" or "Sep 30 2026"
    m_text2 = re.search(r'\b([a-zA-Z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b', raw_str)
    if m_text2:
        month_name = m_text2.group(1).lower()
        d_val = int(m_text2.group(2))
        y_val = int(m_text2.group(3))
        if month_name in MONTH_MAP:
            try:
                return date(y_val, MONTH_MAP[month_name], d_val)
            except ValueError:
                pass

    return None


class NewsValidator:
    """
    Validates news search results for Date, Location, News Type relevance, and Deduplication.
    """

    @staticmethod
    def is_location_relevant(item: Dict[str, Any], location: Optional[str]) -> bool:
        if not location or location.lower() in ["global", "world", "none", "india", "national"]:
            return True

        loc_clean = location.strip().lower()
        aliases = LOCATION_ALIASES.get(loc_clean, [loc_clean])

        title = item.get("title", "").lower()
        content = item.get("content", "").lower()
        url = item.get("url", "").lower()

        combined_text = f"{title} {content} {url}"

        # Check if any alias appears in title, snippet, or URL
        return any(alias in combined_text for alias in aliases)

    @staticmethod
    def is_news_article(item: Dict[str, Any], user_query: str) -> bool:
        query_lower = user_query.lower()
        title = item.get("title", "").lower()
        content = item.get("content", "").lower()
        combined = f"{title} {content}"

        # If user explicitly asked for prices/products, allow them
        user_wants_price = any(kw in query_lower for kw in ["price", "cost", "buy", "hotel", "resort", "car", "bike"])

        if not user_wants_price:
            for bad_kw in NON_NEWS_KEYWORDS:
                if bad_kw in combined:
                    return False

        return True

    @staticmethod
    def filter_and_validate(
        results: List[Dict[str, Any]],
        user_query: str,
        location: Optional[str],
        time_range: str,
        reference_date: date
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Filters and validates web search results.
        Returns: (accepted_results, audit_logs)
        """
        accepted: List[Dict[str, Any]] = []
        audit_logs: List[Dict[str, Any]] = []

        query_lower = user_query.lower()
        # Check if query explicitly asks for historical events (e.g., "2019 election", "2026 election results")
        is_explicit_historical = bool(re.search(r'\b(20\d{2}|19\d{2})\b', query_lower)) and not any(kw in query_lower for kw in ["today", "latest", "recent"])

        seen_titles = set()

        for item in results:
            raw_title = item.get("title", "")
            raw_url = item.get("url", "")
            raw_date_str = item.get("date", "")
            content = item.get("content", "")
            source = item.get("source", "News Source")

            pub_date = parse_publication_date(raw_date_str, content, reference_date)

            is_news = NewsValidator.is_news_article(item, user_query)
            loc_relevant = NewsValidator.is_location_relevant(item, location)

            # Date validation
            is_current = False
            date_reason = ""

            if is_explicit_historical:
                is_current = True
                date_reason = "Explicit historical query allowed"
            elif pub_date is None:
                # If date is unknown but search engine returned it under Google News tab, evaluate conservatively
                if time_range == "today":
                    # For "today", strictly require verification or fresh indicator
                    is_current = False
                    date_reason = "Publication date unknown (Strict 'today' requirement)"
                else:
                    is_current = True
                    date_reason = "Publication date unknown (Accepted under recent news)"
            else:
                days_diff = (reference_date - pub_date).days
                if time_range == "today":
                    if days_diff == 0:
                        is_current = True
                        date_reason = f"Published today ({pub_date.strftime('%Y-%m-%d')})"
                    else:
                        is_current = False
                        date_reason = f"Not published today (Published on {pub_date.strftime('%Y-%m-%d')}, {days_diff} days ago)"
                elif time_range == "recent":
                    if 0 <= days_diff <= 7:
                        is_current = True
                        date_reason = f"Published within recent window ({pub_date.strftime('%Y-%m-%d')}, {days_diff} days ago)"
                    else:
                        is_current = False
                        date_reason = f"Older than recent window (Published on {pub_date.strftime('%Y-%m-%d')}, {days_diff} days ago)"
                else:
                    is_current = True
                    date_reason = f"Published on {pub_date.strftime('%Y-%m-%d')}"

            # Acceptance evaluation
            is_accepted = is_news and loc_relevant and is_current
            rejection_reasons = []

            if not is_news:
                rejection_reasons.append("Non-news product/commercial content")
            if not loc_relevant:
                rejection_reasons.append(f"Not relevant to requested location '{location}'")
            if not is_current:
                rejection_reasons.append(date_reason)

            title_normalized = re.sub(r'[^a-zA-Z0-9]', '', raw_title.lower())
            if is_accepted and title_normalized in seen_titles:
                is_accepted = False
                rejection_reasons.append("Duplicate article headline")

            audit_entry = {
                "title": raw_title,
                "source": source,
                "url": raw_url,
                "published_date": pub_date.strftime("%Y-%m-%d") if pub_date else "Unknown",
                "display_date": pub_date.strftime("%B %d, %Y") if pub_date else "Unknown",
                "location": location or "Global",
                "is_current": is_current,
                "is_news": is_news,
                "location_relevant": loc_relevant,
                "accepted": is_accepted,
                "rejection_reason": "; ".join(rejection_reasons) if rejection_reasons else None
            }

            audit_logs.append(audit_entry)

            # Log detailed audit for developer debugging
            logger.info(
                f"[NewsValidator] Query: '{user_query}' | Title: '{raw_title[:40]}...' | "
                f"Published: {audit_entry['published_date']} | LocRelevant: {loc_relevant} | "
                f"IsNews: {is_news} | Accepted: {is_accepted} | Reason: {audit_entry['rejection_reason'] or 'PASSED'}"
            )

            if is_accepted:
                seen_titles.add(title_normalized)
                accepted.append({
                    "title": raw_title,
                    "url": raw_url,
                    "source": source,
                    "published_date": audit_entry["published_date"],
                    "display_date": audit_entry["display_date"],
                    "content": content
                })

        return accepted, audit_logs

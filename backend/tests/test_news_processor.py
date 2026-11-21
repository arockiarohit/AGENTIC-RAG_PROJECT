import sys
from app.services.news_service import get_news
from app.services.news_processor import process_news_articles

# Set UTF-8 encoding for Windows console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


location = "Theni"

print(f"\nSearching news for: {location}")

articles = get_news(location)

print(f"Found {len(articles)} GNews articles")


processed_articles = process_news_articles(
    articles,
    location=location,
    days=7,
    min_score=40
)


print("\n")
print("=" * 80)
print(f"VERIFIED ARTICLES: {len(processed_articles)}")
print("=" * 80)


for i, article in enumerate(processed_articles, start=1):

    print("\n")
    print("-" * 80)

    print(f"ARTICLE {i}")
    print(f"TITLE   : {article.get('title')}")
    print(f"SOURCE  : {article.get('source')}")
    print(f"DATE    : {article.get('published_at') or article.get('published_date')}")
    print(f"CATEGORY: {article.get('category') or article.get('news_category')}")
    print(f"SCORE   : {article.get('location_score')}")
    print(f"URL     : {article.get('url')}")

    print("\nCONTENT:")
    content_text = article.get("content", "")
    print(content_text[:500])
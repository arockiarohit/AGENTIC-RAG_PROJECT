from app.services.news_service import get_news


articles = get_news("Theni")

print(f"\nFound {len(articles)} recent articles\n")

for i, article in enumerate(articles, start=1):

    print("=" * 80)

    print(f"{i}. {article['title']}")
    print(f"Source     : {article['source']}")
    print(f"Published  : {article['published_at']}")
    print(f"URL        : {article['url']}")
    print(f"Description: {article['description']}")
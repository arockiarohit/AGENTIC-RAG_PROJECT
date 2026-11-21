from app.services.article_scraper import fetch_article


url = "https://www.thehindu.com/news/cities/Madurai/worker-electrocuted-to-death-in-theni/article71552723.ece"

article = fetch_article(url)

if article:
    print("\n" + "=" * 80)
    print("ORIGINAL ARTICLE CONTENT")
    print("=" * 80)
    print(article)
else:
    print("Could not extract article.")
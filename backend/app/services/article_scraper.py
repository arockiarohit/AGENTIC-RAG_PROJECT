import requests
import re
try:
    import trafilatura
except ImportError:
    trafilatura = None


def fetch_article(url: str):
    """
    Fetch and extract the actual article content
    from the original news website.
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/154.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=20
        )

        response.raise_for_status()

        if trafilatura:
            article_text = trafilatura.extract(
                response.text,
                include_comments=False,
                include_tables=False,
                include_links=False
            )
        else:
            # Fallback text extraction if trafilatura is not installed
            clean_html = re.sub(r'<script.*?</script>|<style.*?</style>', '', response.text, flags=re.DOTALL)
            text_blocks = re.findall(r'<p[^>]*>(.*?)</p>', clean_html, flags=re.DOTALL | re.IGNORECASE)
            article_text = "\n".join(re.sub(r'<[^>]+>', '', p).strip() for p in text_blocks if len(p) > 20)

        if not article_text:
            return None

        return article_text

    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None

    except Exception as e:
        print(f"Article extraction failed: {e}")
        return None
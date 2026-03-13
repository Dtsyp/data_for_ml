import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def scrape_url(url: str, css_selector: str = None) -> dict:
    """Fetch a web page and extract text content and links."""
    try:
        resp = requests.get(url, timeout=15, headers=HEADERS)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        if css_selector:
            elements = soup.select(css_selector)
            text = "\n".join(el.get_text(strip=True) for el in elements[:30])
        else:
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)

        # Truncate to avoid huge responses
        text = text[:5000]

        # Extract links
        links = []
        for a in soup.find_all("a", href=True):
            link_text = a.get_text(strip=True)
            if link_text:
                href = a["href"]
                if href.startswith("/"):
                    from urllib.parse import urljoin

                    href = urljoin(url, href)
                links.append({"text": link_text[:100], "href": href})
                if len(links) >= 30:
                    break

        return {"url": url, "text": text, "links": links}

    except Exception as e:
        return {"url": url, "error": str(e), "text": "", "links": []}

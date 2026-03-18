from collections import deque
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup

def crawl(start_url: str) -> set[str]:
    """
    Crawl all pages on the same domain as start_url.
    Returns a set of all discovered URLs.

    TODO: Implement this!

    Hints:
    1. Parse the start_url to get the domain
    2. Use a queue (deque) for BFS
    3. Use a set for visited URLs
    4. For each page, extract all <a href="..."> links
    5. Only follow links on the same domain
    """

    split = urlparse(start_url)
    domain = split.netloc

    queue = deque([start_url])
    visited = set()

    # Logic
    while queue:
        url = queue.popleft()
        print(f"parsing {url}")
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
        except Exception:
            continue
        soup = BeautifulSoup(resp.content, "html.parser")

        for tag in soup.find_all('a'):
            if not tag.has_attr('href'):
                continue

            link = tag['href']

            # Resolve relative href
            absolute_url = urljoin(url, link)

            # Parse the noramlized
            parsed = urlparse(absolute_url)

            link_domain = parsed.netloc

            if link_domain != domain or absolute_url in visited:
                continue

            queue.append(absolute_url)

        visited.add(url)

    return visited

if __name__ == '__main__':
    result = crawl('https://docs.python.org')
    print(f"Found {len(result)} pages")

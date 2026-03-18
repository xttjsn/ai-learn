"""
Level 1: Single-threaded Web Crawler (BFS)

Problem: Given a start URL, crawl all pages on the same domain.
Return all discovered URLs.

Key concepts:
- BFS traversal
- URL parsing (extract domain, resolve relative links)
- Deduplication

Try implementing this yourself first, then check the solution below.
"""

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
    pass


# --------------- SOLUTION BELOW (try first!) ---------------


def crawl_solution(start_url: str) -> set[str]:
    """Reference implementation."""
    parsed_start = urlparse(start_url)
    base_domain = parsed_start.netloc

    visited = set()
    queue = deque([start_url])

    while queue:
        url = queue.popleft()
        print(f"Parsing {url}")

        if url in visited:
            continue
        visited.add(url)

        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
        except Exception:
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        for link in soup.find_all("a", href=True):
            href = link["href"]
            # Resolve relative URLs
            absolute_url = urljoin(url, href)
            parsed = urlparse(absolute_url)

            # Same domain check
            if parsed.netloc == base_domain and absolute_url not in visited:
                queue.append(absolute_url)

    return visited


if __name__ == "__main__":
    # Test with a local server or a small site
    result = crawl_solution('https://docs.python.org')
    print(f"Found {len(result)} pages")
    pass

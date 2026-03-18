"""
Level 2: Web Crawler with URL Normalization

Building on Level 1, add proper URL normalization to avoid
crawling the same page twice under different URL forms.

Key concepts:
- Strip fragments (#section)
- Normalize trailing slashes
- Handle http vs https
- Lowercase domain
- Sort query parameters

Interview tip: This is where you show attention to detail.
Mentioning these edge cases unprompted impresses interviewers.
"""

from collections import deque
from urllib.parse import urlparse, urljoin, urlunparse, parse_qs, urlencode
import requests
from bs4 import BeautifulSoup


def normalize_url(url: str) -> str:
    """
    Normalize a URL to canonical form.

    TODO: Implement this!

    Cases to handle:
    1. Remove fragment (#...)
    2. Lowercase the scheme and host
    3. Remove default ports (80 for http, 443 for https)
    4. Remove trailing slash (except for root "/")
    5. Sort query parameters
    """
    url = url.split('#')[0]
    parsed = urlparse(url)
    scheme = str.lower(parsed.scheme)
    netloc = str.lower(parsed.netloc)
    port = ''
    if netloc.endswith(":80") and scheme == 'http':
        netloc = netloc[:-3]
    if netloc.endswith(":443") and scheme  == 'https':
        netloc = netloc[:-4]

    path = parsed.path
    if path != '/' and path.endswith('/'):
        path = path.rstrip('/')
    if not path:
        path = '/'

    query = '&'.join(sorted(parsed.query.split('&')))
    return f'{scheme}://{netloc}{path}{"?" + query if query else ""}'

    return urlunparse((scheme, netloc, path, ))


def crawl(start_url: str) -> set[str]:
    """
    Crawl with normalized URLs.

    TODO: Implement this! (Same as Level 1 but use normalize_url)
    """
    split = urlparse(start_url)
    domain = split.netloc

    queue = deque([start_url])
    visited = set()

    # Logic
    while queue:
        url = queue.popleft()
        if url in visited:
            continue
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

            # Normalize url
            normalized_url = normalize_url(absolute_url)

            # Parse the noramlized
            parsed = urlparse(normalized_url)

            link_domain = parsed.netloc

            if link_domain != domain or normalized_url in visited:
                continue

            queue.append(normalized_url)

        visited.add(url)

    return visited


# --------------- SOLUTION BELOW ---------------


def normalize_url_solution(url: str) -> str:
    parsed = urlparse(url)

    # Lowercase scheme and host
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    # Remove default ports
    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    elif netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]

    # Remove fragment
    fragment = ""

    # Normalize path - remove trailing slash (keep root)
    path = parsed.path
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    if not path:
        path = "/"

    # Sort query parameters
    query_params = parse_qs(parsed.query, keep_blank_values=True)
    sorted_query = urlencode(sorted(query_params.items()), doseq=True)

    return urlunparse((scheme, netloc, path, parsed.params, sorted_query, fragment))


def crawl_solution(start_url: str) -> set[str]:
    start_url = normalize_url_solution(start_url)
    parsed_start = urlparse(start_url)
    base_domain = parsed_start.netloc

    visited = set()
    queue = deque([start_url])

    while queue:
        url = queue.popleft()
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
            absolute_url = urljoin(url, link["href"])
            normalized = normalize_url_solution(absolute_url)
            parsed = urlparse(normalized)

            if parsed.netloc == base_domain and normalized not in visited:
                queue.append(normalized)

    return visited

if __name__ == '__main__':
    result = crawl('https://docs.python.org')
    print(f'Crawled {len(result)} pages')

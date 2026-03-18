"""
Level 3: Async Concurrent Web Crawler

This is the key follow-up in the Anthropic interview.
The interviewer wants to see you handle concurrency properly.

Key concepts:
- asyncio event loop
- aiohttp for async HTTP
- Bounded concurrency (semaphore)
- Thread-safe data structures (not needed with asyncio single-thread)
- Producer-consumer pattern

Interview approach:
1. Explain WHY async is better here (I/O bound, not CPU bound)
2. Discuss the difference between threading and asyncio
3. Show you understand the GIL implications
"""

import asyncio
from collections import deque
from urllib.parse import urlparse, urljoin, urlencode, parse_qs, urlunparse
from typing import Set
from bs4 import BeautifulSoup

# In interview, you might not have aiohttp available.
# Be ready to discuss it conceptually or use concurrent.futures as alternative.
try:
    import aiohttp
except ImportError:
    aiohttp = None

# normalize function
def normalize_url(url: str) -> str:
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


async def crawl_async(start_url: str, max_concurrent: int = 10) -> set[str]:
    """
    Async web crawler with bounded concurrency.

    TODO: Implement this!

    Hints:
    1. Use asyncio.Semaphore to limit concurrent requests
    2. Use asyncio.Queue or create tasks dynamically
    3. Keep a visited set (safe in asyncio - single thread)
    4. Use aiohttp.ClientSession for async HTTP
    """

    parsed_start = urlparse(start_url)
    domain = parsed_start.netloc

    visited: Set[str] = set()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_and_parse(session: "aiohttp.ClientSession", url: str):
        async with semaphore:
            try:
                async with session.get(ulr, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status != 200:
                        return []
                    html = await resp.text()
            except Exception:
                return []

        bs = BeautifulSoup(html, "html.parser")

        links = []
        for link in bs.find_all('a', href=True):
            absolute = urljoin(url, link['href'])
            parsed = urlparse(absolute)
            # TODO normalize
            clean = normalize_url(absolute)
            if parsed.netloc == domain and clean not in visited:
                links.append(clean)

        return links

    async with aiohttp.ClientSession() as session:
        visited.add(start_url)
        pending = {asyncio.create_task(fetch_and_parse(start_url))}

        while pending:
            done, pending = asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETE)
            for task in done:
                new_urls = task.result()
                for url in new_urls:
                    if url not in visited:
                        visited.add(url)
                        pending.add(asyncio.create_task(fetch_and_parse(url)))

    return visited


# --------------- SOLUTION BELOW ---------------


async def crawl_async_solution(start_url: str, max_concurrent: int = 10) -> set[str]:
    """
    Production-quality async crawler.

    Architecture:
    - Semaphore limits concurrent HTTP requests
    - Tasks are created dynamically as new URLs are discovered
    - visited set prevents duplicate work
    """
    parsed_start = urlparse(start_url)
    base_domain = parsed_start.netloc

    visited: Set[str] = set()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_and_parse(session: "aiohttp.ClientSession", url: str) -> list[str]:
        """Fetch a URL and return discovered same-domain links."""
        async with semaphore:
            try:
                print(f'Crawling {url}')
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status != 200:
                        return []
                    html = await resp.text()
            except Exception:
                return []

        # Parse links (CPU work, but fast enough to not block)
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        links = []
        for link in soup.find_all("a", href=True):
            absolute = urljoin(url, link["href"])
            parsed = urlparse(absolute)
            if parsed.netloc == base_domain:
                # Strip fragment
                clean = absolute.split("#")[0]
                if clean not in visited:
                    links.append(clean)
        return links

    async with aiohttp.ClientSession() as session:
        # Use a task set to track in-flight work
        visited.add(start_url)
        pending = {asyncio.create_task(fetch_and_parse(session, start_url))}

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                new_urls = task.result()
                for url in new_urls:
                    if url not in visited:
                        visited.add(url)
                        pending.add(asyncio.create_task(fetch_and_parse(session, url)))

    return visited


# Alternative: Thread pool version (useful if asyncio isn't available)
def crawl_threaded(start_url: str, max_workers: int = 10) -> set[str]:
    """
    Thread pool crawler using concurrent.futures.

    This is a good fallback if the interviewer doesn't want asyncio,
    or if you need to discuss threading vs async.

    Key points to discuss:
    - GIL: threads work fine for I/O bound (HTTP requests)
    - Need thread-safe visited set (use threading.Lock)
    - concurrent.futures provides clean API
    """
    import concurrent.futures
    import threading
    import requests
    from bs4 import BeautifulSoup

    parsed_start = urlparse(start_url)
    base_domain = parsed_start.netloc

    visited = set()
    visited_lock = threading.Lock()
    queue = deque([start_url])
    queue_lock = threading.Lock()

    def process_url(url: str) -> list[str]:
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
        except Exception:
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        links = []
        for link in soup.find_all("a", href=True):
            absolute = urljoin(url, link["href"])
            parsed = urlparse(absolute)
            if parsed.netloc == base_domain:
                clean = absolute.split("#")[0]
                links.append(clean)
        return links

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        visited.add(start_url)
        futures = {executor.submit(process_url, start_url)}

        while futures:
            done, futures = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                new_urls = future.result()
                for url in new_urls:
                    with visited_lock:
                        if url not in visited:
                            visited.add(url)
                            futures.add(executor.submit(process_url, url))

    return visited


if __name__ == "__main__":
    result = asyncio.run(crawl_async_solution("https://docs.python.org", 1000))
    print(f'Total crawled {len(result)}')

import asyncio
from urllib.parse import urljoin, urlparse, urlencode, urlunparse, parse_qs
import aiohttp
from bs4 import BeautifulSoup

def normalize_url(url):
    # Lower case
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    # Port normalization when http + 80 and https + 443
    if scheme == 'http' and parsed.port == 80:
        netloc = netloc[:-3]
    elif scheme == 'https' and parsed.port == 443:
        netloc = netloc[:-4]

    # Get rid of fragments + sections
    fragments = ''

    # Ending slahes
    path = parsed.path
    if path == '':
        path == '/'
    elif path != '/' and path.endswith('/'):
        path = path.rstrip('/')

    # Sort query
    param = parse_qs(parsed.query)
    query = urlencode(sorted(param.items()))

    # reconstruct url
    return urlunparse((scheme, netloc, path, parsed.params, query, fragments))

async def crawl_async(start_url, max_concurrency):
    parsed_start = urlparse(start_url)
    domain = parsed_start.netloc

    visited = set()
    semaphore = asyncio.Semaphore(max_concurrency)

    async def fetch_and_parse(session: "aiohttp.ClientSession", url):
        async with semaphore:
            try:
                print(f'Crawling {url}')
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status != 200:
                        return []
                    html = await resp.text()
            except Exception:
                return []


        # Parse it, CPU bound but minimal
        soup = BeautifulSoup(html, "html.parser")
        links = []

        for link in soup.find_all("a", href=True):
            joined = urljoin(url, link['href'])
            clean = normalize_url(joined)
            parsed_clean = urlparse(clean)
            if parsed_clean.netloc == domain and clean not in visited:
                links.append(clean)

        return links

    async with aiohttp.ClientSession() as session:
        pending = {asyncio.create_task(fetch_and_parse(session, start_url))}
        visited.add(start_url)

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                for link in task.result():
                    if link not in visited:
                        visited.add(link)
                        pending.add(asyncio.create_task(fetch_and_parse(session, link)))

    return visited


if __name__ == '__main__':
    result = asyncio.run(crawl_async("https://www.anthropic.com", max_concurrency=1000))
    print(f'Crawled {len(result)} pages')

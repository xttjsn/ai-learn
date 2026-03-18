# Coding Q1: Web Crawler

## Problem
Given a starting URL, crawl all pages on the same domain. Return all discovered URLs.

## Progressive Levels

### Level 1: Single-threaded BFS
- Basic BFS/DFS traversal
- URL parsing and same-domain filtering
- Deduplication with a set

### Level 2: URL Normalization
- Handle http/https, trailing slashes, fragments (#), query params
- Canonical URL form

### Level 3: Async/Concurrent Version
- asyncio + aiohttp
- Thread pool with concurrent.futures
- Producer-consumer pattern

### Level 4: Production Concerns
- Rate limiting / politeness
- Retry with backoff
- robots.txt compliance
- Memory-bounded visited set (bloom filter)
- Graceful shutdown

## Files
- `level1_basic.py` — Single-threaded BFS
- `level2_normalized.py` — With URL normalization
- `level3_async.py` — Async concurrent version
- `level4_threaded.py` — Thread pool version
- `test_crawler.py` — Tests

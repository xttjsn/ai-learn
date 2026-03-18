# Coding Q2: LRU Cache (functools.lru_cache style)

## Problem
Implement a decorator that works like Python's `@functools.lru_cache`.
Then extend it with crash recovery (persistence to disk).

## Progressive Levels

### Level 1: Basic LRU Cache Decorator
- OrderedDict-based LRU
- Decorator that caches function results
- Key generation from args/kwargs

### Level 2: Cache Key Generation
- Handle mutable vs immutable args
- Proper key hashing
- Edge cases (unhashable types)

### Level 3: Crash Recovery with WAL
- Write-Ahead Log for persistence
- Restore cache state after crash
- When to write to disk (every set? batched?)

### Level 4: Production Concerns
- Thread safety
- Cache size limits / eviction
- CPU bound vs IO bound discussion
- TTL (time-to-live)
- Cache stampede prevention

## Files
- `level1_basic.py` — Basic LRU cache decorator
- `level2_keys.py` — Proper key generation
- `level3_persistent.py` — With WAL crash recovery
- `test_cache.py` — Tests

"""
Level 1: Basic LRU Cache Decorator

Implement a decorator that works like @functools.lru_cache.

Key concepts:
- OrderedDict for O(1) access + ordering
- Decorator pattern
- Cache key from function arguments

Usage should look like:
    @lru_cache(maxsize=128)
    def expensive_function(x, y):
        ...
"""

from collections import OrderedDict
from functools import wraps
from typing import Any, Callable


def lru_cache(maxsize: int = 128):
    """
    LRU cache decorator.

    TODO: Implement this!

    Hints:
    1. Use OrderedDict — move_to_end() on access, popitem(last=False) on eviction
    2. Create a cache key from args and kwargs
    3. Return a decorator that wraps the function
    """

    def decorator(func: Callable) -> Callable:
        cache = OrderedDict()
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = _make_key(args, kwargs)
            if key in cache:
                cache.move_to_end(key)
                return cache[key]

            # Else, compute
            result = func(*args, **kwargs)

            # set item
            cache[key] = result

            if len(cache) > maxsize:
                cache.popitem(last=False) # pop from front

            return result

        wrapper.cache = cache
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_info = lambda: {
            "size": len(cache),
            "maxsize": maxsize
        }
        return wrapper

    return decorator


# --------------- SOLUTION BELOW ---------------


def lru_cache_solution(maxsize: int = 128):
    """Reference implementation."""

    def decorator(func: Callable) -> Callable:
        cache: OrderedDict[Any, Any] = OrderedDict()

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = _make_key(args, kwargs)

            if key in cache:
                # Move to end (most recently used)
                cache.move_to_end(key)
                return cache[key]

            # Compute result
            result = func(*args, **kwargs)

            # Store in cache
            cache[key] = result

            # Evict if over capacity
            if len(cache) > maxsize:
                cache.popitem(last=False)  # Remove least recently used

            return result

        # Expose cache for inspection
        wrapper.cache = cache
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_info = lambda: {
            "size": len(cache),
            "maxsize": maxsize,
        }

        return wrapper

    return decorator


def _make_key(args: tuple, kwargs: dict) -> tuple:
    """
    Create a hashable cache key from args and kwargs.

    This is a simplified version. The real functools uses _HashedSeq
    for better performance.
    """
    key = args
    if kwargs:
        # Sort kwargs for consistent key regardless of call order
        key += tuple(sorted(kwargs.items()))
    return key


# --------------- TESTING ---------------

if __name__ == "__main__":
    call_count = 0

    @lru_cache_solution(maxsize=3)
    def add(a, b):
        global call_count
        call_count += 1
        return a + b

    # First calls — should compute
    assert add(1, 2) == 3
    assert add(3, 4) == 7
    assert add(5, 6) == 11
    assert call_count == 3

    # Cached call — should NOT compute
    assert add(1, 2) == 3
    assert call_count == 3  # Still 3

    # This should evict (1,2) since maxsize=3
    assert add(7, 8) == 15
    assert call_count == 4

    # (1,2) was evicted, should recompute
    assert add(1, 2) == 3
    assert call_count == 5

    print("All tests passed! ✅")

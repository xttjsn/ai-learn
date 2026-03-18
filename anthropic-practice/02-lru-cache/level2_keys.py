"""
Level 2: Proper Cache Key Generation

The tricky part of LRU cache in interviews: how do you generate keys?

Key concepts:
- Handling different argument types
- kwargs ordering
- Unhashable arguments (lists, dicts)
- Type-sensitive keys: f(1) vs f(1.0)

This is often a follow-up question in the Anthropic interview.
"""

from collections import OrderedDict
from functools import wraps
from typing import Any, Callable


class _HashedSeq(list):
    """
    Like functools._HashedSeq — a list subclass that's hashable.
    Pre-computes hash for performance.
    """
    __slots__ = ("hashvalue",)

    def __init__(self, tup):
        self[:] = tup
        self.hashvalue = hash(tup)

    def __hash__(self):
        return self.hashvalue


def make_key(args: tuple, kwargs: dict, typed: bool = False) -> _HashedSeq:
    """
    Build a cache key from function arguments.

    TODO: Implement this!

    Considerations:
    1. args are positional — use directly
    2. kwargs need consistent ordering
    3. If typed=True, f(1) and f(1.0) should be different keys
    4. Need a separator between args and kwargs

    Hint: Look at how CPython's functools._make_key works.
    """
    key = args
    if kwargs:
        key += tuple(sorted(kwargs.items()))

    if typed:
        key += tuple(type(v) for v in args)
        if kwargs:
            key += tuple(type(v) for _, v in sorted(kwargs.items()))

    return _HashedSeq(key)


# --------------- SOLUTION BELOW ---------------


# Sentinel object to separate args from kwargs in the key
_KWARGS_SENTINEL = object()


def make_key_solution(
    args: tuple,
    kwargs: dict,
    typed: bool = False,
) -> _HashedSeq:
    """
    Reference implementation (mirrors CPython's functools._make_key).
    """
    key = args

    if kwargs:
        # Add sentinel to separate args from kwargs
        key += (_KWARGS_SENTINEL,)
        for item in sorted(kwargs.items()):
            key += item

    if typed:
        # Add types to distinguish f(1) from f(1.0)
        key += tuple(type(v) for v in args)
        if kwargs:
            key += tuple(type(v) for v in kwargs.values())

    # If there's only one arg and no kwargs, and it's hashable,
    # just return it directly (optimization)
    if len(key) == 1 and isinstance(key[0], (int, str, float, bool, type(None))):
        return key[0]

    return _HashedSeq(key)


def lru_cache(maxsize: int = 128, typed: bool = False):
    """LRU cache with proper key generation."""

    def decorator(func: Callable) -> Callable:
        cache: OrderedDict = OrderedDict()

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = make_key_solution(args, kwargs, typed=typed)

            if key in cache:
                cache.move_to_end(key)
                return cache[key]

            result = func(*args, **kwargs)
            cache[key] = result

            if len(cache) > maxsize:
                cache.popitem(last=False)

            return result

        wrapper.cache = cache
        wrapper.cache_clear = lambda: cache.clear()
        return wrapper

    return decorator


# --------------- TESTING ---------------

if __name__ == "__main__":
    # Test: kwargs ordering shouldn't matter
    @lru_cache(maxsize=10)
    def add(a, b=0):
        return a + b

    r1 = add(1, b=2)
    r2 = add(1, b=2)  # Should be cached
    assert r1 == r2 == 3

    # Test: typed=True
    @lru_cache(maxsize=10, typed=True)
    def identity(x):
        return x

    identity(1)
    identity(1.0)  # Different key when typed=True
    assert len(identity.cache) == 2  # Two entries

    # Test: typed=False (default)
    @lru_cache(maxsize=10, typed=False)
    def identity2(x):
        return x

    identity2(1)
    identity2(1.0)  # Same key when typed=False (both hash the same)
    # Note: 1 == 1.0 and hash(1) == hash(1.0) in Python
    assert len(identity2.cache) == 1

    print("All tests passed! ✅")

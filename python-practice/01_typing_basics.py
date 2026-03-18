"""
01 - Modern Python Type Hints
==============================
Python 3.10+ typing features. Run with: python3 01_typing_basics.py
"""

# ─── Basics ───────────────────────────────────────────────────────────
# Python 3.10+ uses built-in types directly (no need for typing.List, typing.Dict)

def greet(name: str, times: int = 1) -> str:
    return (f"Hello, {name}! " * times).strip()


# Union types with | (Python 3.10+) — replaces typing.Union
def normalize_id(id: int | str) -> str:
    return str(id).lower()


# Optional is just X | None
def find_user(user_id: int) -> dict | None:
    users = {1: {"name": "xtt", "role": "admin"}, 2: {"name": "dos", "role": "bot"}}
    return users.get(user_id)


# ─── Collections ──────────────────────────────────────────────────────

def merge_configs(base: dict[str, int], override: dict[str, int]) -> dict[str, int]:
    return {**base, **override}


def unique_words(sentences: list[str]) -> set[str]:
    return {word for s in sentences for word in s.split()}


# ─── Callable types ──────────────────────────────────────────────────
from collections.abc import Callable

def apply_transform(data: list[int], fn: Callable[[int], int]) -> list[int]:
    """Apply fn to each element."""
    return [fn(x) for x in data]


# ─── TypeAlias (Python 3.12 uses `type` statement, 3.10 uses TypeAlias) ──
from typing import TypeAlias

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

def count_keys(data: JSON) -> int:
    """Count total keys in a nested JSON-like structure."""
    if isinstance(data, dict):
        return len(data) + sum(count_keys(v) for v in data.values())
    elif isinstance(data, list):
        return sum(count_keys(item) for item in data)
    return 0


# ─── Generics (Python 3.12 style) ────────────────────────────────────
from typing import TypeVar

T = TypeVar("T")

def first_or_default(items: list[T], default: T) -> T:
    """Return first item or default if empty."""
    return items[0] if items else default


# ─── Literal types ───────────────────────────────────────────────────
from typing import Literal

def set_log_level(level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]) -> str:
    return f"Log level set to {level}"


# ─── TypedDict ───────────────────────────────────────────────────────
from typing import TypedDict, NotRequired

class ServerConfig(TypedDict):
    host: str
    port: int
    debug: NotRequired[bool]  # Optional key


def start_server(config: ServerConfig) -> str:
    debug = config.get("debug", False)
    return f"Starting {'debug ' if debug else ''}server on {config['host']}:{config['port']}"


# ═══════════════════════════════════════════════════════════════════════
# EXERCISES — Fill in the TODOs and run this file
# ═══════════════════════════════════════════════════════════════════════

# TODO 1: Write a function `safe_divide` that takes two floats and returns
#          float | None (returns None if dividing by zero)
def safe_divide(a: float, b: float) -> float | None:
    if b == 0:
        return None
    return a / b


# TODO 2: Write a function `flatten` that takes list[list[T]] and returns list[T]
def flatten(nested: list[list[T]]) -> list[T]:
    flattened = 


# TODO 3: Create a TypedDict called `Player` with:
#          name: str, score: int, active: NotRequired[bool]
#          Then write `format_player(p: Player) -> str` returning "name: score"

# your code here


# ═══════════════════════════════════════════════════════════════════════
# TESTS — Run to verify
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Provided examples
    assert greet("xtt") == "Hello, xtt!"
    assert greet("xtt", 2) == "Hello, xtt! Hello, xtt!"
    assert normalize_id("ABC") == "abc"
    assert normalize_id(42) == "42"
    assert find_user(1) == {"name": "xtt", "role": "admin"}
    assert find_user(999) is None
    assert merge_configs({"a": 1}, {"a": 2, "b": 3}) == {"a": 2, "b": 3}
    assert unique_words(["hello world", "world peace"]) == {"hello", "world", "peace"}
    assert apply_transform([1, 2, 3], lambda x: x * 2) == [2, 4, 6]
    assert count_keys({"a": 1, "b": {"c": 2}}) == 3
    assert first_or_default([], "fallback") == "fallback"
    assert first_or_default([1, 2], 0) == 1
    assert set_log_level("DEBUG") == "Log level set to DEBUG"
    assert start_server({"host": "localhost", "port": 8080}) == "Starting server on localhost:8080"
    print("✅ All provided tests passed!")

    # Exercise tests
    if safe_divide(10, 3) is not None:
        assert abs(safe_divide(10.0, 3.0) - 3.333333) < 0.001
        assert safe_divide(1.0, 0.0) is None
        print("✅ Exercise 1 passed!")
    else:
        print("⬜ Exercise 1: implement safe_divide")

    try:
        assert flatten([[1, 2], [3], [4, 5]]) == [1, 2, 3, 4, 5]
        assert flatten([[], [1]]) == [1]
        print("✅ Exercise 2 passed!")
    except (TypeError, AssertionError):
        print("⬜ Exercise 2: implement flatten")

    try:
        p: Player = {"name": "xtt", "score": 100}  # type: ignore
        assert format_player(p) == "xtt: 100"  # type: ignore
        print("✅ Exercise 3 passed!")
    except NameError:
        print("⬜ Exercise 3: implement Player and format_player")

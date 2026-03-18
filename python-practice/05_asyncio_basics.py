"""
05 - Asyncio Basics
===================
Concurrent I/O with async/await. Run with: python3 05_asyncio_basics.py
"""
import asyncio
import time


# ─── Why async? ──────────────────────────────────────────────────────
# Threads have GIL overhead. Async uses cooperative multitasking:
# one thread, many coroutines, switching at `await` points.
# Perfect for I/O-bound work (HTTP, DB, file I/O, network).
# NOT useful for CPU-bound work (use multiprocessing for that).


# ─── Basic coroutine ─────────────────────────────────────────────────
async def fetch_model(name: str, delay: float = 1.0) -> dict:
    """Simulate fetching model info from an API."""
    print(f"  ⏳ Fetching {name}...")
    await asyncio.sleep(delay)  # Non-blocking sleep (simulates I/O)
    print(f"  ✅ Got {name}")
    return {"name": name, "params": len(name) * 1_000_000_000}


# ─── Sequential vs Concurrent ───────────────────────────────────────
async def sequential():
    """Fetches one by one — slow!"""
    start = time.monotonic()
    r1 = await fetch_model("llama", 1.0)
    r2 = await fetch_model("qwen", 1.0)
    r3 = await fetch_model("gemma", 1.0)
    elapsed = time.monotonic() - start
    print(f"  Sequential: {elapsed:.1f}s (expected ~3s)")
    return [r1, r2, r3]


async def concurrent():
    """Fetches all at once with gather — fast!"""
    start = time.monotonic()
    r1, r2, r3 = await asyncio.gather(
        fetch_model("llama", 1.0),
        fetch_model("qwen", 1.0),
        fetch_model("gemma", 1.0),
    )
    elapsed = time.monotonic() - start
    print(f"  Concurrent: {elapsed:.1f}s (expected ~1s)")
    return [r1, r2, r3]


# ─── create_task — fire and forget (but track it) ───────────────────
async def background_tasks():
    """Tasks start immediately when created."""
    task1 = asyncio.create_task(fetch_model("mistral", 0.5))
    task2 = asyncio.create_task(fetch_model("phi", 0.3))

    # Do other work while tasks run...
    print("  Doing other work...")
    await asyncio.sleep(0.1)

    # Now collect results
    result1 = await task1
    result2 = await task2
    return [result1, result2]


# ─── Timeouts ────────────────────────────────────────────────────────
async def with_timeout():
    """Cancel if too slow."""
    try:
        result = await asyncio.wait_for(fetch_model("huge-model", 5.0), timeout=1.0)
        return result
    except asyncio.TimeoutError:
        print("  ⏰ Timed out!")
        return None


# ─── Async iteration ────────────────────────────────────────────────
async def generate_tokens(prompt: str, count: int = 5):
    """Async generator — like streaming from an LLM."""
    for i in range(count):
        await asyncio.sleep(0.2)  # Simulate generation time
        yield f"token_{i}"


async def stream_response():
    """Consume an async generator."""
    tokens = []
    async for token in generate_tokens("Hello world"):
        tokens.append(token)
        print(f"  Got: {token}")
    return tokens


# ─── Async context manager ──────────────────────────────────────────
class AsyncConnection:
    """Simulates an async database/HTTP connection."""

    def __init__(self, url: str):
        self.url = url
        self.connected = False

    async def __aenter__(self) -> "AsyncConnection":
        print(f"  Connecting to {self.url}...")
        await asyncio.sleep(0.1)
        self.connected = True
        return self

    async def __aexit__(self, *args):
        print(f"  Disconnecting from {self.url}")
        await asyncio.sleep(0.05)
        self.connected = False

    async def query(self, q: str) -> str:
        if not self.connected:
            raise RuntimeError("Not connected")
        await asyncio.sleep(0.1)
        return f"Result for: {q}"


async def use_connection():
    async with AsyncConnection("http://localhost:8080") as conn:
        result = await conn.query("SELECT * FROM models")
        print(f"  {result}")
        return result


# ─── Exception handling in gather ────────────────────────────────────
async def failing_task():
    await asyncio.sleep(0.1)
    raise ValueError("Something broke!")

async def handle_errors():
    """return_exceptions=True collects errors instead of raising."""
    results = await asyncio.gather(
        fetch_model("good", 0.1),
        failing_task(),
        fetch_model("also-good", 0.1),
        return_exceptions=True,
    )
    for r in results:
        if isinstance(r, Exception):
            print(f"  ❌ Error: {r}")
        else:
            print(f"  ✅ {r['name']}")
    return results


# ═══════════════════════════════════════════════════════════════════════
# EXERCISES
# ═══════════════════════════════════════════════════════════════════════

# TODO 1: Write an async function `fetch_all(names: list[str]) -> list[dict]`
#          that fetches all models concurrently using gather
#          Each fetch should use delay=0.5


# TODO 2: Write an async generator `countdown(n: int)` that yields n, n-1, ..., 1
#          with a 0.1s delay between each


# TODO 3: Write an async context manager class `Timer` that:
#          - Records start time in __aenter__, returns self
#          - Records elapsed time in __aexit__
#          - Has a property `elapsed` returning the duration in seconds


# ═══════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════

async def run_tests():
    print("\n── Sequential ──")
    await sequential()

    print("\n── Concurrent ──")
    await concurrent()

    print("\n── Background Tasks ──")
    results = await background_tasks()
    assert len(results) == 2

    print("\n── Timeout ──")
    r = await with_timeout()
    assert r is None

    print("\n── Streaming ──")
    tokens = await stream_response()
    assert len(tokens) == 5

    print("\n── Async Context Manager ──")
    r = await use_connection()
    assert "Result for:" in r

    print("\n── Error Handling ──")
    results = await handle_errors()
    assert isinstance(results[1], ValueError)

    print("\n✅ All provided tests passed!")

    # Exercise tests
    print("\n── Exercises ──")
    try:
        start = time.monotonic()
        results = await fetch_all(["a", "b", "c"])  # type: ignore
        elapsed = time.monotonic() - start
        assert len(results) == 3
        assert elapsed < 1.0, f"Should be concurrent, took {elapsed:.1f}s"
        print("✅ Exercise 1 passed!")
    except NameError:
        print("⬜ Exercise 1: implement fetch_all")

    try:
        nums = []
        async for n in countdown(3):  # type: ignore
            nums.append(n)
        assert nums == [3, 2, 1]
        print("✅ Exercise 2 passed!")
    except NameError:
        print("⬜ Exercise 2: implement countdown")

    try:
        async with Timer() as t:  # type: ignore
            await asyncio.sleep(0.2)
        assert 0.15 < t.elapsed < 0.4  # type: ignore
        print("✅ Exercise 3 passed!")
    except NameError:
        print("⬜ Exercise 3: implement Timer")


if __name__ == "__main__":
    asyncio.run(run_tests())

"""
06 - Asyncio Patterns
=====================
Real-world async patterns. Run with: python3 06_asyncio_patterns.py
"""
import asyncio
import random
import time
from dataclasses import dataclass, field


# ─── Semaphore — limit concurrency ──────────────────────────────────
# Problem: You have 100 URLs but only want 5 concurrent requests

async def fetch_url(url: str, delay: float = 0.5) -> str:
    """Simulate HTTP fetch."""
    await asyncio.sleep(delay)
    return f"<html>{url}</html>"


async def fetch_with_limit(urls: list[str], max_concurrent: int = 5) -> list[str]:
    """Fetch URLs with bounded concurrency."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_fetch(url: str) -> str:
        async with semaphore:
            return await fetch_url(url, delay=random.uniform(0.1, 0.3))

    return await asyncio.gather(*[bounded_fetch(url) for url in urls])


# ─── Producer/Consumer with asyncio.Queue ────────────────────────────
@dataclass
class InferenceRequest:
    prompt: str
    request_id: int


@dataclass
class InferenceResult:
    request_id: int
    text: str


async def producer(queue: asyncio.Queue[InferenceRequest], count: int):
    """Produce inference requests."""
    for i in range(count):
        req = InferenceRequest(prompt=f"Question {i}", request_id=i)
        await queue.put(req)
        print(f"  📤 Produced request {i}")
        await asyncio.sleep(0.05)
    # Signal done
    await queue.put(None)  # type: ignore


async def consumer(
    queue: asyncio.Queue[InferenceRequest],
    results: list[InferenceResult],
    worker_id: int,
):
    """Consume and process requests."""
    while True:
        req = await queue.get()
        if req is None:
            queue.task_done()
            await queue.put(None)  # Re-signal for other consumers
            break
        # Simulate inference
        await asyncio.sleep(random.uniform(0.1, 0.3))
        result = InferenceResult(request_id=req.request_id, text=f"Answer to {req.prompt}")
        results.append(result)
        print(f"  📥 Worker {worker_id} processed request {req.request_id}")
        queue.task_done()


async def producer_consumer_demo():
    queue: asyncio.Queue[InferenceRequest] = asyncio.Queue(maxsize=5)
    results: list[InferenceResult] = []

    # 1 producer, 3 consumers
    await asyncio.gather(
        producer(queue, 10),
        consumer(queue, results, 0),
        consumer(queue, results, 1),
        consumer(queue, results, 2),
    )
    return results


# ─── Retry with exponential backoff ─────────────────────────────────
async def unreliable_api(fail_count: int = 2) -> str:
    """Fails `fail_count` times then succeeds."""
    unreliable_api._calls = getattr(unreliable_api, "_calls", 0) + 1
    if unreliable_api._calls <= fail_count:
        raise ConnectionError(f"Failed (attempt {unreliable_api._calls})")
    return "success"


async def retry(
    coro_factory,  # Callable that returns a new coroutine each time
    max_retries: int = 3,
    base_delay: float = 0.1,
    backoff_factor: float = 2.0,
) -> object:
    """Retry with exponential backoff."""
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                delay = base_delay * (backoff_factor ** attempt)
                print(f"  ⚠️ Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
    raise last_error  # type: ignore


# ─── Event — signal between coroutines ──────────────────────────────
async def event_demo():
    """One coroutine signals another."""
    model_loaded = asyncio.Event()

    async def load_model():
        print("  Loading model...")
        await asyncio.sleep(0.5)
        print("  Model loaded!")
        model_loaded.set()

    async def wait_for_model():
        print("  Waiting for model...")
        await model_loaded.wait()
        print("  Model is ready, starting inference!")
        return "inference_done"

    results = await asyncio.gather(load_model(), wait_for_model())
    return results[1]


# ─── TaskGroup (Python 3.11+) — structured concurrency ──────────────
async def task_group_demo():
    """TaskGroup: if one fails, all get cancelled."""
    results = []

    async def good_task(name: str, delay: float):
        await asyncio.sleep(delay)
        results.append(name)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(good_task("alpha", 0.1))
        tg.create_task(good_task("beta", 0.2))
        tg.create_task(good_task("gamma", 0.15))

    return sorted(results)


# ─── as_completed — process results as they arrive ──────────────────
async def as_completed_demo():
    """Process results in completion order, not submission order."""
    async def timed_task(name: str, delay: float) -> str:
        await asyncio.sleep(delay)
        return name

    tasks = [
        timed_task("slow", 0.3),
        timed_task("fast", 0.1),
        timed_task("medium", 0.2),
    ]

    order = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        order.append(result)
        print(f"  Completed: {result}")

    return order  # Should be: fast, medium, slow


# ═══════════════════════════════════════════════════════════════════════
# EXERCISES
# ═══════════════════════════════════════════════════════════════════════

# TODO 1: Write `batch_process(items: list[str], batch_size: int) -> list[str]`
#          Process items in batches of `batch_size` concurrently
#          Each item takes 0.1s to process (simulate with asyncio.sleep)
#          e.g., 10 items with batch_size=3 → 4 batches


# TODO 2: Write a rate limiter using asyncio:
#          `class RateLimiter` with `async def acquire(self)` that ensures
#          at most N calls per second. Use it to wrap fetch_url.


# TODO 3: Write `async def first_success(coros: list) -> object`
#          that returns the first coroutine to complete successfully,
#          cancelling the rest. If all fail, raise the last error.
#          Hint: use asyncio.wait with FIRST_COMPLETED


# ═══════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════

async def run_tests():
    print("\n── Semaphore (bounded concurrency) ──")
    urls = [f"http://example.com/{i}" for i in range(10)]
    start = time.monotonic()
    results = await fetch_with_limit(urls, max_concurrent=5)
    elapsed = time.monotonic() - start
    assert len(results) == 10
    print(f"  Fetched 10 URLs in {elapsed:.1f}s (max 5 concurrent)")

    print("\n── Producer/Consumer ──")
    results = await producer_consumer_demo()
    assert len(results) == 10
    assert all(isinstance(r, InferenceResult) for r in results)

    print("\n── Retry with backoff ──")
    unreliable_api._calls = 0  # Reset
    result = await retry(lambda: unreliable_api(2), max_retries=3)
    assert result == "success"

    print("\n── Event signaling ──")
    r = await event_demo()
    assert r == "inference_done"

    print("\n── TaskGroup ──")
    r = await task_group_demo()
    assert r == ["alpha", "beta", "gamma"]

    print("\n── as_completed ──")
    order = await as_completed_demo()
    assert order == ["fast", "medium", "slow"]

    print("\n✅ All provided tests passed!")

    # Exercise tests
    print("\n── Exercises ──")
    try:
        start = time.monotonic()
        r = await batch_process([f"item-{i}" for i in range(10)], batch_size=3)  # type: ignore
        elapsed = time.monotonic() - start
        assert len(r) == 10
        assert elapsed < 0.5, f"Should be batched, took {elapsed:.1f}s"
        print("✅ Exercise 1 passed!")
    except NameError:
        print("⬜ Exercise 1: implement batch_process")

    try:
        rl = RateLimiter(max_per_second=10)  # type: ignore
        start = time.monotonic()
        for _ in range(5):
            await rl.acquire()  # type: ignore
        print("✅ Exercise 2 passed!")
    except NameError:
        print("⬜ Exercise 2: implement RateLimiter")

    try:
        async def slow(): await asyncio.sleep(1.0); return "slow"
        async def fast(): await asyncio.sleep(0.1); return "fast"
        r = await first_success([slow(), fast()])  # type: ignore
        assert r == "fast"
        print("✅ Exercise 3 passed!")
    except NameError:
        print("⬜ Exercise 3: implement first_success")


if __name__ == "__main__":
    asyncio.run(run_tests())

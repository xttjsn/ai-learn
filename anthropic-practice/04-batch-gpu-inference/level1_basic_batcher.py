"""
Level 1: Basic Request Batcher

Collect individual inference requests and batch them for GPU processing.
Two triggers to flush a batch:
1. Batch reaches max_batch_size
2. Timeout expires (don't let requests wait forever)

This is the core building block of inference serving systems
(vLLM, TGI, Triton all do this).

Key concepts:
- Producer-consumer pattern
- Batch formation with timeout
- Fan-out results back to individual requesters
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any
from asyncio import Future


@dataclass
class InferenceRequest:
    """A single inference request."""
    prompt: str
    future: Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())
    created_at: float = field(default_factory=time.time)


class BasicBatcher:
    """
    Collects requests and forms batches.

    TODO: Implement this!

    API:
    - submit(prompt) → awaitable result
    - Internally batches requests and calls process_batch()

    Hints:
    1. Use asyncio.Queue to collect requests
    2. Background task that drains the queue
    3. Flush when batch_size reached OR timeout expires
    4. Each request has a Future — resolve it when batch completes
    """

    def __init__(self, max_batch_size: int = 32, max_wait_ms: float = 50):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

    async def submit(self, prompt: str) -> str:
        """Submit a request and await its result."""
        pass

    async def process_batch(self, prompts: list[str]) -> list[str]:
        """
        Simulate GPU inference on a batch.
        In reality, this calls the model.
        """
        # Simulate GPU work — batched is faster than individual
        await asyncio.sleep(0.01)  # 10ms "inference"
        return [f"Response to: {p}" for p in prompts]


# --------------- SOLUTION BELOW ---------------


class BasicBatcherSolution:
    """Reference implementation."""

    def __init__(self, max_batch_size: int = 32, max_wait_ms: float = 50):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._queue: asyncio.Queue[InferenceRequest] = asyncio.Queue()
        self._running = False
        self._task = None

    async def start(self):
        """Start the background batch processing loop."""
        self._running = True
        self._task = asyncio.create_task(self._batch_loop())

    async def stop(self):
        """Stop the batcher gracefully."""
        self._running = False
        if self._task:
            await self._task

    async def submit(self, prompt: str) -> str:
        """Submit a request. Returns when the batch containing it completes."""
        request = InferenceRequest(prompt=prompt)
        await self._queue.put(request)
        return await request.future

    async def _batch_loop(self):
        """Main loop: collect requests into batches and process them."""
        while self._running or not self._queue.empty():
            batch: list[InferenceRequest] = []

            # Wait for at least one request
            try:
                first = await asyncio.wait_for(
                    self._queue.get(), timeout=0.1
                )
                batch.append(first)
            except asyncio.TimeoutError:
                continue

            # Collect more requests until batch full or timeout
            deadline = time.time() + self.max_wait_ms / 1000
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    req = await asyncio.wait_for(
                        self._queue.get(), timeout=remaining
                    )
                    batch.append(req)
                except asyncio.TimeoutError:
                    break

            # Process the batch
            if batch:
                prompts = [r.prompt for r in batch]
                try:
                    results = await self.process_batch(prompts)
                    # Fan out results to individual futures
                    for req, result in zip(batch, results):
                        req.future.set_result(result)
                except Exception as e:
                    for req in batch:
                        req.future.set_exception(e)

    async def process_batch(self, prompts: list[str]) -> list[str]:
        """Simulate GPU batch inference."""
        # Key insight: batch processing time is roughly constant
        # regardless of batch size (GPU parallelism)
        await asyncio.sleep(0.01)
        return [f"Response to: {p}" for p in prompts]


# --------------- TESTING ---------------

async def test_batcher():
    batcher = BasicBatcherSolution(max_batch_size=4, max_wait_ms=50)
    await batcher.start()

    # Submit multiple requests concurrently
    tasks = [
        asyncio.create_task(batcher.submit(f"prompt_{i}"))
        for i in range(10)
    ]

    results = await asyncio.gather(*tasks)
    assert len(results) == 10
    assert all("Response to:" in r for r in results)

    await batcher.stop()
    print(f"Got {len(results)} results")
    print("All tests passed! ✅")


if __name__ == "__main__":
    asyncio.run(test_batcher())

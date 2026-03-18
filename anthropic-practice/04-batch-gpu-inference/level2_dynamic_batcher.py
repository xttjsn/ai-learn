"""
Level 2: Dynamic & Continuous Batching

Key improvements over basic batching:
1. Sequence-length-aware batching (minimize padding waste)
2. Continuous batching (add new requests to running batch)
3. Token budget instead of fixed batch size

This is how modern inference engines (vLLM, TGI) actually work.

Concepts:
- Padding waste: if batch has seqs of length [10, 100], you waste 90 tokens
- Continuous batching: as sequences finish generating, slot in new ones
- Token budget: limit total tokens in batch, not number of sequences
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from asyncio import Future


@dataclass
class Request:
    prompt: str
    prompt_length: int = 0  # Tokenized length
    max_new_tokens: int = 100
    future: Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        # Simplified tokenization: 1 word ≈ 1 token
        self.prompt_length = len(self.prompt.split())


@dataclass
class RunningRequest:
    """A request currently being generated."""
    request: Request
    tokens_generated: int = 0
    output_tokens: list[str] = field(default_factory=list)
    is_done: bool = False


class DynamicBatcher:
    """
    Dynamic batcher with sequence-length-aware batching and continuous batching.

    TODO: Implement this!

    Key differences from basic batcher:
    1. Token budget: max_tokens_in_batch instead of max_batch_size
    2. Sort/bucket by sequence length to minimize padding
    3. Continuous batching: process in iterations, add new requests each iteration
    """

    def __init__(
        self,
        max_batch_tokens: int = 4096,
        max_batch_size: int = 32,
        max_wait_ms: float = 50,
    ):
        self.max_batch_tokens = max_batch_tokens
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

    async def submit(self, prompt: str, max_new_tokens: int = 100) -> str:
        pass


# --------------- SOLUTION BELOW ---------------


class DynamicBatcherSolution:
    """Reference implementation with continuous batching."""

    def __init__(
        self,
        max_batch_tokens: int = 4096,
        max_batch_size: int = 32,
        max_wait_ms: float = 50,
    ):
        self.max_batch_tokens = max_batch_tokens
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._waiting_queue: asyncio.Queue[Request] = asyncio.Queue()
        self._running_batch: list[RunningRequest] = []
        self._running = False
        self._task = None

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._generation_loop())

    async def stop(self):
        self._running = False
        if self._task:
            await self._task

    async def submit(self, prompt: str, max_new_tokens: int = 100) -> str:
        request = Request(prompt=prompt, max_new_tokens=max_new_tokens)
        await self._waiting_queue.put(request)
        return await request.future

    def _current_batch_tokens(self) -> int:
        """Total tokens currently in the running batch."""
        if not self._running_batch:
            return 0
        # Each request occupies: prompt_length + tokens_generated
        return sum(
            r.request.prompt_length + r.tokens_generated
            for r in self._running_batch
        )

    def _can_add_request(self, request: Request) -> bool:
        """Check if adding this request would exceed budget."""
        new_tokens = self._current_batch_tokens() + request.prompt_length
        new_size = len(self._running_batch) + 1
        return (
            new_tokens <= self.max_batch_tokens
            and new_size <= self.max_batch_size
        )

    async def _generation_loop(self):
        """
        Continuous batching loop:
        1. Fill batch with waiting requests (respecting token budget)
        2. Run one generation step (each request generates one token)
        3. Remove completed requests, fan out results
        4. Repeat — new requests can join next iteration
        """
        while self._running or self._running_batch or not self._waiting_queue.empty():
            # Step 1: Fill batch with waiting requests
            filled = False
            while not self._waiting_queue.empty():
                try:
                    request = self._waiting_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                if self._can_add_request(request):
                    self._running_batch.append(RunningRequest(request=request))
                    filled = True
                else:
                    # Put it back — batch is full
                    await self._waiting_queue.put(request)
                    break

            # If no work, wait briefly
            if not self._running_batch:
                try:
                    request = await asyncio.wait_for(
                        self._waiting_queue.get(), timeout=0.1
                    )
                    self._running_batch.append(RunningRequest(request=request))
                except asyncio.TimeoutError:
                    continue

            # Step 2: Run one generation step (simulate)
            await self._generation_step()

            # Step 3: Remove completed requests
            completed = [r for r in self._running_batch if r.is_done]
            self._running_batch = [r for r in self._running_batch if not r.is_done]

            # Fan out results
            for running_req in completed:
                output = " ".join(running_req.output_tokens)
                running_req.request.future.set_result(output)

    async def _generation_step(self):
        """
        Simulate one forward pass: each request generates one token.

        In real inference:
        - This is one transformer forward pass
        - Batch is padded to max sequence length
        - KV cache is updated for each sequence
        """
        # Simulate GPU time — roughly constant per step regardless of batch size
        await asyncio.sleep(0.005)

        for running_req in self._running_batch:
            # Generate one token
            running_req.tokens_generated += 1
            running_req.output_tokens.append(f"tok{running_req.tokens_generated}")

            # Check if done
            if running_req.tokens_generated >= running_req.request.max_new_tokens:
                running_req.is_done = True


# --------------- DISCUSSION ---------------

"""
Key concepts to discuss in the interview:

1. WHY continuous batching?
   Static batching: wait for ALL sequences to finish → GPU idle while
   short sequences pad. Continuous: slot in new requests immediately.
   vLLM paper shows 2-3x throughput improvement.

2. Token budget vs batch size:
   GPU memory = KV cache size, which is proportional to
   sum(sequence_length) across all requests in batch.
   Token budget controls memory, batch size controls compute.

3. Padding waste:
   With static batching, batch of [10, 100] token sequences =
   90 wasted tokens of padding per step × number of steps.
   Solutions: bucket by length, or continuous batching.

4. KV cache management:
   - Pre-allocate KV cache slots for max_seq_len
   - PagedAttention (vLLM): allocate KV cache in pages, like virtual memory
   - When batch is full on memory, preempt low-priority requests

5. Scheduling priorities:
   - First-come-first-served vs priority queues
   - SLO-aware: prioritize requests close to deadline
   - Preemption: swap out KV cache of low-priority request

6. Fault tolerance:
   - Request retry on GPU failure
   - KV cache checkpoint for long sequences
   - Health checks and circuit breakers
"""


# --------------- TESTING ---------------

async def test_dynamic_batcher():
    batcher = DynamicBatcherSolution(
        max_batch_tokens=1000,
        max_batch_size=4,
        max_wait_ms=50,
    )
    await batcher.start()

    # Submit requests with different lengths
    tasks = [
        asyncio.create_task(batcher.submit(f"short prompt {i}", max_new_tokens=5))
        for i in range(8)
    ]

    results = await asyncio.gather(*tasks)
    assert len(results) == 8
    assert all(len(r) > 0 for r in results)

    await batcher.stop()
    print(f"Got {len(results)} results")
    for i, r in enumerate(results):
        print(f"  Request {i}: {r}")
    print("All tests passed! ✅")


if __name__ == "__main__":
    asyncio.run(test_dynamic_batcher())

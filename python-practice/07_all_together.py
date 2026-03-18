"""
07 - All Together: Mini Inference Server
========================================
Combines typing, dataclasses, pydantic, protocols, and asyncio
into a small but realistic inference request pipeline.

Run with: python3 07_all_together.py
"""
import asyncio
import time
import random
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable, TypeVar, Literal
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════
# LAYER 1: Domain Types (Pydantic for API, dataclasses for internal)
# ═══════════════════════════════════════════════════════════════════════

class ModelType(str, Enum):
    LLAMA = "llama"
    QWEN = "qwen"
    GEMMA = "gemma"


# API-facing models (validated)
class CompletionRequest(BaseModel):
    """Incoming API request — validated by Pydantic."""
    model: ModelType
    prompt: str = Field(min_length=1, max_length=8192)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False

    @field_validator("prompt")
    @classmethod
    def strip_prompt(cls, v: str) -> str:
        return v.strip()


class CompletionResponse(BaseModel):
    """Outgoing API response."""
    request_id: str
    model: str
    text: str
    tokens_generated: int
    latency_ms: float
    created_at: datetime = Field(default_factory=datetime.now)


# Internal models (lightweight, no validation overhead)
@dataclass
class InferenceJob:
    """Internal job representation."""
    request_id: str
    request: CompletionRequest
    created_at: float = field(default_factory=time.monotonic)
    priority: int = 0  # Lower = higher priority


@dataclass(frozen=True)
class ModelInfo:
    """Immutable model metadata."""
    name: str
    model_type: ModelType
    params_b: float
    max_context: int
    tokens_per_second: float  # Simulated throughput


# ═══════════════════════════════════════════════════════════════════════
# LAYER 2: Protocols (interfaces)
# ═══════════════════════════════════════════════════════════════════════

@runtime_checkable
class InferenceEngine(Protocol):
    """Any backend that can run inference."""
    async def generate(self, prompt: str, max_tokens: int, temperature: float) -> str: ...
    @property
    def model_name(self) -> str: ...
    @property
    def is_ready(self) -> bool: ...


class MetricsCollector(Protocol):
    """Anything that can record metrics."""
    def record_latency(self, model: str, latency_ms: float) -> None: ...
    def record_tokens(self, model: str, count: int) -> None: ...
    def get_summary(self) -> dict[str, object]: ...


# ═══════════════════════════════════════════════════════════════════════
# LAYER 3: Implementations
# ═══════════════════════════════════════════════════════════════════════

class SimulatedEngine:
    """Simulates an inference engine. Satisfies InferenceEngine protocol."""

    def __init__(self, model_info: ModelInfo):
        self._info = model_info
        self._ready = False

    @property
    def model_name(self) -> str:
        return self._info.name

    @property
    def is_ready(self) -> bool:
        return self._ready

    async def load(self):
        """Simulate model loading."""
        print(f"    Loading {self._info.name} ({self._info.params_b}B params)...")
        await asyncio.sleep(0.3)
        self._ready = True
        print(f"    ✅ {self._info.name} ready")

    async def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        if not self._ready:
            raise RuntimeError(f"Model {self._info.name} not loaded")

        # Simulate token generation based on model speed
        num_tokens = min(max_tokens, random.randint(10, max_tokens))
        gen_time = num_tokens / self._info.tokens_per_second
        await asyncio.sleep(gen_time)

        words = ["The", "answer", "is", "that", "we", "need", "to", "consider",
                 "multiple", "factors", "including", "the", "architecture"]
        text = " ".join(random.choices(words, k=num_tokens))
        return text


@dataclass
class SimpleMetrics:
    """Simple in-memory metrics. Satisfies MetricsCollector protocol."""
    latencies: dict[str, list[float]] = field(default_factory=dict)
    token_counts: dict[str, int] = field(default_factory=dict)

    def record_latency(self, model: str, latency_ms: float) -> None:
        self.latencies.setdefault(model, []).append(latency_ms)

    def record_tokens(self, model: str, count: int) -> None:
        self.token_counts[model] = self.token_counts.get(model, 0) + count

    def get_summary(self) -> dict[str, object]:
        summary = {}
        for model, lats in self.latencies.items():
            summary[model] = {
                "requests": len(lats),
                "avg_latency_ms": sum(lats) / len(lats),
                "p99_latency_ms": sorted(lats)[int(len(lats) * 0.99)] if lats else 0,
                "total_tokens": self.token_counts.get(model, 0),
            }
        return summary


# ═══════════════════════════════════════════════════════════════════════
# LAYER 4: Server (ties everything together with async)
# ═══════════════════════════════════════════════════════════════════════

class InferenceServer:
    """Main server: routes requests to engines, collects metrics."""

    def __init__(self, metrics: MetricsCollector):
        self._engines: dict[ModelType, InferenceEngine] = {}
        self._metrics = metrics
        self._queue: asyncio.Queue[InferenceJob] = asyncio.Queue(maxsize=100)
        self._request_counter = 0
        self._semaphore = asyncio.Semaphore(5)  # Max 5 concurrent inferences

    def register_engine(self, model_type: ModelType, engine: InferenceEngine) -> None:
        if not isinstance(engine, InferenceEngine):
            raise TypeError(f"{engine} does not satisfy InferenceEngine protocol")
        self._engines[model_type] = engine

    async def startup(self):
        """Load all models concurrently."""
        print("\n🚀 Starting inference server...")
        load_tasks = []
        for engine in self._engines.values():
            if hasattr(engine, "load"):
                load_tasks.append(engine.load())  # type: ignore
        await asyncio.gather(*load_tasks)
        print("🟢 All models loaded!\n")

    async def handle_request(self, request: CompletionRequest) -> CompletionResponse:
        """Process a single request."""
        self._request_counter += 1
        request_id = f"req-{self._request_counter:04d}"

        engine = self._engines.get(request.model)
        if not engine:
            raise ValueError(f"No engine for model {request.model}")
        if not engine.is_ready:
            raise RuntimeError(f"Model {request.model} not ready")

        async with self._semaphore:
            start = time.monotonic()
            text = await engine.generate(request.prompt, request.max_tokens, request.temperature)
            elapsed_ms = (time.monotonic() - start) * 1000

        tokens = len(text.split())
        self._metrics.record_latency(engine.model_name, elapsed_ms)
        self._metrics.record_tokens(engine.model_name, tokens)

        return CompletionResponse(
            request_id=request_id,
            model=engine.model_name,
            text=text,
            tokens_generated=tokens,
            latency_ms=round(elapsed_ms, 2),
        )

    async def handle_batch(self, requests: list[CompletionRequest]) -> list[CompletionResponse]:
        """Handle multiple requests concurrently."""
        return await asyncio.gather(*[self.handle_request(r) for r in requests])

    def get_metrics(self) -> dict[str, object]:
        return self._metrics.get_summary()


# ═══════════════════════════════════════════════════════════════════════
# EXERCISES
# ═══════════════════════════════════════════════════════════════════════

# TODO 1: Add a `stream_response` method to InferenceServer that:
#          - Takes a CompletionRequest
#          - Returns an async generator yielding tokens one by one
#          - Each token has a small delay (simulating streaming)
#          Hint: use `async def stream_response(...) -> AsyncGenerator[str, None]`


# TODO 2: Add a `health_check` method to InferenceServer that:
#          - Returns a dict with: {"status": "ok"/"degraded", "models": {name: ready}}
#          - Status is "ok" if all models ready, "degraded" otherwise


# TODO 3: Create a `CachingEngine` class that:
#          - Wraps another InferenceEngine
#          - Caches results by (prompt, max_tokens) tuple
#          - Returns cached result if available (skip generation)
#          - Satisfies InferenceEngine protocol


# ═══════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════

async def main():
    # Setup
    models = {
        ModelType.LLAMA: ModelInfo("llama-3-8b", ModelType.LLAMA, 8.0, 8192, 50.0),
        ModelType.QWEN: ModelInfo("qwen2.5-7b", ModelType.QWEN, 7.0, 32768, 60.0),
        ModelType.GEMMA: ModelInfo("gemma-2-9b", ModelType.GEMMA, 9.0, 8192, 45.0),
    }

    metrics = SimpleMetrics()
    server = InferenceServer(metrics)

    # Register engines
    for model_type, info in models.items():
        engine = SimulatedEngine(info)
        server.register_engine(model_type, engine)

    # Start
    await server.startup()

    # Single request
    print("── Single Request ──")
    resp = await server.handle_request(
        CompletionRequest(model=ModelType.LLAMA, prompt="What is attention?", max_tokens=20)
    )
    print(f"  {resp.request_id}: {resp.tokens_generated} tokens in {resp.latency_ms:.0f}ms")
    assert resp.tokens_generated > 0

    # Batch request
    print("\n── Batch Requests ──")
    batch = [
        CompletionRequest(model=ModelType.LLAMA, prompt=f"Question {i}", max_tokens=15)
        for i in range(5)
    ] + [
        CompletionRequest(model=ModelType.QWEN, prompt=f"Query {i}", max_tokens=15)
        for i in range(5)
    ]

    start = time.monotonic()
    responses = await server.handle_batch(batch)
    elapsed = time.monotonic() - start
    print(f"  Processed {len(responses)} requests in {elapsed:.2f}s")
    assert len(responses) == 10

    # Validation test
    print("\n── Validation ──")
    try:
        CompletionRequest(model=ModelType.LLAMA, prompt="", max_tokens=20)
        assert False, "Should have failed validation"
    except Exception as e:
        print(f"  ✅ Caught invalid request: {type(e).__name__}")

    try:
        CompletionRequest(model=ModelType.LLAMA, prompt="test", max_tokens=99999)
        assert False
    except Exception:
        print("  ✅ Caught invalid max_tokens")

    # Metrics
    print("\n── Metrics ──")
    summary = server.get_metrics()
    for model, stats in summary.items():
        print(f"  {model}: {stats}")

    # Protocol check
    print("\n── Protocol Checks ──")
    engine = SimulatedEngine(models[ModelType.LLAMA])
    assert isinstance(engine, InferenceEngine), "Should satisfy protocol"
    assert isinstance(metrics, MetricsCollector), "Should satisfy protocol"
    print("  ✅ All protocol checks passed")

    print("\n✅ All tests passed! Mini inference server working.")

    # Exercise hints
    print("\n── Exercises ──")
    print("⬜ Exercise 1: Add stream_response to InferenceServer")
    print("⬜ Exercise 2: Add health_check to InferenceServer")
    print("⬜ Exercise 3: Create CachingEngine wrapper")


if __name__ == "__main__":
    asyncio.run(main())

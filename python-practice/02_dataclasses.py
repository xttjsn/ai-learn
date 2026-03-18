"""
02 - Dataclasses
================
Modern Python data containers. Run with: python3 02_dataclasses.py
"""
from dataclasses import dataclass, field, asdict, astuple
from datetime import datetime


# ─── Basic Dataclass ─────────────────────────────────────────────────
@dataclass
class Point:
    x: float
    y: float

    def distance_to(self, other: "Point") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


# Auto-generates: __init__, __repr__, __eq__
# p = Point(1.0, 2.0) → Point(x=1.0, y=2.0)


# ─── Defaults and field() ────────────────────────────────────────────
@dataclass
class Model:
    name: str
    parameters: int
    quantized: bool = False
    tags: list[str] = field(default_factory=list)  # Mutable default → use field()

    @property
    def size_b(self) -> float:
        """Rough size in billions."""
        return self.parameters / 1_000_000_000


# ─── __post_init__ ───────────────────────────────────────────────────
@dataclass
class TrainingRun:
    model_name: str
    epochs: int
    learning_rate: float
    started_at: datetime = field(default_factory=datetime.now)
    run_id: str = field(init=False)  # Computed, not passed to __init__

    def __post_init__(self):
        """Called after __init__ — use for validation or computed fields."""
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        self.run_id = f"{self.model_name}-{self.epochs}ep-{id(self):x}"


# ─── Frozen (immutable) ─────────────────────────────────────────────
@dataclass(frozen=True)
class Config:
    """Immutable config — can be used as dict key or in sets."""
    backend: str
    num_gpus: int
    precision: str = "fp16"


# ─── Ordering ────────────────────────────────────────────────────────
@dataclass(order=True)
class Priority:
    """Ordered by (level, name) — supports <, >, sorted()."""
    level: int
    name: str = field(compare=False)  # Don't include name in ordering


# ─── Slots (Python 3.10+) — faster attribute access, less memory ────
@dataclass(slots=True)
class Token:
    token_id: int
    text: str
    logprob: float


# ─── Inheritance ─────────────────────────────────────────────────────
@dataclass
class BaseModel:
    name: str
    params: int

@dataclass
class FineTunedModel(BaseModel):
    base_model: str = ""
    lora_rank: int = 16

    @property
    def description(self) -> str:
        return f"{self.name} (fine-tuned from {self.base_model}, LoRA r={self.lora_rank})"


# ─── Serialization ───────────────────────────────────────────────────
def demo_serialization():
    m = Model("Qwen2.5-7B", 7_000_000_000, tags=["instruct", "awq"])
    print(f"  asdict:  {asdict(m)}")
    print(f"  astuple: {astuple(m)}")
    # For JSON: json.dumps(asdict(m))


# ═══════════════════════════════════════════════════════════════════════
# EXERCISES
# ═══════════════════════════════════════════════════════════════════════

# TODO 1: Create a @dataclass called `GPUInfo` with:
#   - name: str
#   - vram_gb: int
#   - cuda_cores: int
#   - tdp_watts: int = 350
#   - Add a property `perf_per_watt` returning cuda_cores / tdp_watts


# TODO 2: Create a frozen @dataclass called `Endpoint` with:
#   - host: str
#   - port: int
#   - Add a property `url` returning f"http://{host}:{port}"
#   - Verify you can use it as a dict key


# TODO 3: Create two dataclasses with inheritance:
#   - `Job` with: job_id: str, status: str = "pending"
#   - `TrainingJob(Job)` with: model: str, gpu_count: int = 1
#   - Add __post_init__ to TrainingJob that raises ValueError if gpu_count < 1


# ═══════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Point
    p1, p2 = Point(0, 0), Point(3, 4)
    assert p1.distance_to(p2) == 5.0
    assert p1 != p2
    assert Point(1, 1) == Point(1, 1)

    # Model
    m = Model("Qwen2.5-7B", 7_000_000_000, tags=["instruct"])
    assert m.size_b == 7.0
    assert m.quantized is False
    m2 = Model("Qwen2.5-7B", 7_000_000_000, tags=["instruct"])
    assert m == m2  # Equal by value
    assert m.tags is not m2.tags  # But separate list instances

    # TrainingRun
    run = TrainingRun("llama", 3, 1e-4)
    assert run.run_id.startswith("llama-3ep-")
    try:
        TrainingRun("bad", 1, -0.01)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Frozen
    c1 = Config("glx", 2)
    c2 = Config("glx", 2)
    assert c1 == c2
    assert hash(c1) == hash(c2)
    config_set = {c1, c2}
    assert len(config_set) == 1
    try:
        c1.backend = "egl"  # type: ignore
        assert False, "Should be frozen"
    except AttributeError:
        pass

    # Ordering
    tasks = [Priority(3, "low"), Priority(1, "critical"), Priority(2, "medium")]
    assert sorted(tasks)[0].name == "critical"

    # Slots
    t = Token(42, "hello", -0.5)
    assert t.text == "hello"

    # Inheritance
    ft = FineTunedModel("my-model", 7_000_000, base_model="llama-7b", lora_rank=32)
    assert "LoRA r=32" in ft.description

    # Serialization
    demo_serialization()

    print("✅ All provided tests passed!")

    # Exercise tests
    try:
        gpu = GPUInfo("RTX 3080 Ti", 12, 10240, 350)  # type: ignore
        assert gpu.perf_per_watt == 10240 / 350  # type: ignore
        print("✅ Exercise 1 passed!")
    except NameError:
        print("⬜ Exercise 1: implement GPUInfo")

    try:
        ep = Endpoint("localhost", 8080)  # type: ignore
        assert ep.url == "http://localhost:8080"  # type: ignore
        d = {ep: "test"}
        assert d[ep] == "test"
        print("✅ Exercise 2 passed!")
    except NameError:
        print("⬜ Exercise 2: implement Endpoint")

    try:
        tj = TrainingJob("job-1", model="llama")  # type: ignore
        assert tj.status == "pending"  # type: ignore
        assert tj.gpu_count == 1  # type: ignore
        try:
            TrainingJob("job-2", model="llama", gpu_count=0)  # type: ignore
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
        print("✅ Exercise 3 passed!")
    except NameError:
        print("⬜ Exercise 3: implement Job and TrainingJob")

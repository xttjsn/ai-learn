"""
03 - Pydantic Models
====================
Validation, serialization, settings. Run with: python3 03_pydantic_models.py
pip install pydantic  (if not already installed)
"""
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from datetime import datetime
from enum import Enum


# ─── Basic Model ─────────────────────────────────────────────────────
class User(BaseModel):
    name: str
    email: str
    age: int = Field(ge=0, le=150, description="Age in years")
    tags: list[str] = []

# Pydantic auto-validates on construction:
#   User(name="xtt", email="x@x.com", age=-1)  → ValidationError


# ─── Enums ───────────────────────────────────────────────────────────
class GPUType(str, Enum):
    A100 = "a100"
    H100 = "h100"
    B200 = "b200"
    RTX_3080TI = "rtx_3080ti"


# ─── Nested Models ───────────────────────────────────────────────────
class GPUConfig(BaseModel):
    gpu_type: GPUType
    count: int = Field(ge=1, le=8)
    memory_gb: int

class InferenceRequest(BaseModel):
    model_name: str
    prompt: str = Field(min_length=1, max_length=4096)
    max_tokens: int = Field(default=256, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    gpu: GPUConfig | None = None
    created_at: datetime = Field(default_factory=datetime.now)


# ─── Field Validators ────────────────────────────────────────────────
class DockerImage(BaseModel):
    repository: str
    tag: str = "latest"

    @field_validator("repository")
    @classmethod
    def validate_repo(cls, v: str) -> str:
        if " " in v or not v:
            raise ValueError("repository must be non-empty with no spaces")
        return v.lower()

    @field_validator("tag")
    @classmethod
    def validate_tag(cls, v: str) -> str:
        if not v:
            return "latest"
        return v


# ─── Model Validators (cross-field) ─────────────────────────────────
class PortRange(BaseModel):
    start: int = Field(ge=1, le=65535)
    end: int = Field(ge=1, le=65535)

    @model_validator(mode="after")
    def validate_range(self) -> "PortRange":
        if self.start > self.end:
            raise ValueError(f"start ({self.start}) must be <= end ({self.end})")
        return self


# ─── Serialization ───────────────────────────────────────────────────
def demo_serialization():
    req = InferenceRequest(
        model_name="qwen2.5-7b",
        prompt="Explain transformers",
        gpu=GPUConfig(gpu_type=GPUType.RTX_3080TI, count=1, memory_gb=12),
    )
    # To dict
    d = req.model_dump()
    print(f"  model_dump: {d['model_name']}, gpu={d['gpu']['gpu_type']}")

    # To JSON string
    j = req.model_json_schema()
    print(f"  json_schema keys: {list(j.get('properties', {}).keys())}")

    # From dict
    req2 = InferenceRequest.model_validate(d)
    assert req2.model_name == req.model_name

    # From JSON string
    json_str = req.model_dump_json()
    req3 = InferenceRequest.model_validate_json(json_str)
    assert req3.prompt == req.prompt


# ─── Immutable Model ────────────────────────────────────────────────
class FrozenConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    backend: str
    precision: str = "fp16"


# ─── Aliases (useful for JSON APIs with different naming) ────────────
class APIResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    status_code: int = Field(alias="statusCode")
    error_message: str | None = Field(default=None, alias="errorMessage")


# ═══════════════════════════════════════════════════════════════════════
# EXERCISES
# ═══════════════════════════════════════════════════════════════════════

# TODO 1: Create a Pydantic model `TrainingConfig` with:
#   - model_name: str (min_length=1)
#   - batch_size: int (must be power of 2, between 1 and 1024) — use field_validator
#   - learning_rate: float (gt=0, lt=1)
#   - epochs: int (ge=1, le=1000)
#   - fp16: bool = True


# TODO 2: Create a model `Cluster` with:
#   - name: str
#   - nodes: list[Node] where Node has: hostname: str, gpus: int
#   - Add a @property `total_gpus` returning sum of all node gpus
#   - Add a model_validator ensuring total_gpus >= 1


# TODO 3: Create a model `Experiment` that serializes to/from JSON:
#   - name: str, metrics: dict[str, float], timestamp: datetime
#   - Write a function `round_trip(exp: Experiment) -> Experiment`
#     that dumps to JSON and parses back, asserting equality


# ═══════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # User validation
    u = User(name="xtt", email="xtt@test.com", age=30)
    assert u.age == 30
    try:
        User(name="xtt", email="xtt@test.com", age=-1)
        assert False
    except Exception:
        pass

    # Nested
    req = InferenceRequest(
        model_name="llama",
        prompt="Hello",
        gpu=GPUConfig(gpu_type="h100", count=2, memory_gb=80),
    )
    assert req.gpu.count == 2
    assert req.gpu.gpu_type == GPUType.H100

    # Validator
    img = DockerImage(repository="NVIDIA/Triton")
    assert img.repository == "nvidia/triton"
    try:
        DockerImage(repository="")
        assert False
    except Exception:
        pass

    # Model validator
    PortRange(start=80, end=443)
    try:
        PortRange(start=443, end=80)
        assert False
    except Exception:
        pass

    # Serialization
    demo_serialization()

    # Frozen
    fc = FrozenConfig(backend="glx")
    try:
        fc.backend = "egl"
        assert False
    except Exception:
        pass

    # Aliases
    resp = APIResponse.model_validate({"statusCode": 200, "errorMessage": None})
    assert resp.status_code == 200

    print("✅ All provided tests passed!")

    # Exercise tests
    try:
        tc = TrainingConfig(model_name="llama", batch_size=32, learning_rate=0.001, epochs=10)  # type: ignore
        assert tc.fp16 is True  # type: ignore
        try:
            TrainingConfig(model_name="llama", batch_size=3, learning_rate=0.001, epochs=10)  # type: ignore
            assert False, "batch_size=3 should fail (not power of 2)"
        except Exception:
            pass
        print("✅ Exercise 1 passed!")
    except NameError:
        print("⬜ Exercise 1: implement TrainingConfig")

    try:
        n1 = Node(hostname="gpu-01", gpus=4)  # type: ignore
        c = Cluster(name="dev", nodes=[n1])  # type: ignore
        assert c.total_gpus == 4  # type: ignore
        try:
            Cluster(name="empty", nodes=[])  # type: ignore
            assert False
        except Exception:
            pass
        print("✅ Exercise 2 passed!")
    except NameError:
        print("⬜ Exercise 2: implement Node and Cluster")

    try:
        exp = Experiment(name="run-1", metrics={"loss": 0.5}, timestamp=datetime.now())  # type: ignore
        exp2 = round_trip(exp)  # type: ignore
        assert exp2.name == exp.name  # type: ignore
        print("✅ Exercise 3 passed!")
    except NameError:
        print("⬜ Exercise 3: implement Experiment and round_trip")

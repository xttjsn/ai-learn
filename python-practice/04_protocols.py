"""
04 - Protocols (Structural Subtyping)
=====================================
Duck typing with static type checking. Run with: python3 04_protocols.py
"""
from typing import Protocol, runtime_checkable
from dataclasses import dataclass


# ─── The Problem: You want type safety without inheritance ───────────
#
# In Go, you'd use interfaces. In Rust, traits.
# In Python, Protocol gives you structural subtyping:
# "If it has the right methods, it matches — no need to inherit."


# ─── Defining a Protocol ─────────────────────────────────────────────
class Serializable(Protocol):
    """Any object that can convert to/from dict."""
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: dict) -> "Serializable": ...


class HasName(Protocol):
    """Anything with a name attribute."""
    @property
    def name(self) -> str: ...


# ─── Classes that satisfy Protocols (no inheritance needed!) ─────────
@dataclass
class GPU:
    name: str
    vram_gb: int

    def to_dict(self) -> dict:
        return {"name": self.name, "vram_gb": self.vram_gb}

    @classmethod
    def from_dict(cls, data: dict) -> "GPU":
        return cls(name=data["name"], vram_gb=data["vram_gb"])


@dataclass
class Cluster:
    name: str
    node_count: int

    def to_dict(self) -> dict:
        return {"name": self.name, "node_count": self.node_count}

    @classmethod
    def from_dict(cls, data: dict) -> "Cluster":
        return cls(name=data["name"], node_count=data["node_count"])


# Both GPU and Cluster satisfy Serializable AND HasName — without inheriting!

def save_all(items: list[Serializable]) -> list[dict]:
    """Works with any Serializable — GPU, Cluster, or anything else."""
    return [item.to_dict() for item in items]


def print_names(items: list[HasName]) -> list[str]:
    """Works with anything that has a .name property."""
    return [item.name for item in items]


# ─── runtime_checkable — isinstance() with Protocols ────────────────
@runtime_checkable
class Runnable(Protocol):
    def run(self) -> int: ...


class TrainingJob:
    def run(self) -> int:
        print("    Training...")
        return 0

class InferenceJob:
    def run(self) -> int:
        print("    Inferring...")
        return 0

class NotAJob:
    pass


def execute_if_runnable(obj: object) -> int | None:
    """Only runs if obj satisfies the Runnable protocol."""
    if isinstance(obj, Runnable):
        return obj.run()
    return None


# ─── Protocol with generic ──────────────────────────────────────────
from typing import TypeVar

T = TypeVar("T")

class Repository(Protocol[T]):
    def get(self, id: str) -> T | None: ...
    def save(self, id: str, item: T) -> None: ...
    def list_all(self) -> list[T]: ...


# A simple in-memory implementation
class InMemoryRepo:
    """Satisfies Repository[T] for any T."""
    def __init__(self):
        self._store: dict[str, object] = {}

    def get(self, id: str):
        return self._store.get(id)

    def save(self, id: str, item) -> None:
        self._store[id] = item

    def list_all(self) -> list:
        return list(self._store.values())


# ─── Combining Protocols ────────────────────────────────────────────
class Named(Protocol):
    @property
    def name(self) -> str: ...

class Versioned(Protocol):
    @property
    def version(self) -> str: ...

class NamedAndVersioned(Named, Versioned, Protocol):
    """Combined protocol — must satisfy both."""
    ...

@dataclass
class Package:
    name: str
    version: str


def format_package(pkg: NamedAndVersioned) -> str:
    return f"{pkg.name}@{pkg.version}"


# ═══════════════════════════════════════════════════════════════════════
# EXERCISES
# ═══════════════════════════════════════════════════════════════════════

# TODO 1: Define a Protocol `Measurable` with a method `size_bytes() -> int`
#          Then create two classes `FileBlob` and `MemoryBuffer` that satisfy it
#          Write `total_size(items: list[Measurable]) -> int`


# TODO 2: Define a @runtime_checkable Protocol `Closeable` with `close() -> None`
#          Write `safe_close(obj: object) -> bool` that closes if Closeable, returns True/False


# TODO 3: Define a Protocol `Logger` with method `log(msg: str, level: str) -> None`
#          Implement `ConsoleLogger` and `NullLogger` (does nothing)
#          Write `do_work(logger: Logger) -> None` that logs "starting" and "done"


# ═══════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Serializable
    gpu = GPU("RTX 3080 Ti", 12)
    cluster = Cluster("dev", 3)
    dicts = save_all([gpu, cluster])
    assert dicts == [{"name": "RTX 3080 Ti", "vram_gb": 12}, {"name": "dev", "node_count": 3}]

    # HasName
    names = print_names([gpu, cluster])
    assert names == ["RTX 3080 Ti", "dev"]

    # Round-trip
    gpu2 = GPU.from_dict(gpu.to_dict())
    assert gpu2 == gpu

    # runtime_checkable
    assert isinstance(TrainingJob(), Runnable)
    assert not isinstance(NotAJob(), Runnable)
    assert execute_if_runnable(TrainingJob()) == 0
    assert execute_if_runnable(NotAJob()) is None

    # Generic repository
    repo = InMemoryRepo()
    repo.save("g1", gpu)
    assert repo.get("g1") == gpu
    assert repo.list_all() == [gpu]

    # Combined protocol
    pkg = Package("torch", "2.5.0")
    assert format_package(pkg) == "torch@2.5.0"

    print("✅ All provided tests passed!")

    # Exercise tests
    try:
        fb = FileBlob(1024)  # type: ignore
        mb = MemoryBuffer(512)  # type: ignore
        assert total_size([fb, mb]) == 1536  # type: ignore
        print("✅ Exercise 1 passed!")
    except NameError:
        print("⬜ Exercise 1: implement Measurable, FileBlob, MemoryBuffer, total_size")

    try:
        class FakeConn:
            def __init__(self): self.closed = False
            def close(self): self.closed = True
        fc = FakeConn()
        assert safe_close(fc) is True  # type: ignore
        assert fc.closed
        assert safe_close("not closeable") is False  # type: ignore
        print("✅ Exercise 2 passed!")
    except NameError:
        print("⬜ Exercise 2: implement Closeable and safe_close")

    try:
        cl = ConsoleLogger()  # type: ignore
        nl = NullLogger()  # type: ignore
        do_work(cl)  # type: ignore
        do_work(nl)  # type: ignore
        print("✅ Exercise 3 passed!")
    except NameError:
        print("⬜ Exercise 3: implement Logger, ConsoleLogger, NullLogger, do_work")

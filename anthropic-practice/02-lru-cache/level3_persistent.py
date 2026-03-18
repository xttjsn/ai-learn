"""
Level 3: LRU Cache with Crash Recovery (WAL)

THE key follow-up in the Anthropic interview:
"What if the process crashes? How do you restore the cache?"

Approach: Write-Ahead Log (WAL)
- Log every cache mutation to disk before applying
- On recovery, replay the log to reconstruct state
- Periodically compact the log (snapshot + truncate)

Discussion points the interviewer will ask:
1. When to write to disk? Every put? Batched?
   → Trade-off: durability vs performance
2. What about write amplification?
   → WAL is append-only, fast sequential writes
3. CPU bound vs IO bound?
   → Cache operations are CPU bound (hashing, OrderedDict)
   → Disk writes are IO bound
   → Solution: async writes, write batching
"""

import json
import os
import pickle
from collections import OrderedDict
from functools import wraps
from typing import Any, Callable, Optional


class PersistentLRUCache:
    """
    LRU Cache with WAL-based crash recovery.

    TODO: Implement this!

    Requirements:
    1. Standard LRU cache (get/put with eviction)
    2. WAL: append log entry for every put/eviction
    3. Recovery: rebuild cache from WAL on startup
    4. Compaction: snapshot cache state, truncate WAL

    WAL entry format (one per line):
    {"op": "put", "key": "...", "value": "..."}
    {"op": "evict", "key": "..."}
    """

    def __init__(self, maxsize: int = 128, wal_path: str = "cache.wal"):
        self.cache = OrderedDict()
        self.maxsize = maxsize
        self.wal_path = wal_path
        self.snapshot_path = wal_path + ".snapshot"

        if os.path.exists(self.wal_path) or os.path.exists(self.snapshot_path):
            self.recover()

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]

        return None

    def put(self, key: str, value: Any) -> None:
        # Record a put
        self._record({'op': 'put', 'key': key, value: 'value'})

        if key in self.cache:
            self.cache[key] = value
            self.cache.move_to_end(key)
        else:
            self.cache[key] = value
            while len(self.cache) > self.maxsize:
                # record an evict
                evict_key, _ = self.cache.popitem(last=False)
                self._record({'op': 'evict', 'key': evict_key})


    def _record(self, entry):
        with open(self.wal_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
            f.flush()

    def recover(self) -> None:
        """Replay WAL to restore cache state."""
        # 1. Restore snapshot
        if os.path.exists(self.snapshot_path):
            with open(self.snapshot_path, 'r') as f:
                snapshot = json.load(f)
                self.cache = OrderedDict(snapshot['entries'])

        # 2. Replay WAL
        if os.path.exists(self.wal_path):
            with open(self.wal_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        entry = json.load(line)
                    except json.JSONDecodeError:
                        continue

                    if entry['op'] == 'put':
                        self.cache[entry['key']] = entry['value']
                        self.cache.move_to_end(entry['key'])
                        while len(self.cache) > self.maxsize:
                            self.cache.popitem(last=False)

                    elif entry['op'] == 'evict':
                        self.cache.pop(entry['key'])

    def compact(self) -> None:
        """Snapshot current state, truncate WAL."""
        # 1. Snapshot write
        tmp_path = self.snapshot_path + '.tmp'
        with open(tmp_path, 'w') as f:
            snapshot = {'entries': self.cache.items()}
            json.dump(snapshot, f)
            f.flush()

        os.rename(tmp_path, self.snapshot_path)

        # 2. Truncat wal
        with open(self.wal_path, 'w') as f:
            pass # empties it

# --------------- SOLUTION BELOW ---------------


class PersistentLRUCacheSolution:
    """Reference implementation with WAL."""

    def __init__(self, maxsize: int = 128, wal_path: str = "cache.wal"):
        self.maxsize = maxsize
        self.wal_path = wal_path
        self.snapshot_path = wal_path + ".snapshot"
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self._wal_file = None

        # Recover from existing WAL if present
        if os.path.exists(self.snapshot_path) or os.path.exists(self.wal_path):
            self.recover()

        # Open WAL for appending
        self._wal_file = open(self.wal_path, "a")

    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        # Note: we don't log reads to WAL (read doesn't change data,
        # only LRU order — acceptable to lose on crash)
        return self.cache[key]

    def put(self, key: str, value: Any) -> None:
        # Write to WAL FIRST (write-ahead!)
        self._write_wal({"op": "put", "key": key, "value": value})

        if key in self.cache:
            # In cache
            self.cache.move_to_end(key)
            self.cache[key] = value
        else:
            self.cache[key] = value
            if len(self.cache) > self.maxsize:
                evicted_key, _ = self.cache.popitem(last=False)
                self._write_wal({"op": "evict", "key": evicted_key})

    def _write_wal(self, entry: dict) -> None:
        """Append entry to WAL and flush."""
        if self._wal_file:
            self._wal_file.write(json.dumps(entry) + "\n")
            self._wal_file.flush()
            os.fsync(self._wal_file.fileno())  # Ensure durability

    def recover(self) -> None:
        """
        Recovery procedure:
        1. Load snapshot if exists
        2. Replay WAL entries on top
        """
        # Step 1: Load snapshot
        if os.path.exists(self.snapshot_path):
            with open(self.snapshot_path, "r") as f:
                snapshot = json.load(f)
                self.cache = OrderedDict(snapshot["entries"])

        # Step 2: Replay WAL
        if os.path.exists(self.wal_path):
            with open(self.wal_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        # Corrupted entry (crash mid-write) — skip
                        continue

                    if entry["op"] == "put":
                        self.cache[entry["key"]] = entry["value"]
                        self.cache.move_to_end(entry["key"])
                        # Re-enforce maxsize
                        while len(self.cache) > self.maxsize:
                            self.cache.popitem(last=False)
                    elif entry["op"] == "evict":
                        self.cache.pop(entry["key"], None)

    def compact(self) -> None:
        """
        Compaction: write snapshot of current state, truncate WAL.

        This prevents the WAL from growing unbounded.
        In production, do this periodically or when WAL exceeds a threshold.
        """
        # Write snapshot atomically (write to temp, then rename)
        tmp_path = self.snapshot_path + ".tmp"
        with open(tmp_path, "w") as f:
            snapshot = {
                "entries": list(self.cache.items()),
            }
            json.dump(snapshot, f)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, self.snapshot_path)

        # Truncate WAL
        if self._wal_file:
            self._wal_file.close()
        with open(self.wal_path, "w") as f:
            pass  # Empty the file
        self._wal_file = open(self.wal_path, "a")

    def close(self) -> None:
        if self._wal_file:
            self._wal_file.close()

    def __del__(self):
        self.close()


def persistent_lru_cache(maxsize: int = 128, wal_path: str = "cache.wal"):
    """Decorator version using the persistent cache."""

    def decorator(func: Callable) -> Callable:
        cache = PersistentLRUCacheSolution(maxsize=maxsize, wal_path=wal_path)

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str((args, tuple(sorted(kwargs.items()))))

            result = cache.get(key)
            if result is not None:
                return result

            result = func(*args, **kwargs)
            cache.put(key, result)
            return result

        wrapper._cache = cache
        wrapper.compact = cache.compact
        return wrapper

    return decorator


# --------------- DISCUSSION POINTS ---------------

"""
Interview follow-up questions and answers:

Q: "When should you write to disk? Every put is expensive."
A: Options:
   1. Every put (strongest durability, ~1ms per fsync)
   2. Batched writes (buffer N entries, flush together)
   3. Periodic flush (flush every T ms, risk losing last T ms)
   4. Async writes (background thread, best perf, weaker guarantees)
   Trade-off depends on how much data loss is acceptable.

Q: "Is this CPU bound or IO bound?"
A: Mixed:
   - Cache lookup/insert is CPU bound (hashing, OrderedDict ops)
   - WAL writes are IO bound (disk I/O, fsync)
   - For read-heavy workloads: CPU bound
   - For write-heavy workloads: IO bound (fsync dominates)
   Solution: Use async IO for writes, or an append buffer.

Q: "What about concurrent access?"
A: Add threading.Lock around cache operations.
   WAL writes should be serialized anyway (append-only file).
   For high concurrency, consider sharded caches.

Q: "What if the WAL gets corrupted mid-write?"
A: Each entry is a single JSON line. If we crash mid-write,
   the last line will be incomplete/invalid JSON.
   During recovery, we skip invalid lines (json.JSONDecodeError).
   This means we might lose the last operation, which is acceptable.
"""

# --------------- TESTING ---------------

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        wal = os.path.join(tmpdir, "test.wal")

        # Create cache and add entries
        cache = PersistentLRUCacheSolution(maxsize=3, wal_path=wal)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        assert cache.get("a") == 1
        cache.close()

        # Simulate crash recovery
        cache2 = PersistentLRUCacheSolution(maxsize=3, wal_path=wal)
        assert cache2.get("a") == 1
        assert cache2.get("b") == 2
        assert cache2.get("c") == 3

        # Test compaction
        cache2.compact()
        cache2.put("d", 4)  # Should evict oldest
        cache2.close()

        # Recover after compaction
        cache3 = PersistentLRUCacheSolution(maxsize=3, wal_path=wal)
        assert cache3.get("a") is None  # Evicted
        assert cache3.get("d") == 4
        cache3.close()

        print("All tests passed! ✅")

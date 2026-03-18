"""
Level 2: Optimized File Deduplication

Optimization strategy (3-pass approach):
1. Group by file size — unique sizes can't be duplicates
2. Partial hash — hash first 4KB only, eliminates most non-dupes
3. Full hash — only for files that match on size AND partial hash

This dramatically reduces I/O for large directories.

Interview discussion:
- Why is this faster? Avoids reading entire contents of most files.
- For 10,000 files, maybe only 100 need full hashing.
"""

import hashlib
import os
from collections import defaultdict


def hash_file_partial(filepath: str, num_bytes: int = 4096) -> str:
    """Hash only the first num_bytes of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        sha256.update(f.read(num_bytes))
    return sha256.hexdigest()


def hash_file_full(filepath: str, chunk_size: int = 8192) -> str:
    """Hash entire file content."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def find_duplicates_optimized(directory: str) -> list[list[str]]:
    """
    3-pass optimized deduplication.

    TODO: Implement this!

    Pass 1: Group by file size
    Pass 2: For size-groups > 1, group by partial hash (first 4KB)
    Pass 3: For partial-hash-groups > 1, group by full hash
    """
    pass


# --------------- SOLUTION BELOW ---------------


def find_duplicates_optimized_solution(directory: str) -> list[list[str]]:
    """Reference implementation — 3-pass approach."""

    # Pass 1: Group by file size
    size_groups: dict[int, list[str]] = defaultdict(list)
    for root, dirs, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                size = os.path.getsize(filepath)
                size_groups[size].append(filepath)
            except OSError:
                continue

    # Filter to only groups with potential duplicates
    candidates = []
    for size, files in size_groups.items():
        if len(files) > 1:
            candidates.extend(files)

    if not candidates:
        return []

    # Pass 2: Group by partial hash
    partial_groups: dict[str, list[str]] = defaultdict(list)
    for filepath in candidates:
        try:
            partial_hash = hash_file_partial(filepath)
            partial_groups[partial_hash].append(filepath)
        except OSError:
            continue

    # Pass 3: Full hash only for partial-hash matches
    full_groups: dict[str, list[str]] = defaultdict(list)
    for partial_hash, files in partial_groups.items():
        if len(files) > 1:
            for filepath in files:
                try:
                    full_hash = hash_file_full(filepath)
                    full_groups[full_hash].append(filepath)
                except OSError:
                    continue

    return [group for group in full_groups.values() if len(group) > 1]


# --------------- DISCUSSION ---------------

"""
Interview follow-up: IO bound vs CPU bound?

File dedup is primarily IO BOUND:
- Reading files from disk is the bottleneck
- SHA256 hashing is fast (~1 GB/s on modern CPUs)
- Disk reads are ~100-500 MB/s (SSD) or ~50-100 MB/s (HDD)

Implications:
- Threading WILL help (GIL released during disk I/O)
- Multiple threads can overlap disk reads with CPU hashing
- But don't use too many threads — disk contention with HDDs
  (SSDs handle concurrency better)

For CPU bound hashing of in-memory data:
- Threading won't help (GIL)
- Use multiprocessing instead
- Or use faster hash (xxhash, blake3)
"""

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create files: some with same size but different content
        with open(os.path.join(tmpdir, "a.txt"), "wb") as f:
            f.write(b"x" * 1000)
        with open(os.path.join(tmpdir, "b.txt"), "wb") as f:
            f.write(b"x" * 1000)  # Same as a
        with open(os.path.join(tmpdir, "c.txt"), "wb") as f:
            f.write(b"y" * 1000)  # Same size, different content
        with open(os.path.join(tmpdir, "d.txt"), "wb") as f:
            f.write(b"z" * 500)   # Different size

        dupes = find_duplicates_optimized_solution(tmpdir)
        assert len(dupes) == 1
        assert len(dupes[0]) == 2
        print(f"Duplicates: {dupes}")
        print("All tests passed! ✅")

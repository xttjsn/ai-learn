"""
Level 1: Basic File Deduplication

Find duplicate files by hashing their contents.

Key concepts:
- Content-based hashing (SHA256)
- Group files by hash
- Streaming hash for memory efficiency
"""

import hashlib
import os
from collections import defaultdict
from pathlib import Path

def hash_file(path, chunk_size = 8192) -> str:
    sha256 = hashlib.sha256()
    with open(path, 'r') as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.digest()


def find_duplicates(directory: str) -> list[list[str]]:
    """
    Find all duplicate files in a directory (recursive).
    Returns a list of groups, where each group is a list of
    file paths that have identical content.

    TODO: Implement this!

    Hints:
    1. Walk the directory tree
    2. Hash each file's content
    3. Group files by hash
    4. Return groups with more than one file
    """
    groups = defaultdict(list)

    for root, dirs, files in os.walk(directory):
        for filename in files:
            path = os.path.join(root, filename)
            try:
                file_hash = hash_file(path)
                groups[file_hash].append(path)
            except (PermissionError, OSError):
                continue

    return [paths for _, paths in groups if len(paths) > 1]


# --------------- SOLUTION BELOW ---------------


def hash_file(filepath: str, chunk_size: int = 8192) -> str:
    """Hash file contents using SHA256, reading in chunks (memory efficient)."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def find_duplicates_solution(directory: str) -> list[list[str]]:
    """Reference implementation."""
    hash_to_files: dict[str, list[str]] = defaultdict(list)

    for root, dirs, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                file_hash = hash_file(filepath)
                hash_to_files[file_hash].append(filepath)
            except (PermissionError, OSError):
                continue

    # Return only groups with duplicates
    return [group for group in hash_to_files.values() if len(group) > 1]


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        for name, content in [
            ("a.txt", b"hello"),
            ("b.txt", b"hello"),      # duplicate of a
            ("c.txt", b"world"),
            ("d.txt", b"world"),      # duplicate of c
            ("e.txt", b"unique"),
        ]:
            with open(os.path.join(tmpdir, name), "wb") as f:
                f.write(content)

        dupes = find_duplicates_solution(tmpdir)
        assert len(dupes) == 2  # Two groups of duplicates
        print(f"Found {len(dupes)} duplicate groups: {dupes}")
        print("All tests passed! ✅")

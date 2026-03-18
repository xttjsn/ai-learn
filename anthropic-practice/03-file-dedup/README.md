# Coding Q2 (variant): File Deduplication

## Problem
Given a set of files, find and remove duplicates efficiently.

## Progressive Levels

### Level 1: Basic Hash-based Dedup
- Hash file contents, group by hash
- MD5/SHA256 for content fingerprinting

### Level 2: Optimized Dedup
- Size-first filtering (skip files of unique sizes)
- Partial hashing (first N bytes, then full hash)
- Streaming hash for large files

### Level 3: System Design Discussion
- IO bound vs CPU bound analysis
- Concurrent file hashing
- Distributed dedup across machines

## Files
- `level1_basic.py` — Simple hash-based dedup
- `level2_optimized.py` — Size + partial hash optimization
- `level3_concurrent.py` — Threaded version + discussion

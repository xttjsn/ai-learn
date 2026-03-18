# Cerebras Interview Quiz Set — Sr. Inference ML Runtime Engineer

*Progressive difficulty: Level 1 (Warmup) → Level 5 (Interview Ready)*
*Covers: DS&A, OS/Systems, ML Inference, C++/Python, System Design*

---

## How to Use

- Work through each level sequentially
- Try to answer **without looking anything up** first
- Time yourself: easy = 5 min, medium = 15 min, hard = 25 min
- Mark ✅ or ❌ and revisit missed ones
- For coding: implement in Python or C++

---

## LEVEL 1 — Warmup (Foundations)

### DS&A (Easy LC)

**Q1.1** Implement Flood Fill (LC 733)
Given a 2D grid, a starting pixel (sr, sc), and a new color, flood fill from that pixel. Change all connected pixels of the same original color.
```
Input: image = [[1,1,1],[1,1,0],[1,0,1]], sr=1, sc=1, color=2
Output: [[2,2,2],[2,2,0],[2,0,1]]
```
- [ ] Solved

**Q1.2** Best Time to Buy and Sell Stock (LC 121)
Find max profit from one buy + one sell.
```
Input: [7,1,5,3,6,4] → Output: 5
```
- [ ] Solved

**Q1.3** Linked List Cycle (LC 141)
Detect if a linked list has a cycle. Do it in O(1) space.
- [ ] Solved

**Q1.4** Roman to Integer (LC 13)
Convert "MCMXCIV" → 1994
- [ ] Solved

**Q1.5** Balance Parentheses *(actually asked at Cerebras)*
Given a string of parentheses, return the minimum number of additions to make it valid.
```
Input: "())(" → Output: 2
```
- [ ] Solved

### OS Fundamentals

**Q1.6** What is the difference between a process and a thread? When would you use each?
- [ ] Answered

**Q1.7** Explain the difference between mutex and semaphore. Give a use case for each.
- [ ] Answered

**Q1.8** What is virtual memory? Why do we need it?
- [ ] Answered

**Q1.9** Name 4 CPU scheduling algorithms and briefly describe each.
- [ ] Answered

**Q1.10** What is a page fault? What happens when one occurs?
- [ ] Answered

### C++ Basics

**Q1.11** What is the difference between `stack` and `heap` memory allocation in C++? When is each used?
- [ ] Answered

**Q1.12** What does `std::move` do? When would you use it?
- [ ] Answered

---

## LEVEL 2 — Building Blocks (Medium)

### DS&A (Medium LC)

**Q2.1** Word Search (LC 79) *(high frequency at Cerebras)*
Given a 2D board and a word, find if the word exists by moving to adjacent cells (no reuse).
```
board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
word = "ABCCED" → true
```
- [ ] Solved

**Q2.2** Continuous Subarray Sum (LC 523)
Given an array of non-negative integers and a target k, find if there's a contiguous subarray of length ≥ 2 whose sum is a multiple of k.
```
nums = [23,2,4,6,7], k = 6 → true (because [2,4] sums to 6)
```
- [ ] Solved

**Q2.3** 2D Prefix Sum *(actually asked at Cerebras — talk through, no code)*
Given a 2D matrix, precompute prefix sums so you can answer "sum of submatrix from (r1,c1) to (r2,c2)" in O(1).
- Explain how to build the prefix sum matrix
- Explain the inclusion-exclusion formula for querying
- What's the time and space complexity?
- [ ] Solved

**Q2.4** Subsets II (LC 90)
Given an array that may contain duplicates, return all possible subsets (no duplicate subsets).
```
Input: [1,2,2] → Output: [[],[1],[1,2],[1,2,2],[2],[2,2]]
```
- [ ] Solved

**Q2.5** Next Permutation (LC 31)
Implement next lexicographic permutation in-place.
```
[1,2,3] → [1,3,2]
[3,2,1] → [1,2,3]
```
- [ ] Solved

**Q2.6** All Nodes Distance K in Binary Tree (LC 863)
Given a binary tree, a target node, and K, return all nodes at distance K from target.
- [ ] Solved

### OS / Systems

**Q2.7** Explain shared memory IPC. How do two processes share data? What synchronization is needed?
- [ ] Answered

**Q2.8** What is a deadlock? What are the 4 necessary conditions? How do you prevent it?
- [ ] Answered

**Q2.9** Explain the difference between DMA and memory-mapped I/O. Why does DMA matter for high-performance systems?
- [ ] Answered

**Q2.10** What is a TLB? What happens on a TLB miss? Why is TLB important for performance?
- [ ] Answered

### C++ Deep

**Q2.11** *(Actually asked at Cerebras)* Why is `std::unordered_map` rarely the best hash map? How are `tbb::concurrent_hash_map` and `absl::flat_hash_map` better? Under what circumstances?

Consider:
- Memory layout (node-based vs flat)
- Cache performance
- Load factor behavior
- Concurrency
- [ ] Answered

**Q2.12** What is the difference between `std::shared_ptr` and `std::unique_ptr`? What's the overhead of `shared_ptr`? When would you avoid it in a hot path?
- [ ] Answered

---

## LEVEL 3 — ML & Inference Knowledge

### ML Fundamentals

**Q3.1** Explain the Transformer architecture end-to-end. Include:
- Input embedding + positional encoding
- Multi-head self-attention (with the Q, K, V math)
- Feed-forward network
- Layer norm placement (pre-norm vs post-norm)
- [ ] Answered

**Q3.2** What is the KV cache in autoregressive LLM inference?
- Why do we need it?
- What does it store?
- How does memory usage scale with sequence length?
- What is the memory formula for KV cache size?
- [ ] Answered

**Q3.3** Explain the difference between the **prefill** and **decode** phases in LLM inference.
- Which is compute-bound? Which is memory-bandwidth-bound?
- How does batching affect each?
- [ ] Answered

**Q3.4** What is **continuous batching** (aka iteration-level batching)?
- How is it different from static batching?
- Why does it improve throughput?
- What are the implementation challenges?
- [ ] Answered

**Q3.5** Explain **Mixture of Experts (MoE)**:
- How does the router/gating work?
- What is top-k selection?
- What is the load balancing problem and how is it addressed?
- Why is MoE particularly suited for Cerebras hardware?
- [ ] Answered

### Inference Systems

**Q3.6** What is **PagedAttention** (from vLLM)?
- What problem does it solve?
- How does it manage KV cache memory?
- How is it analogous to virtual memory paging?
- [ ] Answered

**Q3.7** Explain **speculative decoding**:
- How does draft model + verify work?
- When does it help? When does it not?
- What's the theoretical speedup?
- [ ] Answered

**Q3.8** What are the main **quantization** approaches for inference?
- PTQ vs QAT
- INT8, FP8, INT4 — tradeoffs
- How does quantization affect memory bandwidth vs compute?
- [ ] Answered

**Q3.9** You're serving an LLM and users complain about high latency. Walk through how you'd diagnose and fix it:
- What metrics would you look at?
- How do you identify the bottleneck (compute vs memory vs I/O)?
- What optimizations would you try?
- [ ] Answered

---

## LEVEL 4 — Cerebras-Specific & System Design

### Cerebras Architecture

**Q4.1** Explain the **weight streaming** execution model:
- How do weights flow from MemoryX → WSE?
- Why does this eliminate the memory bandwidth bottleneck?
- How does it compare to GPU inference (weights in HBM)?
- When is a model "too big" even for weight streaming?
- [ ] Answered

**Q4.2** Why is SRAM better than HBM/DRAM for inference workloads?
- Compare: bandwidth, latency, cost per bit, density
- Why did Cerebras choose 44GB SRAM over HBM?
- What are the tradeoffs?
- [ ] Answered

**Q4.3** Explain the **SwarmX** architecture:
- How do multiple CS-3 systems coordinate for inference?
- How is tensor parallelism implemented across wafer-scale engines?
- What are the communication challenges?
- [ ] Answered

**Q4.4** Cerebras claims 15x faster inference than GPUs. Where does this speedup come from?
- Break it down: what contributes to the speedup?
- Are there workloads where GPUs would still win?
- [ ] Answered

### System Design (Cerebras-Style)

**Q4.5** Design an **inference serving system** for Cerebras hardware:
- How do you handle incoming API requests?
- Request queuing and scheduling
- Batching strategy
- Model loading and weight streaming
- How do you maintain the OpenAI-compatible API contract?
- How do you handle streaming responses (SSE)?
- How do you scale across multiple CS-3 systems?
- [ ] Designed

**Q4.6** Design a **model onboarding pipeline**:
- A customer has a fine-tuned Llama-70B in HuggingFace format
- How do you translate it to run on Cerebras hardware?
- Checkpoint conversion
- Graph compilation
- Weight placement in MemoryX
- Validation and performance testing
- [ ] Designed

**Q4.7** Design a **multi-tenant inference platform**:
- Multiple customers sharing Cerebras hardware
- Isolation, fairness, priority scheduling
- How do you handle different models for different customers?
- SLA management (latency guarantees)
- Cost attribution
- [ ] Designed

### Open-Ended Graph/Design *(Cerebras asks these)*

**Q4.8** Design an **efficient graph traversal** for a social network with 1B nodes:
- How would you find all users within 3 hops of a given user?
- Memory constraints — can't load the full graph
- How would you parallelize this on 900K cores?
- [ ] Designed

---

## LEVEL 5 — Mock Interview (Timed)

*Simulate a real Cerebras interview round. Set a 45-minute timer.*

### Mock Round 1: Coding + Knowledge (45 min)

**Part A — Knowledge (15 min)**
1. Walk me through your most impactful systems project. What was the hardest technical decision?
2. Explain how LLM inference works at a high level. Where are the bottlenecks?
3. What's the difference between latency and throughput optimization? Give an example where optimizing for one hurts the other.

**Part B — Coding (20 min)**
Implement: **Split Array Largest Sum** (LC 410) *(the only Hard in Cerebras's problem set)*
Given an array and k, split it into k non-empty contiguous subarrays such that the largest sum among them is minimized. Return that minimized largest sum.
```
Input: nums = [7,2,5,10,8], k = 2
Output: 18 (split: [7,2,5] and [10,8])
```
Hint: Binary search on the answer.

**Part C — Discussion (10 min)**
4. Why Cerebras? Why this role specifically?
5. Ask 2-3 thoughtful questions about the team/tech.

### Mock Round 2: Technical Deep Dive (45 min)

**Scenario:** You're tasked with adding support for a new model architecture (e.g., DeepSeek-V3, which uses MoE + multi-head latent attention) to the Cerebras inference runtime.

Walk through:
1. What do you need to understand about the model architecture first?
2. How would you approach the model translation (PyTorch → Cerebras runtime)?
3. What performance challenges do you anticipate with MoE on this hardware?
4. How would you validate correctness? Performance?
5. How would you handle the KV cache differently for multi-head latent attention?

### Mock Round 3: OS & Systems (45 min)

**Part A — Knowledge (15 min)**
1. Explain process scheduling in a system with 900K cores. How is this different from a traditional OS?
2. How does shared memory work? What happens when two threads write to the same cache line?
3. What is false sharing? How do you fix it?

**Part B — Coding (20 min)**
Implement a **thread-safe bounded queue** (producer-consumer pattern) in C++ or Python:
- `put(item)` — blocks if full
- `get()` → item — blocks if empty
- Support multiple producers and consumers

**Part C — Discussion (10 min)**
4. Tell me about a time you debugged a tricky performance issue. What tools did you use?
5. How do you approach working with a compiler team when the runtime needs new capabilities?

---

## Score Tracking

| Level | Total Qs | Completed | Score |
|-------|----------|-----------|-------|
| 1 — Warmup | 12 | /12 | |
| 2 — Building Blocks | 12 | /12 | |
| 3 — ML & Inference | 9 | /9 | |
| 4 — Cerebras & Design | 8 | /8 | |
| 5 — Mock Interviews | 3 rounds | /3 | |
| **Total** | **44** | | |

---

## Quick Reference: What Cerebras Actually Asked

| Question | Role | Source |
|----------|------|--------|
| Balance parentheses (LC easy) | Runtime Intern | Glassdoor |
| Open-ended graph search | AI Intern | Glassdoor |
| 2D prefix sum (talk-through) | Kernel Engineer | Glassdoor |
| `std::unordered_map` vs `tbb`/`absl` | SWE | Glassdoor |
| Scheduling & shared memory | Runtime Intern | Glassdoor |
| Computer architecture + compiler basics | Kernel Engineer | Glassdoor |
| Transformers & optimization | MTS | Glassdoor |
| Flood Fill (LC 733) | Various | InterviewSolver |
| Word Search (LC 79) | Various | InterviewSolver |
| Split Array Largest Sum (LC 410) | Various | InterviewSolver |

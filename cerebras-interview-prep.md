# Cerebras Interview Prep — Sr. Inference ML Runtime Engineer

*Compiled: March 15, 2026 | Role: Sr. Inference ML Runtime Engineer | Team: AI Cloud*
*Location: Sunnyvale CA or Toronto Canada*

---

## Table of Contents

1. [The Role](#the-role)
2. [Interview Process](#interview-process)
3. [Interview Themes & What to Expect](#interview-themes)
4. [Coding Problems (LeetCode)](#coding-problems)
5. [System Design / Technical Deep Dives](#system-design)
6. [Cerebras Architecture & Inference Stack](#cerebras-architecture)
7. [1Point3Acres 面经](#1point3acres)
8. [Prep Strategy](#prep-strategy)
9. [Resources](#resources)

---

## The Role

**Sr. Inference ML Runtime Engineer** sits in the **AI Cloud** team — the group building Cerebras's inference-as-a-service platform. From the job board listing:

> "Lead the Inference ML team in developing tools and APIs for large-scale ML applications, enhancing performance and usability, while collaborating across engineering teams."

### Official Job Description (from Greenhouse/Simplify)

> The Inference ML Engineering team at Cerebras Systems is dedicated to enabling our fast generative inference solution through **simple APIs powered by a distributed runtime that runs on large clusters of our own hardware**. Our mission is to empower enterprises, developers, and researchers to unlock the full potential of our platform, leveraging its performance, scalability, and flexibility. The team works closely with cross-functional groups, including **compiler developers, cluster orchestrators, ML scientists, cloud architects, and product teams**, to deliver high-impact solutions that redefine the boundaries of ML performance and usability.

> As a Senior Software Engineer on the Inference ML Engineering team, you will play a key role in **designing and implementing APIs, ML features, and tools** that enable running state-of-the-art generative AI models on our custom hardware. You will **architect solutions that enable seamless model translation and execution**, ensuring high throughput and low latency, while maintaining ease of use. Your responsibilities will include **leading technical initiatives**, collaborating with other engineering teams to enhance the developer experience, **enabling key ML features at scale**, maintaining our speed advantage, achieving high throughput, and supporting a wide range of ML workloads.

*Note: Responsibilities and Skills/Qualifications sections were JS-rendered and not extractable, but the role description above is very detailed.*

### What This Role Involves (Derived from JD + Context)

- **Distributed runtime development** — the runtime that runs on large clusters of Cerebras hardware
- **API design & implementation** — the inference APIs (OpenAI-compatible, streaming, function calling)
- **Model translation & execution** — getting models to run seamlessly on WSE
- **Performance optimization** — high throughput, low latency, maintaining speed advantage
- **ML feature enablement** — supporting new ML capabilities at scale (reasoning, agents, code-gen)
- **Cross-functional collaboration** — compiler, cluster orchestration, ML scientists, cloud, product
- **Developer experience** — making the platform easy to use

### Adjacent Roles on the Same Team (AI Cloud)

- Staff DevRel Engineer - AI Inference
- Staff/Sr. Deployment Engineer, AI Inference
- Principal Engineer, AI Inference Reliability
- Staff SW Engineer, Observability
- Sr. Staff/Principal SW Security Engineer, AI Inference

### Related Roles (Software)

- Senior Runtime Engineer
- Kernel Engineer (writes compute kernels for WSE)
- Performance Engineer - Inference
- LLM Inference Performance & Evals Engineer
- LLVM Compiler Engineer
- Inference Frontend

---

## Interview Process

From [Cerebras Interviewing Guide](https://coda.io/@cerebras-careers/cerebras-interviewing-guide/interviewing-cerebras-2):

### Steps

1. **Exploratory** (45 min)
   - Background discussion + technical conversation
   - May include a coding section depending on role
   - With a member of the engineering team

2. **Deep Dive Interviews** (4 × 45 min)
   - **Coding** (1 round) — HackerRank, language of your choice
   - **Technical Deep Dive** (2 rounds) — domain-specific discussions related to inference/runtime
   - **Hiring Manager** (1 round) — Teams video, communication, problem-solving style, team fit

3. **Debrief & Follow Up**
   - Feedback within 24-48h
   - May schedule follow-up discussions
   - Every offer includes a **CEO/founders call**

### Platform & Logistics

- **Coding:** HackerRank (no whiteboard)
- **Video:** Microsoft Teams
- **Tips:** Test connection beforehand, camera on, blur background available in Teams
- **Culture:** They explicitly value curiosity — ask questions about their tech

---

## Interview Themes

### From Glassdoor (30 reviews, updated March 2026)

**Overall stats:**
- Difficulty: **2.85/5** (moderate)
- Experience: **46.1% positive**, 30% negative
- Average hiring time: **13 days** (but SWE reported ~2 months from first interview to offer)
- How people got in: **50% applied online**, 28% recruiter

**SWE Interview (Mar 2026, accepted offer):**
> "Four 1-hr rounds. Each is **half resume and knowledge questions, and half coding**. Half of the coding questions were **DS&A** and the other half are **ML and OS related**. Took about two months from first interview to offer."
> - Question: "Why is `std::unordered_map` rarely the best hash map? How are `tbb` or `absl` flavors better, and under what circumstances?"

**Kernel Engineer (Mar 2026, rejected):**
> "One interview about **computer architecture**, some **compiler basics** and a couple of Leetcode style questions which seemed out of place. The interviewer seemed nice enough and gave hints. There was **no coding, I just had to talk it out**."
> - Question: "**Prefix sum but in a 2D array**"

**AI Intern (Oct 2025):**
> "2×30min interviews, **leetcode easy and a design type question (graph)**. No behavioral parts or even going over my resume."
> - Question: "**Open ended graph search question**"

**Runtime Engineer Intern (Oct 2025):**
> "45 minutes, pretty easy questions, easy leetcode question and then a question about **scheduling and shared memory resource** — things covered in a university course. Brush up on **operating system concepts**."
> - Question: "**Balance the number of parenthesis**" (LC easy)

**MTS (Nov 2025, Bengaluru):**
> "Mostly got asked questions from my CV and had to **code in front of the person online**. Got asked questions regarding **transformers and optimisation**. 2 interviews of an hour each."

### From Blind (Warning Signs to Be Aware Of)

Several Blind posts report **frustrating interview experiences**:
- Verbal offers followed by additional interviews during negotiation, then rejection
- "Immature management" — some interviewers described as uncommunicative/rude
- One poster: "They say you're the best during the interview, will make you wait, then find another candidate and reject with vanilla email"
- System design interviewer described as "highly uncommunicative and seemed rude"

**However:** These seem to be outlier experiences, and 46% of Glassdoor reviews are positive. The recent SWE hire (Mar 2026) accepted the offer, so the process works for many.

**Takeaway:** Be prepared for a potentially long process (~2 months). If you get a verbal offer, try to get things in writing quickly.

### Key Themes for Inference ML Runtime

Based on role + interview reports:

1. **OS Fundamentals** — scheduling, shared memory, concurrency, resource management
2. **Computer Architecture** — memory hierarchy, cache, SRAM vs DRAM, bandwidth
3. **ML Inference Systems** — serving, batching, KV cache, attention optimization
4. **Runtime Systems** — execution models, memory management, scheduling on custom hardware
5. **Practical Coding** — use-case oriented, not just LeetCode puzzles
6. **Compiler Basics** — IR, optimization passes, graph compilation (relevant to their stack)
7. **Python/C++ proficiency** — runtime work likely involves both

---

## Coding Problems

Source: InterviewSolver (23 reported problems, ranked by frequency).

### Difficulty Distribution

- Easy: 6 (26%)
- Medium: 16 (70%)
- Hard: 1 (4%)

### Top Topics

1. **Array** — most frequent
2. **String**
3. **Hash Table**
4. **Two Pointers**
5. **Depth-First Search**

### Easy (6)

| # | Problem | LC# | Topics |
|---|---------|-----|--------|
| 1 | **Flood Fill** | 733 | Array, DFS, BFS, Matrix |
| 2 | **Roman to Integer** | 13 | Hash Table, Math, String |
| 3 | **High Five** | 1086 | Array, Hash Table, Sorting |
| 4 | **Best Time to Buy and Sell Stock** | 121 | Array, DP |
| 5 | **Linked List Cycle** | 141 | Hash Table, Linked List, Two Pointers |
| 6 | **Max Nesting Depth of Parentheses** | 1614 | String, Stack |

### Medium (16)

| # | Problem | LC# | Topics |
|---|---------|-----|--------|
| 1 | **Make String a Subsequence Using Cyclic Increments** | 2825 | Two Pointers, String |
| 2 | **Word Search** | 79 | Array, String, Backtracking, Matrix |
| 3 | **Continuous Subarray Sum** | 523 | Array, Hash Table, Math, Prefix Sum |
| 4 | **Subsets II** | 90 | Array, Backtracking, Bit Manipulation |
| 5 | **Reverse Words in a String** | 151 | Two Pointers, String |
| 6 | **Analyze User Website Visit Pattern** | 1152 | Array, Hash Table, Sorting |
| 7 | **Maximize Amount After Two Days of Conversions** | 3387 | Array, String, DFS, BFS, Graph |
| 8 | **Reverse Integer** | 7 | Math |
| 9 | **Remove All Adjacent Duplicates in String II** | 1209 | String, Stack |
| 10 | **Remove All Occurrences of a Substring** | 1910 | String, Stack, Simulation |
| 11 | **All Nodes Distance K in Binary Tree** | 863 | Hash Table, Tree, DFS, BFS |
| 12 | **Zigzag Conversion** | 6 | String |
| 13 | **Adding Spaces to a String** | 2109 | Array, Two Pointers, Simulation |
| 14 | **Find Score of an Array After Marking All Elements** | 2593 | Array, Hash Table, Sorting, Heap |
| 15 | **Next Permutation** | 31 | Array, Two Pointers |
| 16 | **Find Leaves of Binary Tree** | 366 | Tree, DFS |

### Hard (1)

| # | Problem | LC# | Topics |
|---|---------|-----|--------|
| 1 | **Split Array Largest Sum** | 410 | Array, Binary Search, DP, Greedy, Prefix Sum |

---

## System Design / Technical Deep Dives

Cerebras does NOT do traditional "design Twitter" system design. Their deep dives are **domain-specific**. For a Sr. Inference ML Runtime Engineer, expect:

### Inference Serving Architecture

- **Design an inference serving system** — request routing, batching (continuous/dynamic), queue management
- **How does Cerebras achieve 3,000+ tokens/sec?** — weight streaming eliminates memory bottleneck
- **Continuous batching vs static batching** — why continuous batching matters for LLM serving
- **Multi-model serving** — how to schedule different models across hardware
- **Autoscaling** — scaling inference capacity based on demand
- **API design** — OpenAI-compatible endpoints, streaming responses, function calling

### Weight Streaming & Runtime

- **Weight streaming execution model** — weights stored in MemoryX, streamed layer-by-layer to WSE
  - Why this eliminates the memory bandwidth bottleneck
  - How it differs from GPU inference (weights in HBM, compute-bound vs memory-bound)
  - Layer-by-layer execution: only one layer's weights need to be on-chip at a time
- **Runtime scheduling** — how to schedule operations across 900K cores
- **Memory management** — 44GB on-chip SRAM, no cache hierarchy, explicit placement
- **Pipeline parallelism** — overlapping weight loading with computation

### KV Cache & Attention

- **KV cache management** for long-context inference
- **How KV cache works differently without HBM** — all in SRAM on WSE
- **PagedAttention** and memory-efficient attention variants
- **Speculative decoding** — draft model + verify pattern

### Performance Optimization

- **Profiling and bottleneck analysis** — identifying compute vs memory vs I/O bottlenecks
- **Kernel optimization** — fused operations, memory access patterns
- **Quantization** — INT8/FP8 inference, mixed precision
- **Latency breakdown** — prefill vs decode phase optimization

### Distributed Inference

- **Multi-WSE inference** (SwarmX architecture) — how multiple wafer-scale engines coordinate
- **MemoryX** — external weight storage, streaming weights to compute
- **Tensor parallelism vs pipeline parallelism** on custom hardware
- **Fault tolerance** — handling hardware failures in wafer-scale systems

### OS / Low-Level (Expect These!)

- **Process scheduling algorithms** — FIFO, SJF, Round Robin, Priority, MLFQ
- **Shared memory** — IPC, memory-mapped files, semaphores
- **Concurrency** — mutexes, condition variables, lock-free data structures
- **Virtual memory** — paging, TLB, page faults
- **DMA and memory-mapped I/O**

---

## Cerebras Architecture & Inference Stack

See also: `06_cerebras_architecture/cerebras_wse_deep_dive.md`

### Hardware

| Component | Spec |
|-----------|------|
| **WSE-3** | 3rd gen Wafer-Scale Engine, entire 300mm silicon wafer |
| **AI Cores** | 900,000 cores |
| **On-chip SRAM** | 44 GB (no HBM/DRAM!) |
| **Memory BW** | 21 PB/s |
| **Interconnect** | 2D mesh fabric, all on-chip |
| **Process** | 5nm TSMC |

### System Architecture

```
┌─────────────────────────────────────────┐
│              CS-3 System                 │
│  ┌───────────────────────────────────┐  │
│  │         WSE-3 (Compute)           │  │
│  │   900K cores, 44GB SRAM           │  │
│  │   21 PB/s memory bandwidth        │  │
│  └──────────────┬────────────────────┘  │
│                 │ weight streaming       │
│  ┌──────────────▼────────────────────┐  │
│  │       MemoryX (Weight Store)      │  │
│  │   External memory for model       │  │
│  │   weights, streams to WSE         │  │
│  └───────────────────────────────────┘  │
│                                          │
│  ┌───────────────────────────────────┐  │
│  │       SwarmX (Interconnect)       │  │
│  │   Connects multiple CS-3s for     │  │
│  │   distributed inference           │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### Why Cerebras Inference Is Fast

**The core insight:** LLM inference on GPUs is **memory-bandwidth-bound**, not compute-bound.

- GPU: weights sit in HBM (80GB, 3.35 TB/s) → must fetch through HBM for every token
- Cerebras: 44GB SRAM at 21 PB/s → **6,000x more bandwidth** than GPU HBM
- **Weight streaming:** for models larger than 44GB, weights stream from MemoryX layer-by-layer
  - Only need one layer's weights on-chip at a time
  - MemoryX → WSE streaming is pipelined with compute

### Inference Product (as of March 2026)

- **API endpoint:** inference.cerebras.ai
- **Supported models:** GPT-OSS-120B (3,000 tok/s), Llama 3.1-8B, Llama 3.3-70B, Qwen 3-32B, Qwen 3-235B, ZAI GLM-4.7
- **Claim:** Up to 15x faster than GPU inference
- **Use cases:** Real-time agents, instant code generation, sub-1s reasoning
- **Scale:** 6 new datacenters, targeting 40M+ tokens/sec total capacity
- **OpenAI-compatible API**

### Recent Major Developments (Know These for "Why Cerebras?")

1. **OpenAI Partnership** — Multi-year, **$10B deal** for 750MW of inference compute
   - GPT-5.3-Codex-Spark launched on Cerebras hardware (first milestone)
   - OpenAI using Cerebras for fast inference workloads
2. **Amazon AWS Partnership** — Cerebras chips in AWS data centers, linked to Amazon Trainium3
   - Split inference: Trainium3 handles prefill, Cerebras handles decode
3. **G42 India Deployment** — **8 exaFLOPS** AI supercomputer (~64 WSE-3 accelerators)
4. **Valuation:** $23.1B (raised $1B recently), Series H, $2.8B total funding
5. **Company size:** 501-1,000 employees
6. **IPO:** Filed previously, still being explored

### Key Differentiators to Articulate

1. **No memory wall** — SRAM bandwidth eliminates the GPU HBM bottleneck
2. **Weight streaming** — enables serving models larger than on-chip memory
3. **Deterministic execution** — no cache misses, predictable latency
4. **Single-chip simplicity** — no inter-GPU communication overhead for models that fit
5. **Wafer-scale integration** — 900K cores communicate via on-chip fabric, not PCIe/NVLink

---

## 1Point3Acres 面经 (一亩三分地)

Source: [1P3A Cerebras Interview](https://www.1point3acres.com/interview/company/Cerebras)

Known posts (2 reported):

### Kernel Engineer — Fulltime Video Interview (Oct-Dec 2024)
<!-- TODO: Paste content from 1P3A post here — login required -->
- Role: Software Engineer (Kernel Engineer)
- Type: Fulltime
- Stage: Video Interview

### Kernel Engineer — Intern Phone Screen (Apr-Jun 2023)
<!-- TODO: Paste content from 1P3A post here — login required -->
- Role: Software Engineer (Kernel Engineer)
- Type: Intern
- Stage: Technical Phone Screen
- 2 replies, 1 upvote

> **Note:** 1P3A requires login to view full content. Log in at 1point3acres.com and search "Cerebras" in the interview section to read the full 面经. Update this section with details.

### What "Kernel Engineer" vs "Runtime Engineer" at Cerebras

| | Kernel Engineer | Runtime Engineer |
|---|---|---|
| **Focus** | Compute kernels (matmul, attention, etc.) | Execution runtime, scheduling, serving |
| **Analogy** | Writing CUDA kernels | Building the CUDA runtime / TensorRT |
| **Language** | Likely C/C++, Cerebras SDK | C++, Python, systems programming |
| **Level** | Low-level compute optimization | Systems-level orchestration |
| **Relevance** | Related but different from your target role | **Your target role** |

---

## Prep Strategy

### Priority 1: Inference Systems Knowledge (Most Important for This Role)

- [ ] Understand weight streaming deeply — read `06_cerebras_architecture/cerebras_wse_deep_dive.md`
- [ ] Review LLM inference mechanics: prefill vs decode, KV cache, continuous batching
- [ ] Study `07_inference_optimization/` and `inference-systems-cheatsheet.md`
- [ ] Know the Cerebras inference product (models, speeds, architecture)
- [ ] Be ready to discuss: "How would you design the runtime for weight streaming inference?"
- [ ] Understand PagedAttention, speculative decoding, quantization for inference

### Priority 2: OS & Systems Fundamentals (Confirmed by Glassdoor)

- [ ] Scheduling algorithms (FIFO, SJF, RR, Priority, MLFQ)
- [ ] Shared memory & IPC (semaphores, mutexes, condition variables)
- [ ] Virtual memory (paging, TLB, page faults, memory mapping)
- [ ] Concurrency (deadlock, livelock, lock-free structures)
- [ ] DMA, memory-mapped I/O

### Priority 3: LeetCode (Moderate — Not the Main Focus)

- [ ] Focus on mediums — arrays, strings, hash tables, two pointers
- [ ] **Top priority problems:** Flood Fill (733), Word Search (79), Continuous Subarray Sum (523)
- [ ] Practice implementing practical use cases, not just algorithm puzzles
- [ ] Be comfortable with Python and C++ for coding rounds
- [ ] Time target: solve mediums in 20-25 min

### Priority 4: Computer Architecture & Compiler Basics

- [ ] Cache hierarchy (L1/L2/L3, cache lines, associativity)
- [ ] SRAM vs DRAM — why SRAM is faster but more expensive (this is Cerebras's core bet)
- [ ] Memory models, pipeline hazards, SIMD/SIMT
- [ ] Compiler basics: AST → IR → optimization → codegen
- [ ] Graph compilation for ML models

### Behavioral / Hiring Manager Round

- [ ] "Why Cerebras?" — genuinely exciting hardware, solving the memory wall problem
- [ ] "Why this role?" — intersection of ML systems + runtime engineering + novel hardware
- [ ] "Tell me about a time you optimized a system for performance"
- [ ] "How do you approach debugging performance issues in production?"
- [ ] Prepare 2-3 questions about the team, tech stack, and roadmap

---

## Compensation (Levels.fyi)

Source: [Levels.fyi — Cerebras SWE](https://www.levels.fyi/companies/cerebras-systems/salaries/software-engineer)

| Level | Total Comp (US) | Base | Stock |
|-------|----------------|------|-------|
| L2 (Entry) | $194K | $165K | $30K |
| L4 | $201K | $176K | $25K |
| L5 (Senior) | $371K | $226K | $145K |

- 4-year vesting: 25% per year (standard)
- Pre-IPO equity (Cerebras has been exploring IPO)
- "Sr." title likely maps to L5 range

---

## Cerebras Open Source & SDK (Know This!)

Familiarize yourself with their public repos — you may be working on or adjacent to these:

### [cerebras-cloud-sdk-python](https://github.com/Cerebras/cerebras-cloud-sdk-python)
- OpenAI-compatible Python SDK for Cerebras Inference API
- Sync + async clients (httpx, optional aiohttp)
- Streaming via SSE, typed responses (Pydantic)
- TCP warming on construction for lower TTFT
- API endpoint: `https://api.cerebras.ai/v1`
- Models: `gpt-oss-120b`, `llama3.1-8b`, `llama3.3-70b`, `qwen3-32b`, etc.

### [modelzoo](https://github.com/Cerebras/modelzoo)
- Reference implementations for training on Cerebras hardware
- Models: Llama, Mixtral, DINOv2, LLaVA, GPT-2/3, T5, etc.
- Includes checkpoint converters (Cerebras ↔ HuggingFace)
- CLI for data preprocessing, training, validation
- Apache 2.0 license

### Inference API Features
- Chat completions + text completions
- Streaming (SSE)
- Function calling support
- OpenAI drop-in compatible

> **Interview tip:** Being able to discuss the SDK design, API patterns, and how inference serving works behind the scenes will be very relevant for this role.

---

## Key Research Papers (arXiv)

Must-read papers for understanding the technology you'd be working with:

### Directly Relevant to Inference
1. **"WaferLLM: Large Language Model Inference at Wafer Scale"** (Feb 2025)
   - Inference optimization specifically for wafer-scale hardware
   - **READ THIS** — most directly relevant to your role

2. **"Benchmarking the Performance of Large Language Models on the Cerebras Wafer Scale Engine"** (Aug 2024)
   - Performance evaluation of LLMs on WSE
   - Good for understanding performance characteristics

3. **"Cerebras-GPT: Open Compute-Optimal Language Models Trained on the Cerebras Wafer-Scale Cluster"** (Apr 2023)
   - Official Cerebras paper on compute-optimal training
   - Shows their thinking on scaling laws

### Architecture & Compiler
4. **"SPADA: A Spatial Dataflow Architecture Programming Language"** (Nov 2025)
   - Programming model for spatial dataflow architectures like WSE
   - Relevant for understanding the execution model

5. **"A System Level Compiler for Massively-Parallel, Spatial, Dataflow Architectures"** (Jun 2025)
   - MACH compiler for wafer-scale architectures
   - Understanding compiler → runtime interface

6. **"An MLIR Lowering Pipeline for Stencils at Wafer-Scale"** (Jan 2026)
   - MLIR-based compilation targeting WSE

7. **"A Comparison of the Cerebras Wafer-Scale Integration Technology with Nvidia GPU-based Systems for AI"** (Mar 2025)
   - Direct WSE vs GPU comparison — good for "why Cerebras" answers

### Performance & Optimization
8. **"K2-Think: A Parameter-Efficient Reasoning System"** (Sep 2025)
   - Uses speculative decoding + inference-optimized hardware (Cerebras)
   - Relevant to reasoning model serving

9. **"Near-Optimal Wafer-Scale Reduce"** (Apr 2024)
   - Collective operations on WSE — important for distributed inference

---

## Resources

### Official
- [Cerebras Interviewing Guide](https://coda.io/@cerebras-careers/cerebras-interviewing-guide/interviewing-cerebras-2)
- [Cerebras Inference Product Page](https://cerebras.ai/inference)
- [Cerebras Inference API Docs](https://inference-docs.cerebras.ai)
- [Job Board (Greenhouse)](https://boards.greenhouse.io/embed/job_board?for=cerebrassystems)
- [Cerebras Blog](https://cerebras.ai/blog)

### GitHub
- [cerebras-cloud-sdk-python](https://github.com/Cerebras/cerebras-cloud-sdk-python) — Inference SDK
- [modelzoo](https://github.com/Cerebras/modelzoo) — Model implementations & training tools

### Interview Data
- [Glassdoor — 28 Reviews](https://www.glassdoor.com/Interview/Cerebras-CA-Interview-Questions-E1821335.htm)
- [Blind — Cerebras Discussions](https://www.teamblind.com/company/Cerebras-Systems/posts/cerebras-systems-interview)
- [InterviewSolver — 23 LC Problems](https://interviewsolver.com/interview-questions/cerebras-systems)
- [1P3A — Cerebras Interviews](https://www.1point3acres.com/interview/company/Cerebras)
- [Levels.fyi — Compensation](https://www.levels.fyi/companies/cerebras-systems/salaries/software-engineer)

### Research Papers (arXiv)
- [WaferLLM: LLM Inference at Wafer Scale](https://arxiv.org/search/?query=WaferLLM&searchtype=all) ⭐
- [Cerebras-GPT: Compute-Optimal Language Models](https://arxiv.org/abs/2304.03208)
- [SPADA: Spatial Dataflow Architecture Programming](https://arxiv.org/search/?query=SPADA+spatial+dataflow&searchtype=all)
- [Full Cerebras paper list](https://arxiv.org/search/?query=cerebras+wafer+scale&searchtype=all&order=-announced_date_first)

### Local Study Materials
- `06_cerebras_architecture/cerebras_wse_deep_dive.md` — WSE hardware deep dive
- `07_inference_optimization/` — inference optimization techniques
- `inference-systems-cheatsheet.md` — quick reference
- `03_serving_and_batching/` — serving system fundamentals
- `09_kv_cache_and_attention_variants/` — KV cache mechanics
- `10_quantization/` — quantization for inference

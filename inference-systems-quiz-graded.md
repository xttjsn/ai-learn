# LLM Inference Systems — Quiz (GRADED)

**Total: 52 / 100**

| Section | Earned | Possible |
|---|---|---|
| 1: Two Phases | 5 | 10 |
| 2: Memory Math | 11 | 15 |
| 3: Batching | 7.5 | 10 |
| 4: PagedAttention | 10 | 12 |
| 5: Parallelism | 5 | 12 |
| 6: Key Optimizations | 6.5 | 15 |
| 7: Long Context & Attention | 0 | 8 |
| 8: System Design | 0 | 10 |
| 9: Quick Fire | 7 | 8 |

---

## Section 1: Two Phases (5/10)

**Q1 (2/2) ✅:** What are the two phases of LLM inference, and what is each phase bound by?

**Your answer:**
Prefill and decoding. Prefill is bounded by computation, decoding is bounded by memory

> **Correct.** Prefill is compute-bound (processing all input tokens in parallel), decode is memory-bandwidth-bound (loading weights for one token at a time).

---

**Q2 (1/3) 🟡:** A Llama 70B model is running on a single A100 (312 TFLOPS FP16, 2 TB/s bandwidth). What is the approximate roofline crossover batch size where decode goes from memory-bound to compute-bound? Show your reasoning.

**Your answer:**
In decoding, 1 token requires 70B flops (ignore attention score) = 70,000,000,000 op, to fill 312 TFLOPS, we need 312 * 1024 * 1024 * 1024 * 1024 = 4900 tokens

Decoding 1 token requires ... (not sure)

> **Partial credit — right direction but wrong approach.** The roofline crossover is about the arithmetic intensity, not filling compute capacity.
>
> **Correct approach:** The crossover happens when compute time = memory time.
> - Memory: must load 140 GB of weights per decode step regardless of batch size → 140 GB / 2 TB/s = 70 ms
> - Compute: batch_size × 140 GFLOPS per token → batch_size × 140 GFLOPS / 312 TFLOPS
> - Set equal: batch_size × 140e9 / 312e12 = 70ms → batch_size ≈ **156**
>
> Simpler shortcut: **crossover ≈ compute_flops / memory_bandwidth = 312 TFLOPS / 2 TB/s = 156 ops/byte**. Since each weight is 2 bytes (FP16) and does 2 FLOPs per element per batch item, crossover batch ≈ 156. The key insight: this is just the **ops:byte ratio** of the hardware.

---

**Q3 (2/2) ✅:** What do TTFT and TTBT stand for, and which phase does each correspond to?

**Your answer:**
TTFT = Time To First Token → corresponding to prefill phase
TTBT = Time Between Token → corresponding to decoding phase

> **Correct.** (Minor: TTBT is "Time Between Tokens" — plural, but that's nitpicking.)

---

**Q4 (0/3) ❌:** Why does batching help the decode phase so much but barely affects prefill?

**Your answer:**
Because prefill is compute bound, (not sure what to follow)

> **Incomplete.** You identified that prefill is compute-bound but didn't explain the mechanism.
>
> **Key insight:** Decode is memory-bound — you load the entire model weights to generate just ONE token. That's terrible hardware utilization. Batching lets you **amortize the weight load across B tokens** — same memory reads, B× more useful compute. The GPU goes from idle to actually doing math.
>
> Prefill is already compute-bound — it processes many tokens per request in parallel, so the GPU is already busy. Adding more requests doesn't help much; you're already saturating the ALUs.
>
> **Remember:** Batching helps when you're memory-bound (lots of data loading, little compute). It doesn't help when you're already compute-bound.

---

## Section 2: Memory Math (11/15)

**Q5 (3/3) ✅:** How much memory do Llama 70B weights consume in FP16? In INT8?

**Your answer:**
70B × 2 = 140GB in FP16
70B in INT8

> **Correct.** 140 GB FP16, 70 GB INT8.

---

**Q6 (4/4) ✅:** Calculate the KV cache memory per token for Llama 70B in FP16.

**Your answer:**
per token KV cache memory = 2 × 80 × 8 × 128 × 2 = 327,680 bytes

> **Correct. 320 KB per token.** Formula: 2 (K+V) × num_layers × num_kv_heads × head_dim × bytes_per_element. Your question "why not check embedding size?" — because KV heads have their own projection dimension (num_kv_heads × head_dim), which is independent of the full embedding size. In GQA models, kv_heads × head_dim < embedding_dim.

---

**Q7 (2/4) 🟡:** Concurrent 4K-context requests on 4× A100 80GB.

**Your answer:**
320GB - 70GB = 250GB → should be 320 - 140 = 180GB (weights are 140GB in FP16!)
320GB × 10% = 32GB
250 - 32 = 218GB → should be 180 - 32 = 148 GB
4K × 327,680 = 1.34GB → ✅ correct (1.31 GB)
218 / 1.34 ≈ 162 requests → **should be ~113 requests**

> **Right method, wrong weight number.** You used 70 GB for FP16 weights instead of 140 GB. That's the INT8 size. The formula and KV cache math were correct though.
>
> **Corrected:** 320 - 140 - 32 = 148 GB available. 148 / 1.31 ≈ **113 concurrent requests**.

---

**Q8 (2/4) 🟡:** FlashAttention question.

**Your answer:**
4096 × 4096 × 2 = 33MB. It's problematic because O(S²). FlashAttention addresses this by tiling and using shared memory instead of global memory, then using online softmax trick.

> **Partially correct.** The 33 MB is for ONE attention head. You need to multiply by the number of heads.
> - Llama 70B has 64 query heads: 64 × 4096 × 4096 × 2 = **2 GB per layer**
> - With 80 layers, that's **160 GB** just for attention scores — exceeds GPU memory!
>
> Your FlashAttention explanation captured the key ideas (tiling, SRAM, online softmax) but lacked precision. FlashAttention computes attention in **tiles** that fit in SRAM (shared memory), never materializing the full S×S matrix in HBM. It uses the **online softmax** trick to compute exact attention incrementally. Memory goes from O(S²) → **O(S)**.

---

## Section 3: Batching (7.5/10)

**Q9 (3/3) ✅:** Continuous batching explanation.

**Your answer:**
Static batching batches all tasks and runs until all finish. Continuous batching schedules at iteration steps, evicting finished requests and inserting new ones.

> **Correct and well explained.** The key insight about GPU idling with static batching is spot on.

---

**Q10 (3/4) 🟡:** Chunked prefill (Sarathi).

**Your answer:**
Chunked prefill segments prefill to 512 tokens per segment and interleaves them with decoding tasks. Solves prefill interfering with TTBT.

> **Mostly correct.** The chunk size isn't always 512 — it's configurable (often matches the token budget). The key nuance you missed: without chunked prefill, a long prefill **monopolizes the GPU** for its entire duration, stalling ALL decode requests. It's not just "interleaving" — it's ensuring decode requests get scheduled every iteration to maintain consistent TTBT. You got the core problem and solution though.

---

**Q11 (1.5/3) 🟡:** Token budget of 2048, 20 active decode requests, new 6000-token prompt.

**Your answer:**
Segment 6000-token prompt into ~12 prefill jobs of 500 tokens, interleave with existing decode requests.

> **Right idea, wrong math.** Each decode request consumes 1 token from the budget. So: 2048 - 20 = **2028 tokens available** for prefill per step. The 6000-token prompt gets chunked into ~3 chunks (2028 + 2028 + 1944), not 12. Each step processes decode tokens + one prefill chunk. The new request doesn't wait for existing decodes to finish — it starts immediately with the remaining budget.
>
> Your question about KV cache spilling is good — in practice, if KV cache memory is full, the request gets queued (or existing requests get preempted/swapped to CPU).

---

## Section 4: PagedAttention (10/12)

**Q12 (4/4) ✅:** PagedAttention explanation.

**Your answer:**
Mimics OS virtual memory with block tables. Divides GPU memory into fixed blocks with virtual-to-physical mapping and reference counting. Reduces external fragmentation.

> **Excellent.** Great analogy and you mentioned reference counting, which many people miss. Internal fragmentation is limited to the last block only.

---

**Q13 (3/4) 🟡:** 145 tokens, block size 16.

**Your answer:**
10 blocks, ~0.94 block internal fragmentation → ~15 tokens wasted. Way better than pre-allocating 2048.

> **Close!** 145 / 16 = 9.0625 → **10 blocks** ✅. Internal fragmentation = 10 × 16 - 145 = **15 tokens** ✅. But you said "0.94 block" — it's 15/16 = **0.9375 blocks**, so essentially right, just slightly off. Pre-allocation comparison: 2048 tokens would need 128 blocks vs. 10 blocks = **92% memory savings**. You got the right idea but could have quantified the comparison more precisely.

---

**Q14 (3/4) 🟡:** Beam search and copy-on-write.

**Your answer:**
Beam search sequences share prefix → share KV cache blocks. Copy-on-write when next token diverges → copy the block. Evict when sequence drops outside top-k.

> **Good understanding.** Minor correction: CoW doesn't happen when tokens "diverge" — it happens when a **shared block needs to be modified** (i.e., a new token is appended to a block that has refcount > 1). The system copies the block, decrements the old block's refcount, and writes the new token to the copy. Eviction of beams that fall out of top-k simply decrements refcounts and frees blocks when refcount hits 0.

---

## Section 5: Parallelism (5/12)

**Q15 (2/3) 🟡:** Tensor parallelism.

**Your answer:**
TP splits model weights across GPUs. Requires fast interconnect (NCCL) for 2 all-reduce operations per layer, one before layer norm and one before ??

> **Right concept, incomplete.** TP splits weight matrices (columns of the first linear, rows of the second) so each GPU computes a partial result. The 2 all-reduces per transformer block are: one after the **attention output projection** and one after the **MLP down projection** (both are row-parallel layers that need to sum partial results). It requires NVLink-class bandwidth because these all-reduces happen **every single layer** — with 80 layers, any latency compounds fast.

---

**Q16 (0/3) ❌:** Bubble ratio.

**Your answer:**
Can't do the math

> **Formula:** Bubble ratio = (P - 1) / (P - 1 + M) where P = pipeline stages, M = micro-batches.
> - (4 - 1) / (4 - 1 + 12) = 3 / 15 = **20%**
>
> **Intuition:** The first and last stages are idle while the pipeline fills and drains. More micro-batches (M) amortize this overhead. That's why M >> P is desired.

---

**Q17 (2/3) 🟡:** Ring Attention.

**Your answer:**
Each GPU holds partial attention, ring-based computation without materializing entire attention score. Gemini uses it.

> **Mostly right.** More precisely: the sequence is split across GPUs. Each GPU holds its own **KV chunk** and the full **Q chunk** for its portion. KV blocks are passed around the ring — each GPU computes partial attention with the visiting KV block, then passes it to the next GPU. After a full rotation, each GPU has computed full attention over the entire sequence. **Gemini** is a reasonable answer though the canonical reference is the Ring Attention paper (UC Berkeley). Google likely uses a variant.

---

**Q18 (1/3) 🟡:** Serving Llama 405B.

**Your answer:**
810 GB / 80 GB = need at least 9 GPUs, say 10. Use TP.

> **Minimum GPU count is right** (at least 11 to leave room for KV cache + overhead, but 10-11 is reasonable). However, **pure TP across 10+ GPUs is not practical** — TP doesn't scale well beyond 8 GPUs because all-reduce cost grows and you need NVLink domains (typically 8 GPUs per node). The correct answer: **TP=8 within a node + PP=2 across nodes** (16 GPUs) or similar hybrid. At 10+ GPUs you almost always combine TP + PP.

---

## Section 6: Key Optimizations (6.5/15)

**Q19 (0/3) ❌:** Speculative decoding.

**Your answer:**
Speculative decoding

> **No answer provided.**
>
> **Speculative decoding:** Use a small, fast **draft model** to generate K candidate tokens autoregressively. Then run the large **target model** on all K+1 positions in a **single forward pass** (parallel, like prefill). Accept tokens where the target model agrees; reject and resample where it doesn't. Speedup = **K × acceptance_rate**. The acceptance rate depends on how well the draft model approximates the target. Typical speedup: **2-3×**. Key insight: verification is cheap (parallel) while generation is expensive (sequential).

---

**Q20 (2/3) 🟡:** GQA explanation.

**Your answer:**
Grouped Query Attention. Reduces number of heads. One head represents multi-dimensional aspects. Reduction factor is 8.

> **Reduction factor is correct** (64/8 = 8× less KV cache). But your explanation of the mechanism is vague. **GQA** groups multiple query heads to share a single KV head. Llama 70B: 64 query heads share 8 KV heads → groups of 8 query heads share 1 KV pair. This reduces KV cache by 8× vs MHA (where each query head has its own KV head). Quality impact is minimal because the KV projections are the less expressive part of attention.

---

**Q21 (0/3) ❌:** CUDA Graphs.

**Your answer:**
Cuda graph is the computation graph. We can prune the unused path.

> **Incorrect.** CUDA Graphs are not about pruning.
>
> **CUDA Graphs** capture a sequence of GPU kernel launches into a **replayable graph**. Normally, each kernel launch has CPU overhead (~10μs). For inference decode, which launches hundreds of tiny kernels per step, this overhead dominates. CUDA Graphs record the entire sequence once, then **replay it in a single launch** — eliminating per-kernel CPU overhead. Speedup: 10-30% for decode.
>
> **Main constraint:** The graph is static — tensor shapes must be identical on every replay. This means you need separate graphs for different batch sizes, and dynamic shapes (varying sequence lengths) require padding or bucketing.

---

**Q22 (3/3) ✅:** Prefix caching (RadixAttention).

**Your answer:**
SGLang's method. Builds a radix/prefix tree to manage KV cache blocks by checking shared prefixes. Most beneficial for chatbots with shared system prompts.

> **Correct.** The radix tree enables O(prefix_length) lookup for cached KV blocks. Also benefits few-shot prompting, tool-use patterns, and any workload with repetitive prefixes.

---

**Q23 (1.5/3) 🟡:** Quantization comparison.

**Your answer:**
2 bytes, 1 byte, 1 byte (int), half a byte. INT8 is better than FP8, INT4 worst quality.

> **Sizes correct.** But "INT8 is better than FP8" is **not generally true** — it depends on the use case.
>
> **Corrected comparison:**
> - **FP16 (2B):** Baseline quality, no degradation. 1× savings.
> - **FP8 (1B):** 2× savings. Minimal quality loss. Better for **activations** because it handles the dynamic range of intermediate values well. H100 has native FP8 support.
> - **INT8 (1B):** 2× savings. Better for **weights** (more uniform distribution). Needs calibration. Mature ecosystem (GPTQ, SmoothQuant).
> - **INT4 (0.5B):** 4× savings. Noticeable quality degradation, especially on reasoning tasks. GPTQ/AWQ help. Good for fitting large models on consumer GPUs.
>
> FP8 vs INT8 is a tradeoff, not a clear winner.

---

## Section 7: Long Context & Attention (0/8)

**Q24 (0/4) ❌:** Attention sinks / StreamingLLM.

**Your answer:** No idea

> **Attention sinks:** Researchers found that LLMs allocate disproportionately high attention scores to the **first few tokens** regardless of their content. These are "attention sinks" — they act as a dumping ground for excess attention probability mass (because softmax must sum to 1).
>
> **StreamingLLM** keeps a small **attention sink window** (first 4 tokens) + a **rolling window** of recent tokens. This lets the model process infinite-length streams with fixed memory. Without the sink tokens, the rolling window alone causes quality collapse because the model's attention distribution breaks.

---

**Q25 (0/4) ❌:** Three approaches for extending context length.

**Your answer:** No idea

> Three approaches:
> 1. **RoPE scaling (Position Interpolation / NTK-aware):** Scale or modify the rotary position embeddings so positions beyond the training window map to the learned range. Cheap — just change the frequency base. Needs some fine-tuning.
> 2. **Ring Attention / sequence parallelism:** Distribute the sequence across GPUs as discussed in Q17. Doesn't change the model — just makes long sequences computationally feasible.
> 3. **Landmark / sparse attention:** Replace full attention with sparse patterns (local windows + global landmarks/tokens). Reduces O(S²) to O(S·√S) or O(S·log S). Examples: Longformer, BigBird, landmark attention.

---

## Section 8: System Design (0/10)

**Q26 (0/5) ❌:** Production LLM serving system design.

**Your answer:** (blank)

> **Example answer:**
> - **Demand:** 10K users × 2K context = 20M tokens of KV cache. At 320 KB/token = ~6.4 TB KV cache.
> - **Hardware:** Multiple nodes of 8× H100 80GB. With TP=8 per node, each replica serves one copy of Llama 70B. Need enough replicas for throughput.
> - **Parallelism:** TP=8 within node (NVLink). Scale horizontally with replicas behind a load balancer.
> - **Optimizations:** PagedAttention (memory efficiency), continuous batching + chunked prefill (throughput + latency), prefix caching (shared system prompt), GQA already built into Llama 70B.
> - **Stack:** vLLM or TensorRT-LLM behind an API gateway. Autoscale replicas on queue depth.
> - **Metrics:** TTFT (P50/P99), TTBT (P50/P99), request throughput, queue depth, GPU memory utilization, KV cache utilization, token/s per GPU.

---

**Q27 (0/5) ❌:** Debugging P99 TTBT spike.

**Your answer:** (blank)

> **Debugging process:**
> 1. **Check batch size** — has traffic spiked? Larger batches = more decode compute per step. Look at queue depth and concurrent requests.
> 2. **Check for prefill interference** — are long prefills starving decode? Look at prefill lengths in recent requests. Fix: enable/tune chunked prefill.
> 3. **KV cache pressure** — is the system swapping KV cache to CPU? Check memory utilization. Swapping causes massive latency spikes.
> 4. **CUDA Graph misses** — new batch sizes triggering graph recompilation? Check for CUDA graph cache hit rate.
> 5. **GPU thermal throttling** — check `nvidia-smi` for clock speed drops.
> 6. **GC pauses** — Python garbage collection can cause jitter. Check if GC correlates with latency spikes.
> 7. **Network** — if using TP across nodes, check interconnect latency. NVLink vs PCIe degradation.

---

## Section 9: Quick Fire (7/8)

**Q28 (0/1) ❌:** A100 80GB HBM bandwidth?
**Your answer:** 10 TB/s
> **Correct answer: 2 TB/s.** You're off by 5×. (This was given in Q2!)

**Q29 (0/1) ❌:** H100 80GB FP16 TFLOPS?
**Your answer:** 500 TFLOPS
> **Correct answer: ~990 TFLOPS** (with sparsity: ~2000). Close-ish but not close enough for interview precision.

**Q30 (1/1) ✅:** Flash Attention memory complexity?
**Your answer:** O(S)
> **Correct.**

**Q31 (1/1) ✅:** Continuous vs static batching speedup?
**Your answer:** 2-3×
> **Correct.** (Commonly cited as 2-4× depending on workload variance.)

**Q32 (1/1) ✅:** Speculative decoding speedup?
**Your answer:** 2-3×
> **Correct.**

**Q33 (1/1) ✅:** Llama 70B FP16 minimum A100 80GB GPUs?
**Your answer:** 2
> **Correct.** 140 GB / 80 GB → 2 GPUs minimum.

**Q34 (2/1) ✅:** Autoscaling signal?
**Your answer:** TTFT and TTBT
> **Correct.** Queue depth is also commonly accepted, but latency metrics are the gold standard.

**Q35 (1/1) ✅:** PagedAttention memory utilization improvement?
**Your answer:** 3-4×
> **Correct.** (vLLM paper shows up to 4× improvement in effective memory utilization.)

---

## 📊 Summary & Coaching

**Score: 52/100**

### Strengths 💪
- **Memory math fundamentals** are solid — you nail KV cache calculations, weight sizing, and PagedAttention mechanics
- **Batching concepts** well understood — continuous batching, chunked prefill core ideas are there
- **Quick fire numbers** mostly good — you know the practical ballpark figures

### Gaps to Focus On 🎯

1. **Roofline model / arithmetic intensity** (Q2, Q4) — This is CRITICAL for NVIDIA interviews. Understand ops:byte ratio, when you're compute vs memory bound, and WHY batching helps. Practice the crossover calculation until it's automatic.

2. **Parallelism strategies** (Q15-Q18) — You know TP exists but not the details. Study: where the all-reduces happen, why TP maxes out at 8 GPUs, when to combine TP+PP. This WILL come up at NVIDIA.

3. **Long context & attention** (Q24-Q25) — Complete blind spot. Study StreamingLLM, RoPE scaling, and sparse attention patterns. At least know the concepts.

4. **System design** (Q26-Q27) — Blank answers here are concerning for a staff-level interview. Practice the end-to-end design: sizing → hardware → parallelism → optimizations → monitoring. And debugging methodology matters — walk through it systematically.

5. **CUDA Graphs & speculative decoding** (Q19, Q21) — These are core inference optimizations you should be able to explain clearly.

6. **Hardware numbers** — Know A100 (2 TB/s, 312 TFLOPS) and H100 (3.35 TB/s, 990 TFLOPS) cold. These are your anchors for all back-of-envelope calculations.

### Priority for NVIDIA Interview Today 🔴
Given you have ~4 hours: focus on **roofline analysis**, **parallelism (TP/PP)**, and **system design patterns**. These are the highest-probability topics for an AI infra role at NVIDIA.

Good luck at 2 PM! 🚀

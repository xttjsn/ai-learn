# LLM Inference Systems — Interview Cheatsheet

## 1. The Two Phases

| | Prefill | Decode |
|---|---|---|
| What | Process entire prompt in parallel | Generate 1 token per step |
| Bound | **Compute** (big matmuls) | **Memory bandwidth** (load all weights for 1 token) |
| FLOPs | `2 × prompt_len × params` | `2 × params` per token |
| Arithmetic intensity | High → GPU happy | ~1 op/byte → GPU starved |
| Latency metric | **TTFT** (time to first token) | **TTBT** (time between tokens) |

**Key insight:** Batching helps decode enormously — same weight load, B tokens out instead of 1.

Roofline crossover on A100 (2TB/s BW, 312 TFLOPS FP16): batch ≈ 156.

---

## 2. Memory Math — Know These Cold

### Model Weights
```
Params × bytes_per_param
70B × 2 bytes (FP16) = 140 GB
70B × 1 byte (INT8)  = 70 GB
```

### KV Cache Per Token
```
2 × n_layers × n_kv_heads × head_dim × dtype_bytes

Llama 70B (FP16): 2 × 80 × 8 × 128 × 2 = 327,680 bytes ≈ 320 KB/token
Full 2K context: 320KB × 2048 ≈ 640 MB per request
Full 8K context: 320KB × 8192 ≈ 2.56 GB per request
```

### Quick Estimates
| Model | Weights (FP16) | KV/token (FP16) | KV @ 4K context |
|---|---|---|---|
| 7B (32L, 32H, d128) | 14 GB | 32 KB | 128 MB |
| 13B (40L, 40H, d128) | 26 GB | 40 KB | 160 MB |
| 70B (80L, 8KVH, d128) | 140 GB | 320 KB | 1.28 GB |
| 405B (126L, 8KVH, d128) | 810 GB | 504 KB | 2.02 GB |

### Attention Scores (Prefill, Without FlashAttn)
```
n_heads × seq_len² × dtype_bytes

Llama 70B, S=4096, FP16:
64 × 4096 × 4096 × 2 = 2.15 GB  ← per layer!
```
Flash Attention: O(S) memory instead of O(S²). Never materializes full score matrix.

### Training Activation Memory
```
≈ S × B × H × L × (10 + 24/T) bytes (mixed precision)
Llama 70B, B=1, S=4096, T=4: ≈ 43 GB
```
Activation checkpointing: save every √L layers, recompute rest → ~90% reduction, ~33% slower.

---

## 3. Batching Techniques

### Static Batching
- Pad all sequences to max length, run until slowest finishes
- Wastes GPU on finished sequences
- **Never say this is acceptable in an interview**

### Continuous Batching (Orca)
- Schedule at **iteration level** — every decode step
- Finished request → free slot → admit new request immediately
- GPU never idles on completed sequences

### Chunked Prefill (Sarathi / SGLang)
- Long prefills stall decode requests → TTBT spikes
- Split prefill into chunks (e.g. 512 tokens), interleave with decode
- Token budget per step (e.g. 2048): pack decode tokens + prefill chunk
- Result: consistent latency for all in-flight requests

### Dynamic Batching (Triton / TorchServe)
- Collect requests within time window or batch-size threshold
- Still pads to max length within batch — better than static, worse than continuous

---

## 4. PagedAttention (vLLM)

**Problem:** Pre-allocating max_seq_len KV cache per request wastes ~50% memory.

**Solution:** Virtual memory for KV cache.
- Fixed-size blocks (e.g. 16 tokens)
- Block table maps logical → physical blocks
- Allocate on demand as sequence grows
- ~96%+ memory utilization (vs ~50-60% pre-allocated)

```
Request A (145 tokens): block_table = [7, 22, 41, 3, 88, 91, 102, 66, 11]
                        9 blocks × 16 = 144 capacity, last block 1/16 full

Request B (30 tokens):  block_table = [12, 55]
                        2 blocks
```

**Copy-on-write:** Beam search / parallel sampling share prefix blocks. Fork only on divergence.

**Custom CUDA kernel:** Standard attention assumes contiguous KV. PagedAttention kernel takes block table + physical block pointers, computes attention across non-contiguous blocks using online softmax.

---

## 5. Overload Handling

| Strategy | How | When |
|---|---|---|
| Queuing | FIFO / priority / SLO-based | Default |
| Swap to CPU | Copy KV cache to CPU RAM via PCIe (~32 GB/s) | Long sequences |
| Recomputation | Drop KV, re-prefill when rescheduled | Short sequences |

**Preemption policy:** evict lowest-priority or longest sequence first.

---

## 6. Parallelism Strategies

### Tensor Parallelism (TP)
- Split weight matrices across GPUs within one node
- Each GPU holds a **slice** of every layer
- Requires all-reduce every layer → needs fast interconnect (NVLink)
- Typical: TP=2, 4, or 8 within a node

### Pipeline Parallelism (PP)
- Split layers across GPUs: GPU 0 = layers 0-19, GPU 1 = layers 20-39, etc.
- Micro-batches flow through like assembly line
- **Bubble ratio** = `(P-1) / (M+P-1)` where P=stages, M=micro-batches
- 1F1B schedule: interleave forward/backward to minimize bubble

### Sequence Parallelism (SP) / Ring Attention
- Split the **sequence** across devices
- Each device holds a chunk of KV cache
- KV blocks rotate around a ring — compute attention tile, pass KV to neighbor
- Communication overlapped with compute
- How Gemini does 1M context

### Expert Parallelism (EP) — MoE models
- Each GPU holds a subset of experts
- All-to-all communication to route tokens to correct expert
- Load imbalance = main challenge

### Typical Configs
```
Llama 70B inference:   TP=4 on one node (4× A100/H100)
Llama 405B inference:  TP=8, PP=2 (16 GPUs, 2 nodes)
Llama 70B training:    TP=4, PP=2, DP=64 (512 GPUs)
```

---

## 7. Key Optimizations

### Flash Attention
- Fused kernel: Q,K,V → output in one pass, tile-by-tile
- O(S) memory instead of O(S²)
- 2-4× faster than standard attention
- Uses **online softmax** (log-sum-exp running max) — no need for two passes

### GQA / MQA
```
MHA: 64 Q heads, 64 KV heads → full KV cache
GQA: 64 Q heads, 8 KV heads  → 8× smaller KV cache
MQA: 64 Q heads, 1 KV head   → 64× smaller KV cache
```
Llama 2 70B, Gemini, Gemma all use GQA.

### Speculative Decoding
```
Draft model (small, fast) → generates K=5 candidate tokens
Target model (large) → verifies all K+1 positions in ONE forward pass
Accept/reject via modified rejection sampling
Speedup ≈ (avg_accepted + 1) / (K × draft_time/target_time + 1)
Typical: 2-3× decode speedup
```

### Quantization
| Type | Bits | Weight Memory | Quality Impact |
|---|---|---|---|
| FP16/BF16 | 16 | Baseline | None |
| FP8 | 8 | 0.5× | Minimal |
| INT8 (W8A8) | 8 | 0.5× | Minimal |
| INT4 (GPTQ/AWQ) | 4 | 0.25× | Slight degradation |
| GGUF Q4_K_M | ~4.5 | ~0.28× | Acceptable for most |

KV cache can also be quantized independently (FP8 KV cache = 2× more concurrent requests).

### CUDA Graphs
- Capture decode step as a static graph → replay without kernel launch overhead
- Eliminates CPU-side scheduling cost for small batches
- Constraint: graph shape must be fixed (padded to max batch size)

### Prefix Caching (SGLang RadixAttention)
- Many requests share system prompt → cache that prefix KV once
- Trie/radix tree to find longest matching prefix
- 2-3× throughput for chatbot workloads with shared system prompts

---

## 8. Attention Variants for Long Context

| Method | Complexity | How |
|---|---|---|
| Full dense | O(S²) | Every token attends to every token |
| Sliding window | O(S×W) | Attend to last W tokens only (Mistral) |
| Dilated / strided | O(S×S/k) | Every k-th token at long range |
| Global + local | O(S×(W+G)) | Few global tokens + local window (BigBird, Longformer) |
| Ring Attention | O(S²/N) per device | Distribute across N devices, rotate KV |
| StreamingLLM | O(S×(W+sink)) | Keep first few "sink" tokens + sliding window |

**Attention sinks:** First few tokens accumulate attention mass regardless of content. Removing them crashes generation. StreamingLLM keeps them as anchors.

**RoPE scaling for extended context:**
- Position interpolation: scale positions by `orig_ctx / new_ctx`
- NTK-aware: scale the frequency basis of RoPE
- YaRN: combines NTK + temperature scaling per dimension
- All allow extending context without full retraining

---

## 9. Monitoring Metrics

| Metric | What | Target |
|---|---|---|
| TTFT | Time to first token | < 500ms typical |
| TTBT (P50/P99) | Inter-token latency | < 30ms typical |
| Throughput | Tokens/sec across all requests | Maximize |
| KV cache utilization | % of KV cache memory in use | 80-95% ideal |
| Queue depth | Pending requests | Scale trigger |
| GPU utilization | SM activity | Misleading alone (can be 99% at B=1) |
| Goodput | Tokens/sec of useful output (excluding padding/speculation waste) | True efficiency |

**Autoscaling signal:** Queue depth + KV utilization, NOT GPU util.

---

## 10. System Architecture — End to End

```
                         ┌─────────────┐
                         │  Clients    │
                         └──────┬──────┘
                                │
                         ┌──────▼──────┐
                         │Load Balancer│  (KV-cache-aware routing)
                         └──────┬──────┘
                                │
                    ┌───────────┼───────────┐
                    │           │           │
              ┌─────▼─────┐ ┌──▼──┐ ┌─────▼─────┐
              │ Replica 0 │ │ R1  │ │ Replica N │
              └─────┬─────┘ └─────┘ └───────────┘
                    │
         ┌──────────┼──────────┐
         │          │          │
    ┌────▼────┐ ┌───▼───┐ ┌───▼───┐
    │Scheduler│ │Prefix │ │Tokeniz│
    │         │ │Cache  │ │-er    │
    └────┬────┘ └───────┘ └───────┘
         │
    ┌────▼────────────────────────┐
    │ GPU Workers (TP=4)          │
    │ ┌──────┐┌──────┐┌──────┐   │
    │ │GPU 0 ││GPU 1 ││GPU 2 │...│
    │ │      ││      ││      │   │
    │ │KV    ││KV    ││KV    │   │
    │ │Cache ││Cache ││Cache │   │
    │ └──────┘└──────┘└──────┘   │
    └─────────────────────────────┘
```

---

## 11. Common Follow-Up Questions

**"How does this change for MoE models (e.g. Mixtral, DeepSeek)?"**
- Only ~25% of params active per token, but full model in VRAM
- Expert parallelism: all-to-all routing across GPUs
- Load imbalance: if all tokens route to same expert, that GPU bottlenecks
- Auxiliary loss during training to encourage balanced routing

**"What if you need 100K+ context?"**
- Ring attention / sequence parallelism across devices
- Sparse attention patterns (local + global)
- RoPE scaling (YaRN, NTK-aware)
- KV cache quantization (FP8/INT4)
- Offload distant KV to CPU, prefetch on demand

**"How would you reduce cost?"**
- Quantize (FP8 → halve GPU count)
- Speculative decoding (2-3× throughput for free)
- Prefix caching (shared system prompts)
- Right-size instances: don't use 8×H100 if 2×A100 + INT8 fits
- Spot instances for batch workloads

**"How do you handle multi-turn conversations?"**
- Cache KV from previous turns (session affinity to same replica)
- If session migrates, re-prefill from conversation history
- Prefix caching helps — previous turns are a shared prefix

---

## 12. Numbers to Have Memorized

```
A100 80GB:    312 TFLOPS FP16, 2 TB/s HBM bandwidth, 600 GB/s NVLink
H100 80GB:    990 TFLOPS FP16, 3.35 TB/s HBM, 900 GB/s NVLink
B200:         ~2.25 PFLOPS FP16, 8 TB/s HBM

Llama 70B weights (FP16):  140 GB → minimum 2× A100 (TP=2)
Llama 70B KV cache:        320 KB/token → 1.28 GB per 4K request
Llama 70B decode:          ~40 tokens/sec single request on 4×A100
Llama 70B prefill:         ~2000 tokens/sec on 4×A100

vLLM PagedAttention:       ~2-4× more concurrent requests vs naive
Flash Attention:           ~2-4× faster, O(S) memory vs O(S²)
Speculative decoding:      ~2-3× decode throughput
Continuous batching:       ~10-20× throughput vs static batching
```

---

*Last updated: March 2026 | For: NVIDIA, Cerebras, and general ML infra interviews*

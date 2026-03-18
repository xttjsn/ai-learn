# System Design: Batch GPU Inference at Scale

## The Problem

Design a system that serves LLM inference requests at scale.
Requirements:
- Low latency (p50 < 200ms TTFT, p99 < 2s)
- High throughput (thousands of requests/sec)
- Efficient GPU utilization (>80%)
- Handle variable-length sequences
- Graceful degradation under load

---

## Architecture

```
                    ┌──────────────┐
                    │  API Gateway  │
                    │  (rate limit,  │
                    │   auth, route) │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Request Queue │
                    │  (priority    │
                    │   queue)      │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   Scheduler   │
                    │  (batch       │
                    │   formation)  │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────▼─────┐ ┌───▼───┐ ┌─────▼─────┐
        │ GPU Worker │ │  ...  │ │ GPU Worker │
        │ (model +   │ │       │ │ (model +   │
        │  KV cache)  │ │       │ │  KV cache)  │
        └─────┬─────┘ └───┬───┘ └─────┬─────┘
              │            │            │
              └────────────┼────────────┘
                           │
                    ┌──────▼───────┐
                    │ Response      │
                    │ Fan-out       │
                    │ (SSE/WebSocket)│
                    └──────────────┘
```

---

## Component Deep Dives

### 1. API Gateway
- Rate limiting per user/API key
- Authentication & authorization
- Request validation (max tokens, prompt length)
- Load balancing across scheduler instances
- Streaming support (SSE for token-by-token)

### 2. Request Queue
- Priority queue (premium users, shorter requests)
- SLO-aware ordering: requests close to deadline get priority
- Backpressure: reject or queue when overloaded
- Persistence: don't lose requests on crash (Redis/Kafka)

### 3. Scheduler (THE key component)
The scheduler decides:
- **What** goes into each batch
- **When** to flush the batch
- **Where** (which GPU) to send it

**Batch formation strategies:**

| Strategy | Pros | Cons |
|----------|------|------|
| Fixed size | Simple | Padding waste |
| Fixed timeout | Bounded latency | Variable batch size |
| Token budget | Memory-efficient | Complex |
| Continuous | Best utilization | Most complex |

**Continuous batching algorithm:**
```
while True:
    # Remove completed sequences
    for req in running_batch:
        if req.is_done():
            send_response(req)
            free_kv_cache(req)

    # Add new sequences from queue
    while queue.not_empty() and can_fit(queue.peek()):
        req = queue.pop()
        allocate_kv_cache(req)
        running_batch.add(req)

    # Run one forward pass
    outputs = model.forward(running_batch)

    # Update each sequence
    for req, output in zip(running_batch, outputs):
        req.append_token(output)
        if output == EOS or req.at_max_length():
            req.mark_done()
```

### 4. GPU Workers

**KV Cache Management (critical!):**
- Each sequence needs KV cache: `2 × num_layers × d_model × seq_len × dtype_size`
- Example: Llama-70B, seq_len=2048, fp16:
  `2 × 80 × 8192 × 2048 × 2 bytes ≈ 5.2 GB per sequence`
- GPU memory is the bottleneck, not compute!

**PagedAttention (vLLM's key innovation):**
- Allocate KV cache in pages (blocks of tokens)
- Like virtual memory: logical → physical mapping
- No fragmentation, no pre-allocation waste
- Enables memory sharing across sequences (beam search, parallel sampling)

**Memory management strategies:**
1. **Pre-allocation:** Reserve max_seq_len for each slot. Simple but wasteful.
2. **Dynamic:** Allocate as sequence grows. Fragmentation risk.
3. **Paged (vLLM):** Block-based allocation. Best of both worlds.

### 5. Multi-GPU Routing

**Strategies:**
- **Model parallelism:** Split model across GPUs (for large models)
  - Tensor parallelism: split layers across GPUs
  - Pipeline parallelism: different layers on different GPUs
- **Data parallelism:** Full model on each GPU, route requests
  - Round-robin (simple)
  - Least-loaded (better)
  - Affinity-based (keep similar sequences together)

**Load balancing signals:**
- GPU memory utilization
- Queue depth per worker
- Current batch size
- Estimated time to completion

### 6. Streaming Response
- Server-Sent Events (SSE) for HTTP
- WebSocket for bidirectional
- Each token sent as it's generated
- Client sees Time-To-First-Token (TTFT) + inter-token latency

---

## Key Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| TTFT | Time to first token | < 200ms p50 |
| ITL | Inter-token latency | < 30ms |
| Throughput | Tokens/sec across all requests | Maximize |
| GPU Utilization | % time GPU is computing | > 80% |
| Batch Efficiency | Useful tokens / total tokens | > 70% |
| Queue Depth | Pending requests | Monitor |

---

## Failure Modes & Mitigations

| Failure | Mitigation |
|---------|-----------|
| GPU OOM | Preempt low-priority requests, reduce batch size |
| GPU crash | Retry request on another GPU, health checks |
| Long tail latency | Timeout + retry, preemption |
| Overload | Backpressure, load shedding, autoscaling |
| Model corruption | Checksum validation, rollback |

---

## Scaling

**Vertical:** Bigger GPUs (A100 → H100), more VRAM
**Horizontal:** More GPU workers, model replicas
**Autoscaling signals:**
- Queue depth > threshold → scale up
- GPU utilization < threshold → scale down
- Time-of-day patterns (pre-provision for known peaks)

---

## Interview Tips

1. **Start with the API contract** — what does the client see?
2. **Draw the architecture** — keep it simple, 5 boxes max
3. **Deep dive on the scheduler** — this is where the magic is
4. **Discuss KV cache** — shows you understand GPU inference
5. **Mention vLLM/PagedAttention** — shows you know SOTA
6. **Talk numbers** — estimate memory, throughput, latency
7. **Discuss trade-offs** — latency vs throughput, memory vs compute

## Estimation Example

"How many concurrent requests can one A100 (80GB) serve for Llama-70B?"

- Model weights (fp16): ~140GB → need 2 A100s (tensor parallel)
- Per-GPU: 80GB - 70GB weights = 10GB for KV cache
- KV cache per sequence (2048 tokens): ~5.2GB
- So: 10GB / 5.2GB ≈ **1-2 concurrent sequences per GPU**
- With quantization (int4): model = 35GB, 45GB for KV → ~8 sequences
- With PagedAttention: even better (no fragmentation waste)

This is why efficient batching matters so much!

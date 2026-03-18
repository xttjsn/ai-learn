# System Design: Batch GPU Requests / Batch Inference

## Problem
Design a system that efficiently batches incoming inference requests
to maximize GPU utilization. This is SD Q1 at Anthropic — and arguably
the most relevant to their actual work.

## Progressive Levels

### Level 1: Basic Request Batching
- Collect individual requests into batches
- Fixed batch size, fixed timeout
- Simple queue → batch → GPU → fan-out results

### Level 2: Dynamic Batching
- Adaptive batch sizes based on load
- Sequence length-aware batching (pad to max in batch)
- Continuous batching (inflight batching)

### Level 3: Production System Design
- Multi-GPU routing and load balancing
- KV cache management
- Request prioritization / SLOs
- Preemption and scheduling
- Fault tolerance and retries

### Level 4: Full Architecture
- API gateway → Request queue → Batch scheduler → GPU workers → Response fan-out
- Monitoring, autoscaling, model versioning
- A/B testing, canary deployments

## Files
- `level1_basic_batcher.py` — Simple batching with timeout
- `level2_dynamic_batcher.py` — Dynamic + continuous batching
- `level3_system_design.md` — Full system design writeup

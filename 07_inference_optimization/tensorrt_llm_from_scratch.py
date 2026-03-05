"""
TensorRT-LLM: Maximum Performance LLM Inference on NVIDIA GPUs

This module explains TensorRT-LLM architecture and optimization techniques.
Based on:
  - NVIDIA TensorRT-LLM (2023-2024)
  - TensorRT optimization principles
  - In-flight batching and FP8 quantization

================================================================================
PART 1: WHAT IS TensorRT-LLM?
================================================================================

TensorRT-LLM is NVIDIA's inference engine for LLMs. It combines:
  1. TensorRT — NVIDIA's deep learning inference optimizer (graph optimization,
     kernel fusion, quantization)
  2. LLM-specific optimizations — KV cache management, in-flight batching,
     speculative decoding
  3. Multi-GPU support — tensor parallelism, pipeline parallelism

The stack:

    ┌──────────────────────────────────────────────┐
    │  User API (Python)                           │
    │  model = LLM("llama-70b")                    │
    │  output = model.generate("Hello world")      │
    └───────────────────┬──────────────────────────┘
                        │
    ┌───────────────────┴──────────────────────────┐
    │  TensorRT-LLM Runtime                        │
    │  - Request scheduling (in-flight batching)   │
    │  - KV cache management (paged)               │
    │  - Multi-GPU orchestration                   │
    └───────────────────┬──────────────────────────┘
                        │
    ┌───────────────────┴──────────────────────────┐
    │  TensorRT Engine (C++)                       │
    │  - Fused CUDA kernels                        │
    │  - Optimized memory layout                   │
    │  - FP8/INT8/INT4 compute                     │
    └───────────────────┬──────────────────────────┘
                        │
    ┌───────────────────┴──────────────────────────┐
    │  NVIDIA GPU Hardware                         │
    │  - Tensor Cores (matrix multiply)            │
    │  - CUDA Cores (general compute)              │
    │  - HBM3 (memory)                             │
    └──────────────────────────────────────────────┘

Key difference from vLLM/SGLang:
  - vLLM/SGLang use PyTorch for compute, optimize at the SCHEDULING level
  - TensorRT-LLM optimizes at the KERNEL level (fused custom CUDA kernels)
  - This gives TensorRT-LLM the best raw per-request performance
  - But less flexibility (model changes require rebuild)


================================================================================
PART 2: KERNEL FUSION — THE CORE OPTIMIZATION
================================================================================

What is kernel fusion?

    On a GPU, every "kernel" (operation) involves:
    1. Load data from HBM → registers
    2. Compute
    3. Store results back to HBM

    For a transformer layer, standard PyTorch does:

    ┌─────────────────────────────────────────────────────────┐
    │  UN-FUSED (standard PyTorch):                           │
    │                                                         │
    │  Kernel 1: LayerNorm                                    │
    │    HBM → load x → normalize → store y → HBM            │
    │                                                         │
    │  Kernel 2: Q projection                                 │
    │    HBM → load y → matmul W_Q → store Q → HBM           │
    │                                                         │
    │  Kernel 3: K projection                                 │
    │    HBM → load y → matmul W_K → store K → HBM           │
    │                                                         │
    │  Kernel 4: V projection                                 │
    │    HBM → load y → matmul W_V → store V → HBM           │
    │                                                         │
    │  Kernel 5: Q @ K^T                                      │
    │    HBM → load Q,K → matmul → store scores → HBM        │
    │                                                         │
    │  Kernel 6: Softmax                                      │
    │    HBM → load scores → softmax → store probs → HBM     │
    │                                                         │
    │  Kernel 7: probs @ V                                    │
    │    HBM → load probs,V → matmul → store out → HBM       │
    │                                                         │
    │  Kernel 8: Output projection                            │
    │    HBM → load out → matmul W_O → store → HBM           │
    │                                                         │
    │  Kernel 9: Residual add                                 │
    │    HBM → load x, out → add → store → HBM               │
    │                                                         │
    │  Kernel 10: LayerNorm (FFN)                             │
    │    HBM → load → normalize → store → HBM                │
    │                                                         │
    │  Kernel 11: FFN up projection                           │
    │    HBM → load → matmul → store → HBM                   │
    │                                                         │
    │  Kernel 12: GELU activation                             │
    │    HBM → load → gelu → store → HBM                     │
    │                                                         │
    │  Kernel 13: FFN down projection                         │
    │    HBM → load → matmul → store → HBM                   │
    │                                                         │
    │  Kernel 14: Residual add                                │
    │    HBM → load → add → store → HBM                      │
    │                                                         │
    │  14 kernel launches, 14 HBM round trips!                │
    │  Most time spent loading/storing, not computing.        │
    └─────────────────────────────────────────────────────────┘

TensorRT-LLM fuses these into fewer, larger kernels:

    ┌─────────────────────────────────────────────────────────┐
    │  FUSED (TensorRT-LLM):                                  │
    │                                                         │
    │  Fused Kernel 1: LayerNorm + QKV projection             │
    │    HBM → load x, W_QKV → norm + matmul → store Q,K,V   │
    │    (1 read of x instead of 2, QKV in one matmul)        │
    │                                                         │
    │  Fused Kernel 2: FlashAttention (Q@K + softmax + @V)    │
    │    HBM → load Q,K,V → attention → store out             │
    │    (scores NEVER written to HBM — see module 05)        │
    │                                                         │
    │  Fused Kernel 3: Output proj + residual + LayerNorm     │
    │    HBM → load out, x → matmul + add + norm → store      │
    │                                                         │
    │  Fused Kernel 4: FFN (up + GELU + down + residual)      │
    │    HBM → load → all FFN ops → store                     │
    │    (SwiGLU: gate and up fused into one matmul)          │
    │                                                         │
    │  4 kernel launches, 4 HBM round trips!                  │
    │  3.5× fewer memory round trips.                         │
    └─────────────────────────────────────────────────────────┘

    Concrete impact:
    For Llama-70B, single token generation:

    Framework        Time per token    Relative
    ──────────────   ──────────────    ────────
    PyTorch (naive)  ~55 ms            1.0×
    PyTorch + FA2    ~45 ms            1.2×
    vLLM             ~42 ms            1.3×
    TensorRT-LLM     ~35 ms            1.6×

    The gap is mostly from kernel fusion reducing HBM traffic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass
import math


# ============================================================================
# PART 3: IN-FLIGHT BATCHING
# ============================================================================

"""
The batching evolution:

    ┌──────────────────────────────────────────────────────┐
    │  STATIC BATCHING (naive):                            │
    │                                                      │
    │  Batch all requests together. ALL must finish         │
    │  before ANY new request can start.                   │
    │                                                      │
    │  Request 1: ████████████                             │
    │  Request 2: ████████████████████                     │
    │  Request 3: ████████████████                         │
    │  Request 4: ████████                                 │
    │                    ↑ All padded to longest!           │
    │             ════════════════════ Batch completes      │
    │                                 │                    │
    │  New batch starts here ─────────┘                    │
    │                                                      │
    │  Problem: Short requests wait for long ones.          │
    │  GPU utilization: LOW (padding wastes compute)        │
    └──────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────┐
    │  CONTINUOUS BATCHING (vLLM):                         │
    │                                                      │
    │  When a request finishes, immediately replace it     │
    │  with a new one. No waiting for the whole batch.     │
    │                                                      │
    │  Request 1: ████████████                             │
    │  Request 5:             ███████████  (starts when    │
    │  Request 2: ████████████████████      1 finishes)    │
    │  Request 3: ████████████████                         │
    │  Request 6:                 ████████                  │
    │  Request 4: ████████                                 │
    │  Request 7:         ████████████████                  │
    │                                                      │
    │  Better: no wasted slots. But still iteration-level: │
    │  entire batch does ONE decode step together.          │
    └──────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────┐
    │  IN-FLIGHT BATCHING (TensorRT-LLM):                  │
    │                                                      │
    │  Mix PREFILL and DECODE in the SAME batch!            │
    │                                                      │
    │  Iteration 1:                                        │
    │    Req 1: DECODE (1 token)    ← generating           │
    │    Req 2: DECODE (1 token)    ← generating           │
    │    Req 3: PREFILL (100 tokens) ← just arrived!       │
    │                                                      │
    │  All processed in ONE kernel call.                    │
    │                                                      │
    │  Why this matters:                                    │
    │  - Prefill is COMPUTE-bound (lots of tokens)          │
    │  - Decode is MEMORY-bound (1 token, load all weights) │
    │  - Mixing them: prefill saturates compute while       │
    │    decode tokens ride along nearly free!               │
    │                                                      │
    │  Think of it like carpooling:                         │
    │  The bus (weight loading) runs anyway for decode.      │
    │  Prefill tokens hop on and use the spare compute.     │
    └──────────────────────────────────────────────────────┘
"""


@dataclass
class InFlightRequest:
    """Represents a request in the in-flight batching scheduler."""
    request_id: int
    token_ids: List[int]
    phase: str  # "prefill" or "decode"
    tokens_generated: int = 0
    max_tokens: int = 100
    is_complete: bool = False


class InFlightBatchScheduler:
    """
    Simplified in-flight batch scheduler.

    Each iteration:
    1. Check for completed requests (remove them)
    2. Check for new requests (add prefill)
    3. Build batch mixing prefill + decode requests
    4. Execute one iteration

    ┌────────────────────────────────────┐
    │  Scheduler State                   │
    │                                    │
    │  Active requests:                  │
    │    [Req1: decode] [Req2: decode]   │
    │    [Req3: prefill]                 │
    │                                    │
    │  Waiting queue:                    │
    │    [Req4] [Req5] [Req6]            │
    │                                    │
    │  Each iteration:                   │
    │    1. Decode existing requests     │
    │    2. Prefill new requests         │
    │    3. Mix into one batch           │
    └────────────────────────────────────┘
    """
    def __init__(self, max_batch_size: int = 32, max_tokens_per_batch: int = 4096):
        self.active: List[InFlightRequest] = []
        self.waiting: List[InFlightRequest] = []
        self.max_batch_size = max_batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
        self.completed: List[InFlightRequest] = []

    def add_request(self, request: InFlightRequest):
        """Add a new request to the waiting queue."""
        self.waiting.append(request)

    def step(self) -> dict:
        """
        Execute one scheduling step.

        Returns info about what was batched together.
        """
        # Remove completed requests
        self.active = [r for r in self.active if not r.is_complete]

        # Calculate current token budget
        decode_tokens = len(self.active)  # each decode = 1 token
        remaining_budget = self.max_tokens_per_batch - decode_tokens

        # Add new prefill requests if budget allows
        new_prefills = []
        while self.waiting and remaining_budget > 0:
            req = self.waiting[0]
            prefill_tokens = len(req.token_ids)
            if prefill_tokens <= remaining_budget and \
               len(self.active) + len(new_prefills) < self.max_batch_size:
                new_prefills.append(self.waiting.pop(0))
                remaining_budget -= prefill_tokens
            else:
                break

        # Execute: all active requests do decode, new requests do prefill
        for req in self.active:
            req.tokens_generated += 1
            if req.tokens_generated >= req.max_tokens:
                req.is_complete = True
                self.completed.append(req)

        for req in new_prefills:
            req.phase = "decode"  # after prefill, switch to decode
            self.active.append(req)

        return {
            "decode_requests": len(self.active) - len(new_prefills),
            "prefill_requests": len(new_prefills),
            "total_batch_tokens": decode_tokens + sum(
                len(r.token_ids) for r in new_prefills
            ),
            "waiting": len(self.waiting),
        }


# ============================================================================
# PART 4: QUANTIZATION IN TensorRT-LLM
# ============================================================================

"""
TensorRT-LLM has deep quantization integration — not just post-hoc
weight compression, but compute-level quantization using Tensor Cores.

    ┌─────────────────────────────────────────────────────────┐
    │  QUANTIZATION LEVELS:                                    │
    │                                                         │
    │  FP16:  Standard half precision                          │
    │         Weight: 2 bytes, Compute: FP16 Tensor Cores      │
    │         No accuracy loss, baseline performance            │
    │                                                         │
    │  FP8 (E4M3):  8-bit float                               │
    │         Weight: 1 byte, Compute: FP8 Tensor Cores        │
    │         Minimal accuracy loss, 2× throughput vs FP16     │
    │         Supported on H100+ (Hopper architecture)         │
    │                                                         │
    │  INT8:  8-bit integer                                    │
    │         Weight: 1 byte, Compute: INT8 Tensor Cores       │
    │         Small accuracy loss with calibration             │
    │         2× throughput vs FP16                             │
    │                                                         │
    │  INT4:  4-bit integer (weights only)                     │
    │         Weight: 0.5 bytes, Compute: dequant → FP16       │
    │         Moderate accuracy loss, 4× memory reduction      │
    │         Compute still in FP16 (weight-only quantization) │
    │                                                         │
    │  Memory impact for Llama-70B:                            │
    │                                                         │
    │  Precision    Weight Size    Fits on          Speedup    │
    │  FP16         140 GB         2× H100          1.0×       │
    │  FP8          70 GB          1× H100          ~1.8×      │
    │  INT8         70 GB          1× H100          ~1.8×      │
    │  INT4         35 GB          1× H100          ~2.5×      │
    │  (weight-only quant, KV cache separate)                  │
    └─────────────────────────────────────────────────────────┘

How TensorRT-LLM handles quantization:

    Build time (offline):
    1. Calibrate: run sample data through FP16 model
    2. Compute scale factors per tensor/channel/group
    3. Quantize weights
    4. Build fused kernels with quantized compute paths
    5. Output: optimized TensorRT engine file

    Runtime:
    1. Load quantized engine
    2. Fused kernels do: dequant → compute → quant (all in SRAM!)
    3. No separate dequantization step
    4. Tensor Cores natively compute in FP8/INT8

    ┌─────────────────────────────────────────────┐
    │  FP8 COMPUTE (H100 Tensor Cores):           │
    │                                             │
    │  Input A (FP8) ──┐                          │
    │                   ├── FP8 matmul ── FP16 out │
    │  Input B (FP8) ──┘                          │
    │                                             │
    │  The Tensor Core does:                       │
    │  1. Load FP8 values                          │
    │  2. Multiply in higher precision internally  │
    │  3. Accumulate in FP32                       │
    │  4. Output in FP16 (or FP8 with scaling)     │
    │                                             │
    │  2× more values per byte = 2× throughput     │
    └─────────────────────────────────────────────┘
"""


# ============================================================================
# PART 5: KV CACHE MANAGEMENT
# ============================================================================

"""
TensorRT-LLM's KV cache management:

    Like vLLM, uses paged KV cache to avoid fragmentation.
    Unlike vLLM, the paging is integrated into FUSED attention kernels.

    ┌─────────────────────────────────────────────────────┐
    │  Standard attention: expects contiguous KV cache     │
    │                                                     │
    │  Key cache: [seq_len, num_heads, head_dim]           │
    │  ████████████████████████████████████████            │
    │  ^ contiguous memory block                           │
    │                                                     │
    │  Problem: variable-length sequences waste memory     │
    │  (allocate for max_seq_len, most is unused)          │
    │                                                     │
    │  Paged KV cache: fixed-size blocks                   │
    │                                                     │
    │  Block table for Request 1: [B3, B7, B1, B9]        │
    │  Block table for Request 2: [B2, B5, B8]            │
    │                                                     │
    │  Physical blocks:                                    │
    │  B1[████] B2[████] B3[████] B4[free]                │
    │  B5[████] B6[free] B7[████] B8[██░░]                │
    │  B9[██░░] ...                                       │
    │                                                     │
    │  TensorRT-LLM difference:                            │
    │  The fused attention kernel handles paging internally │
    │  → No separate "gather" step to make KV contiguous   │
    │  → The kernel reads directly from scattered blocks    │
    │  → One less HBM round trip                           │
    └─────────────────────────────────────────────────────┘

    KV cache quantization:
    TensorRT-LLM can also quantize the KV cache to FP8/INT8:

    FP16 KV cache for Llama-70B at 4K context, batch=32:
        80 layers × 2 × 8 × 128 × 4096 × 32 × 2 bytes ≈ 42 GB

    FP8 KV cache:
        Same but 1 byte → ~21 GB (50% reduction!)

    This lets you serve MORE concurrent requests.
"""


# ============================================================================
# PART 6: MULTI-GPU INFERENCE — TENSOR PARALLELISM
# ============================================================================

class TensorParallelLinear:
    """
    Conceptual demonstration of tensor parallelism in TensorRT-LLM.

    For a linear layer y = xW:
        Split W across GPUs along the output dimension.

        ┌─────────────────────────────────────┐
        │  Full weight matrix W: [d_in, d_out] │
        │                                     │
        │  GPU 0: W_0 = W[:, :d_out//N]       │
        │  GPU 1: W_1 = W[:, d_out//N:2*d_out//N] │
        │  ...                                │
        │  GPU N: W_N = W[:, (N-1)*d_out//N:] │
        │                                     │
        │  Each GPU computes: y_i = x @ W_i   │
        │  Then all-gather to get full y       │
        └─────────────────────────────────────┘

    For attention, the split is more clever:
        Q, K, V projections: split output dim (each GPU gets some heads)
        Output projection: split input dim
        This way, only ONE all-reduce per attention layer.

    For FFN:
        Up projection: split output dim (column parallel)
        Down projection: split input dim (row parallel)
        ONE all-reduce per FFN layer.

    Total communication per transformer layer:
        2 all-reduces (1 attention + 1 FFN)
        × data size per all-reduce: batch × seq × d_model × 2 bytes
    """
    pass


"""
    TensorRT-LLM multi-GPU vs vLLM multi-GPU:

    Both use the same TP strategy (Megatron-style, see module 04).
    Difference is in the kernel implementation:

    vLLM:   PyTorch ops → NCCL all-reduce → PyTorch ops
    TRT-LLM: Fused kernel → custom all-reduce → Fused kernel

    TensorRT-LLM's custom all-reduce:
    - Uses NVLink directly (bypasses NCCL overhead for small messages)
    - Fused with the previous computation (no separate kernel launch)
    - For 2-8 GPU TP, this can be 20-30% faster than NCCL

    ┌──────────────────────────────────────────────┐
    │  vLLM all-reduce:                            │
    │  Kernel end → sync → NCCL launch → sync → Kernel start │
    │  ~~~~~~~~~~~~  ~~~~   ~~~~~~~~   ~~~~   ~~~~~~~~~~~~    │
    │  compute      overhead  network   overhead  compute     │
    │                                                        │
    │  TRT-LLM custom all-reduce:                            │
    │  Kernel end+send → receive+Kernel start                │
    │  ~~~~~~~~~~~~~~~~   ~~~~~~~~~~~~~~~~~~~~~~              │
    │  overlapped!        overlapped!                         │
    └──────────────────────────────────────────────┘
"""


# ============================================================================
# PART 7: COMPARISON — vLLM vs SGLang vs TensorRT-LLM
# ============================================================================

"""
    ┌──────────────────────────────────────────────────────────────────┐
    │  DETAILED COMPARISON                                             │
    ├─────────────────┬──────────────┬──────────────┬─────────────────┤
    │  Aspect         │  vLLM        │  SGLang      │  TensorRT-LLM   │
    ├─────────────────┼──────────────┼──────────────┼─────────────────┤
    │  Language       │  Python      │  Python      │  C++ (Python API)│
    ├─────────────────┼──────────────┼──────────────┼─────────────────┤
    │  Compute engine │  PyTorch +   │  PyTorch +   │  TensorRT       │
    │                 │  custom CUDA │  custom CUDA │  (fused kernels) │
    ├─────────────────┼──────────────┼──────────────┼─────────────────┤
    │  Model loading  │  HuggingFace │  HuggingFace │  Build engine    │
    │                 │  (seconds)   │  (seconds)   │  (minutes-hours) │
    ├─────────────────┼──────────────┼──────────────┼─────────────────┤
    │  Flexibility    │  High        │  High        │  Medium          │
    │                 │  (any model) │  (any model) │  (supported only)│
    ├─────────────────┼──────────────┼──────────────┼─────────────────┤
    │  Iteration speed│  Fast        │  Fast        │  Slow (rebuild)  │
    ├─────────────────┼──────────────┼──────────────┼─────────────────┤
    │  Raw throughput │  Good        │  Good+       │  Best            │
    │  (simple prompts│              │              │                  │
    │   large batch)  │              │              │                  │
    ├─────────────────┼──────────────┼──────────────┼─────────────────┤
    │  Complex apps   │  Basic       │  Excellent   │  Basic           │
    │  (multi-turn,   │              │  (DSL, radix)│                  │
    │   branching)    │              │              │                  │
    ├─────────────────┼──────────────┼──────────────┼─────────────────┤
    │  Quantization   │  GPTQ, AWQ   │  GPTQ, AWQ   │  FP8, INT8, INT4│
    │                 │  (weight-only)│ (weight-only)│  (compute-level) │
    ├─────────────────┼──────────────┼──────────────┼─────────────────┤
    │  Hardware       │  NVIDIA,AMD  │  NVIDIA,AMD  │  NVIDIA only     │
    │  support        │  (ROCm)      │  (ROCm)      │                  │
    └─────────────────┴──────────────┴──────────────┴─────────────────┘

    When to use which:

    ┌──────────────────────────────────────────────────────────┐
    │  "I want easy setup and good performance"                │
    │  → vLLM (most popular, good default choice)              │
    ├──────────────────────────────────────────────────────────┤
    │  "I have complex LLM programs with branching/tool use"   │
    │  → SGLang (RadixAttention + DSL = big wins)              │
    ├──────────────────────────────────────────────────────────┤
    │  "I need absolute maximum throughput on NVIDIA GPUs"      │
    │  → TensorRT-LLM (fused kernels, FP8, custom all-reduce) │
    ├──────────────────────────────────────────────────────────┤
    │  "I'm on AMD GPUs"                                       │
    │  → vLLM or SGLang (TRT-LLM is NVIDIA only)              │
    ├──────────────────────────────────────────────────────────┤
    │  "I want to experiment and iterate quickly"               │
    │  → vLLM or SGLang (no build step)                        │
    ├──────────────────────────────────────────────────────────┤
    │  "I'm deploying at scale in production"                   │
    │  → TensorRT-LLM (best TCO if on NVIDIA)                 │
    │    or SGLang (if complex programs)                        │
    └──────────────────────────────────────────────────────────┘
"""


# ============================================================================
# PART 8: DEMO — IN-FLIGHT BATCHING SIMULATION
# ============================================================================

def demo():
    """
    Demonstrate in-flight batching concepts.
    """
    print("=" * 70)
    print("TensorRT-LLM CONCEPTS DEMO")
    print("=" * 70)

    # ── Demo 1: In-Flight Batching ──
    print("\n" + "─" * 70)
    print("DEMO 1: In-Flight Batching Simulation")
    print("─" * 70)

    scheduler = InFlightBatchScheduler(max_batch_size=8, max_tokens_per_batch=512)

    # Add initial requests
    for i in range(4):
        scheduler.add_request(InFlightRequest(
            request_id=i,
            token_ids=list(range(50 + i * 10)),  # varying prompt lengths
            phase="prefill",
            max_tokens=20 + i * 5,
        ))

    print("\nSimulating 15 iterations of in-flight batching:")
    print(f"{'Step':>4} {'Decode':>7} {'Prefill':>8} {'Tokens':>7} {'Waiting':>8} {'Done':>5}")
    print("─" * 45)

    for step in range(15):
        # Add a new request every 3 steps
        if step % 3 == 0 and step > 0:
            scheduler.add_request(InFlightRequest(
                request_id=100 + step,
                token_ids=list(range(40)),
                phase="prefill",
                max_tokens=15,
            ))

        info = scheduler.step()
        print(f"{step:4d} {info['decode_requests']:7d} {info['prefill_requests']:8d} "
              f"{info['total_batch_tokens']:7d} {info['waiting']:8d} "
              f"{len(scheduler.completed):5d}")

    print(f"\nTotal completed: {len(scheduler.completed)} requests")

    # ── Demo 2: Kernel Fusion Impact ──
    print("\n" + "─" * 70)
    print("DEMO 2: Kernel Fusion Impact (Conceptual)")
    print("─" * 70)

    # Simulate HBM round trips
    d_model = 8192
    n_heads = 64
    d_ff = 28672  # Llama-70B FFN
    bytes_per_elem = 2  # FP16

    unfused_reads = (
        d_model +  # LayerNorm read
        d_model +  # Q proj read
        d_model +  # K proj read
        d_model +  # V proj read
        d_model * 3 +  # attention reads
        d_model +  # output proj read
        d_model +  # residual read
        d_model +  # LayerNorm read
        d_model +  # FFN up read
        d_ff +     # GELU read
        d_ff +     # FFN down read
        d_model    # residual read
    )

    fused_reads = (
        d_model +      # Fused LayerNorm+QKV read
        d_model * 3 +  # FlashAttention (Q,K,V)
        d_model * 2 +  # Fused output+residual+LN
        d_model + d_ff  # Fused FFN
    )

    print(f"\n  Unfused HBM reads per layer: {unfused_reads * bytes_per_elem / 1024:.0f} KB")
    print(f"  Fused HBM reads per layer:   {fused_reads * bytes_per_elem / 1024:.0f} KB")
    print(f"  Reduction: {(1 - fused_reads/unfused_reads)*100:.0f}%")
    print(f"  For 80 layers: {(unfused_reads - fused_reads) * 80 * bytes_per_elem / 1024**2:.0f} MB saved per token")

    # ── Demo 3: Quantization Memory Impact ──
    print("\n" + "─" * 70)
    print("DEMO 3: Quantization Memory Impact")
    print("─" * 70)

    param_count = 70e9  # 70B

    precisions = {
        "FP16":  2.0,
        "FP8":   1.0,
        "INT8":  1.0,
        "INT4":  0.5,
    }

    print(f"\n  Llama-70B weight memory:")
    for name, bytes_per_param in precisions.items():
        size_gb = param_count * bytes_per_param / 1e9
        h100s = math.ceil(size_gb / 80)
        print(f"  {name:5s}: {size_gb:6.1f} GB → {h100s} H100(s) needed")

    # KV cache impact
    print(f"\n  KV cache for batch=32, seq=4096:")
    n_layers = 80
    n_kv_heads = 8  # GQA
    head_dim = 128

    for name, bpp in [("FP16", 2.0), ("FP8", 1.0), ("INT8", 1.0)]:
        kv_size = n_layers * 2 * n_kv_heads * head_dim * 4096 * 32 * bpp
        print(f"  {name:5s} KV cache: {kv_size / 1e9:.1f} GB")

    print(f"\n{'=' * 70}")
    print("KEY TAKEAWAYS:")
    print("  1. TensorRT-LLM's kernel fusion reduces HBM round trips by ~50%+")
    print("  2. In-flight batching mixes prefill+decode for better GPU utilization")
    print("  3. FP8 quantization: 2× memory reduction with minimal accuracy loss")
    print("  4. Custom all-reduce beats NCCL for small TP groups")
    print("  5. Best raw throughput, but less flexible than vLLM/SGLang")
    print("=" * 70)


if __name__ == "__main__":
    demo()

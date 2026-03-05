"""
FSDP and ZeRO: Fully Sharded Data Parallelism for Memory-Efficient Training

This module explains ZeRO (Zero Redundancy Optimizer) and PyTorch FSDP
from the ground up with working implementations.
Based on:
  - "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
    (Rajbhandari et al., 2020)
  - PyTorch FSDP (Fully Sharded Data Parallel)

================================================================================
PART 1: THE MEMORY PROBLEM — WHERE DOES ALL THE MEMORY GO?
================================================================================

Let's do exact math for training a 7B parameter model (like LLaMA-7B):

    Model parameters: 7 billion (7 × 10^9)

    In mixed-precision training (the standard approach):

    ┌────────────────────────────────────────────────────────────┐
    │  MEMORY BREAKDOWN (per GPU, with standard Data Parallelism)│
    │                                                            │
    │  1. FP16 Parameters (weights):                             │
    │     7B × 2 bytes = 14 GB                                   │
    │                                                            │
    │  2. FP16 Gradients:                                        │
    │     7B × 2 bytes = 14 GB                                   │
    │                                                            │
    │  3. Optimizer States (Adam):                               │
    │     - FP32 copy of parameters: 7B × 4 bytes = 28 GB       │
    │     - FP32 momentum (m):       7B × 4 bytes = 28 GB       │
    │     - FP32 variance (v):       7B × 4 bytes = 28 GB       │
    │     Subtotal:                  84 GB                       │
    │                                                            │
    │  TOTAL MODEL STATE: 14 + 14 + 84 = 112 GB                 │
    │                                                            │
    │  4. Activations (depends on batch size, seq length):       │
    │     For batch=4, seq=2048: ~20-40 GB                       │
    │                                                            │
    │  5. Temporary buffers, fragmentation: ~5-10 GB             │
    │                                                            │
    │  GRAND TOTAL: ~140-160 GB per GPU                          │
    │                                                            │
    │  A100 GPU: 80 GB                                           │
    │  → DOESN'T FIT on a single GPU!                            │
    └────────────────────────────────────────────────────────────┘

The shocking part: OPTIMIZER STATES are 75% of model state memory!

    Parameters:   14 GB  (12.5%)
    Gradients:    14 GB  (12.5%)
    Optimizer:    84 GB  (75.0%)  ← THIS IS THE PROBLEM
                 ─────
    Total:       112 GB

And with standard Data Parallelism (DP), EVERY GPU has a COMPLETE COPY
of everything! With 8 GPUs, that's 8 × 112 = 896 GB of redundant state.

    ┌─────────────────────────────────────────────────────────┐
    │  Standard Data Parallelism (DP) with 4 GPUs:            │
    │                                                         │
    │  GPU 0          GPU 1          GPU 2          GPU 3     │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐│
    │  │Params 14G│  │Params 14G│  │Params 14G│  │Params 14G││
    │  │Grads  14G│  │Grads  14G│  │Grads  14G│  │Grads  14G││
    │  │OptSt  84G│  │OptSt  84G│  │OptSt  84G│  │OptSt  84G││
    │  └──────────┘  └──────────┘  └──────────┘  └──────────┘│
    │  112 GB each = 448 GB total                             │
    │  But only 112 GB of UNIQUE data!                        │
    │  Redundancy: 4× = 336 GB WASTED                         │
    └─────────────────────────────────────────────────────────┘

ZeRO's insight: SHARD the redundant state across GPUs.


================================================================================
PART 2: ZeRO STAGE 1 — PARTITION OPTIMIZER STATES
================================================================================

Instead of every GPU storing all optimizer states, each GPU stores only 1/N.

    ┌─────────────────────────────────────────────────────────┐
    │  ZeRO Stage 1 with 4 GPUs:                              │
    │                                                         │
    │  GPU 0          GPU 1          GPU 2          GPU 3     │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐│
    │  │Params 14G│  │Params 14G│  │Params 14G│  │Params 14G││
    │  │Grads  14G│  │Grads  14G│  │Grads  14G│  │Grads  14G││
    │  │OptSt  21G│  │OptSt  21G│  │OptSt  21G│  │OptSt  21G││
    │  │(1/4)     │  │(1/4)     │  │(1/4)     │  │(1/4)     ││
    │  └──────────┘  └──────────┘  └──────────┘  └──────────┘│
    │  49 GB each (down from 112!)                            │
    │                                                         │
    │  Savings: 84 - 21 = 63 GB per GPU                       │
    │  Memory per GPU: 14 + 14 + 21 = 49 GB ← FITS on A100!  │
    └─────────────────────────────────────────────────────────┘

How it works:
    1. Forward pass: same as DP (all GPUs have all parameters)
    2. Backward pass: same as DP (all GPUs compute all gradients)
    3. All-reduce gradients: same as DP
    4. Optimizer step: each GPU updates only ITS partition of parameters
    5. All-gather: broadcast updated parameters to all GPUs

Communication cost: same as standard DP (all-reduce) + one all-gather.
The all-gather is small relative to training time.


================================================================================
PART 3: ZeRO STAGE 2 — PARTITION GRADIENTS TOO
================================================================================

    ┌─────────────────────────────────────────────────────────┐
    │  ZeRO Stage 2 with 4 GPUs:                              │
    │                                                         │
    │  GPU 0          GPU 1          GPU 2          GPU 3     │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐│
    │  │Params 14G│  │Params 14G│  │Params 14G│  │Params 14G││
    │  │Grads 3.5G│  │Grads 3.5G│  │Grads 3.5G│  │Grads 3.5G││
    │  │(1/4)     │  │(1/4)     │  │(1/4)     │  │(1/4)     ││
    │  │OptSt  21G│  │OptSt  21G│  │OptSt  21G│  │OptSt  21G││
    │  │(1/4)     │  │(1/4)     │  │(1/4)     │  │(1/4)     ││
    │  └──────────┘  └──────────┘  └──────────┘  └──────────┘│
    │  38.5 GB each (down from 49!)                           │
    └─────────────────────────────────────────────────────────┘

Key insight: during backward pass, once a gradient is REDUCED and sent
to the GPU that owns it, OTHER GPUs can FREE that gradient memory.

    How it works:
    1. Forward: same (all GPUs have all params)
    2. Backward: compute gradients, but use REDUCE-SCATTER instead of ALL-REDUCE
       - Reduce-scatter: each GPU receives the REDUCED gradient
         for only its partition
       - Other GPUs discard that partition's gradient → memory freed!
    3. Optimizer step: each GPU updates its partition
    4. All-gather: broadcast updated params

Communication: replace all-reduce with reduce-scatter + all-gather.
    All-reduce = reduce-scatter + all-gather (same total volume!)
    So ZeRO Stage 2 has the SAME communication cost as standard DP.


================================================================================
PART 4: ZeRO STAGE 3 — PARTITION EVERYTHING (FULL SHARDING)
================================================================================

    ┌─────────────────────────────────────────────────────────┐
    │  ZeRO Stage 3 with 4 GPUs:                              │
    │                                                         │
    │  GPU 0          GPU 1          GPU 2          GPU 3     │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐│
    │  │Params3.5G│  │Params3.5G│  │Params3.5G│  │Params3.5G││
    │  │(1/4)     │  │(1/4)     │  │(1/4)     │  │(1/4)     ││
    │  │Grads 3.5G│  │Grads 3.5G│  │Grads 3.5G│  │Grads 3.5G││
    │  │(1/4)     │  │(1/4)     │  │(1/4)     │  │(1/4)     ││
    │  │OptSt  21G│  │OptSt  21G│  │OptSt  21G│  │OptSt  21G││
    │  │(1/4)     │  │(1/4)     │  │(1/4)     │  │(1/4)     ││
    │  └──────────┘  └──────────┘  └──────────┘  └──────────┘│
    │  28 GB each (down from 112!)                            │
    │  4× reduction in memory!                                │
    │                                                         │
    │  With N GPUs: memory per GPU = 112/N GB                  │
    │  With 64 GPUs: 112/64 = 1.75 GB per GPU (for 7B model!) │
    │  → Can train MUCH larger models                          │
    └─────────────────────────────────────────────────────────┘

Now parameters are also sharded! This means:
    - Before each layer's forward pass: ALL-GATHER to reconstruct full params
    - After each layer's forward pass: discard the non-owned params
    - Before each layer's backward pass: ALL-GATHER again
    - After each layer's backward pass: REDUCE-SCATTER gradients

Communication cost:
    Forward: one all-gather per layer
    Backward: one all-gather + one reduce-scatter per layer
    = 3× the communication of standard DP per layer

    But: communication can be OVERLAPPED with computation!
    While layer L is computing, layer L+1's params are being gathered.

    ┌──────────────────────────────────────────────────────┐
    │  ZeRO-3 Communication Timeline:                      │
    │                                                      │
    │  Compute:  [Layer1 fwd][Layer2 fwd][Layer3 fwd]...   │
    │  Comms:    [AG L2][AG L3]...(overlapped!)             │
    │                                                      │
    │  AG = All-Gather (prefetch next layer's params)       │
    │  RS = Reduce-Scatter (send gradients after backward)  │
    │                                                      │
    │  With good overlap, the extra communication is hidden │
    │  behind computation. Effective slowdown: ~10-15%      │
    └──────────────────────────────────────────────────────┘

This is what PyTorch FSDP implements!


================================================================================
PART 5: COMMUNICATION PATTERNS — ALL-GATHER AND REDUCE-SCATTER
================================================================================

Let's understand the two key operations:

    ALL-GATHER: Each GPU has a shard → every GPU gets all shards

        Before:
        GPU 0: [A]        GPU 1: [B]        GPU 2: [C]        GPU 3: [D]

        After:
        GPU 0: [A,B,C,D]  GPU 1: [A,B,C,D]  GPU 2: [A,B,C,D]  GPU 3: [A,B,C,D]

        Data moved: (N-1)/N × data_size per GPU
        For our 7B model, one layer (~100M params):
            (3/4) × 200 MB = 150 MB per GPU

    REDUCE-SCATTER: Each GPU has full data → each gets REDUCED shard

        Before (each GPU has full gradient):
        GPU 0: [g0,g1,g2,g3]  GPU 1: [g0,g1,g2,g3]  ...

        After (each GPU has sum of its shard):
        GPU 0: [Σg0]          GPU 1: [Σg1]          GPU 2: [Σg2]  GPU 3: [Σg3]

        Data moved: (N-1)/N × data_size per GPU (same as all-gather)

    Total communication per training step (ZeRO-3):
        Forward:  L × all-gather   = L × (N-1)/N × P/L = (N-1)/N × P
        Backward: L × all-gather   = (N-1)/N × P
                  L × reduce-scatter = (N-1)/N × P
        Total: 3 × (N-1)/N × P ≈ 3P for large N

        Standard DP (all-reduce only):
        Total: 2 × (N-1)/N × P ≈ 2P

        So ZeRO-3 uses ~1.5× more communication than standard DP.
        In exchange: N× less memory!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import math
import copy


# ============================================================================
# PART 6: WORKING IMPLEMENTATION — TOY ZeRO
# ============================================================================

class ShardedParameter:
    """
    A parameter that is sharded across multiple (simulated) GPUs.

    In real FSDP, each GPU stores only 1/N of each parameter.
    Here we simulate this with slices of a tensor.

    Full parameter: [p0, p1, p2, p3, p4, p5, p6, p7]
    With 4 GPUs:
        GPU 0 stores: [p0, p1]
        GPU 1 stores: [p2, p3]
        GPU 2 stores: [p4, p5]
        GPU 3 stores: [p6, p7]
    """
    def __init__(self, full_param: torch.Tensor, num_gpus: int, gpu_id: int):
        self.num_gpus = num_gpus
        self.gpu_id = gpu_id
        self.full_size = full_param.shape

        # Flatten and shard
        flat = full_param.detach().flatten()
        # Pad to be evenly divisible
        pad_size = (num_gpus - len(flat) % num_gpus) % num_gpus
        if pad_size:
            flat = torch.cat([flat, torch.zeros(pad_size)])

        shard_size = len(flat) // num_gpus
        self.shard = flat[gpu_id * shard_size:(gpu_id + 1) * shard_size].clone()
        self.shard.requires_grad_(True)
        self.shard_size = shard_size
        self.padded_size = len(flat)

    def all_gather(self, all_shards: List[torch.Tensor]) -> torch.Tensor:
        """
        Simulate all-gather: collect shards from all GPUs.

        In real FSDP, this is torch.distributed.all_gather().
        Here we just concatenate the shards.

        Communication cost: (N-1)/N × shard_size × N = (N-1) × shard_size
        """
        gathered = torch.cat(all_shards)
        # Unpad and reshape
        return gathered[:math.prod(self.full_size)].reshape(self.full_size)

    def reduce_scatter(self, full_gradient: torch.Tensor,
                       all_full_gradients: List[torch.Tensor]) -> torch.Tensor:
        """
        Simulate reduce-scatter: sum gradients, each GPU gets its shard.

        In real FSDP, this is torch.distributed.reduce_scatter().

        Communication cost: (N-1)/N × full_size
        """
        # Sum all gradients
        summed = torch.stack(all_full_gradients).sum(dim=0)
        # Flatten and extract this GPU's shard
        flat = summed.flatten()
        if len(flat) < self.padded_size:
            flat = torch.cat([flat, torch.zeros(self.padded_size - len(flat))])
        return flat[self.gpu_id * self.shard_size:(self.gpu_id + 1) * self.shard_size]


class ToyZeROStage3:
    """
    A toy implementation of ZeRO Stage 3 / FSDP.

    Demonstrates the key concepts:
    1. Parameters are sharded across GPUs
    2. All-gather before forward/backward to reconstruct params
    3. Reduce-scatter after backward to distribute gradients
    4. Each GPU only updates its shard with its local optimizer state

    This is a SIMULATION — we don't actually use multiple GPUs.
    Instead, we maintain separate state for each "simulated GPU."
    """
    def __init__(self, model: nn.Module, num_gpus: int = 4):
        self.num_gpus = num_gpus
        self.model_template = model  # Original model (for shape reference)

        # Create sharded parameters for each GPU
        self.gpu_shards: List[Dict[str, ShardedParameter]] = []
        for gpu_id in range(num_gpus):
            shards = {}
            for name, param in model.named_parameters():
                shards[name] = ShardedParameter(param.data, num_gpus, gpu_id)
            self.gpu_shards.append(shards)

        # Create optimizer states for each GPU (only for its shards)
        self.optimizer_states: List[Dict[str, dict]] = []
        for gpu_id in range(num_gpus):
            states = {}
            for name in self.gpu_shards[gpu_id]:
                shard = self.gpu_shards[gpu_id][name]
                states[name] = {
                    "m": torch.zeros_like(shard.shard),  # momentum
                    "v": torch.zeros_like(shard.shard),  # variance
                    "step": 0,
                }
            self.optimizer_states.append(states)

        self.communication_volume = 0  # Track total bytes communicated

    def memory_per_gpu(self) -> dict:
        """Calculate memory usage per GPU."""
        param_bytes = 0
        grad_bytes = 0
        opt_bytes = 0

        # Using GPU 0 as representative
        for name, shard in self.gpu_shards[0].items():
            param_bytes += shard.shard.numel() * shard.shard.element_size()
            grad_bytes += shard.shard.numel() * 2  # FP16 gradients
            opt_bytes += shard.shard.numel() * 4 * 3  # FP32: params + m + v

        return {
            "parameters_mb": param_bytes / 1024**2,
            "gradients_mb": grad_bytes / 1024**2,
            "optimizer_mb": opt_bytes / 1024**2,
            "total_mb": (param_bytes + grad_bytes + opt_bytes) / 1024**2,
        }

    def forward_pass(self, model: nn.Module, inputs: torch.Tensor,
                     gpu_id: int) -> torch.Tensor:
        """
        Forward pass for one GPU.

        1. ALL-GATHER: reconstruct full parameters from all shards
        2. Load full parameters into model
        3. Forward pass
        4. Discard non-owned parameters (in real FSDP, free the memory)
        """
        # Step 1: All-gather each parameter
        for name, param in model.named_parameters():
            all_shards = [self.gpu_shards[g][name].shard for g in range(self.num_gpus)]
            full_param = self.gpu_shards[gpu_id][name].all_gather(all_shards)
            param.data = full_param

            # Track communication
            shard_size = self.gpu_shards[gpu_id][name].shard_size
            self.communication_volume += shard_size * (self.num_gpus - 1) * 4

        # Step 2: Forward pass
        output = model(inputs)
        return output

    def backward_pass(self, loss: torch.Tensor, model: nn.Module,
                      gpu_id: int):
        """
        Backward pass for one GPU.

        1. ALL-GATHER params (needed again for backward)
        2. Compute gradients
        3. REDUCE-SCATTER: each GPU gets reduced gradient for its shard
        4. Free non-owned gradients
        """
        loss.backward()

        # Reduce-scatter gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Simulate: all GPUs have their local gradient
                # In reality, each GPU computed grads on different data
                # Here we simulate by using the same grad (simplified)
                all_grads = [param.grad.clone() for _ in range(self.num_gpus)]
                reduced_shard = self.gpu_shards[gpu_id][name].reduce_scatter(
                    param.grad, all_grads
                )
                # Store reduced gradient shard
                self.gpu_shards[gpu_id][name].shard.grad = reduced_shard / self.num_gpus

                # Track communication
                self.communication_volume += param.grad.numel() * 4

    def optimizer_step(self, gpu_id: int, lr: float = 1e-3,
                       beta1: float = 0.9, beta2: float = 0.999,
                       eps: float = 1e-8):
        """
        Adam optimizer step — each GPU only updates its shard.

        This is where the memory savings come from:
        Each GPU only needs optimizer states for 1/N of the parameters.
        """
        for name, shard_param in self.gpu_shards[gpu_id].items():
            if shard_param.shard.grad is None:
                continue

            state = self.optimizer_states[gpu_id][name]
            state["step"] += 1
            t = state["step"]

            grad = shard_param.shard.grad.float()
            param = shard_param.shard.float()

            # Adam update (only on this GPU's shard!)
            state["m"] = beta1 * state["m"] + (1 - beta1) * grad
            state["v"] = beta2 * state["v"] + (1 - beta2) * grad ** 2

            m_hat = state["m"] / (1 - beta1 ** t)
            v_hat = state["v"] / (1 - beta2 ** t)

            param = param - lr * m_hat / (torch.sqrt(v_hat) + eps)
            shard_param.shard.data = param.half() if shard_param.shard.dtype == torch.float16 else param


# ============================================================================
# PART 7: COMPARISON WITH MEGATRON-LM
# ============================================================================

"""
    ┌──────────────────────────────────────────────────────────────────┐
    │  ZeRO/FSDP vs Megatron-LM (covered in module 04):              │
    │                                                                  │
    │  Approach       │ Memory      │ Communication  │ Complexity      │
    │ ────────────────┼─────────────┼────────────────┼────────────────│
    │  Data Parallel  │ Full copy   │ 2P (all-reduce)│ Low            │
    │  (baseline)     │ per GPU     │                │                │
    │ ────────────────┼─────────────┼────────────────┼────────────────│
    │  ZeRO-1        │ Opt/N       │ 2P + P         │ Low            │
    │                 │ params+grad │ (AR + AG)      │                │
    │ ────────────────┼─────────────┼────────────────┼────────────────│
    │  ZeRO-2        │ Opt+Grad/N  │ 2P             │ Low            │
    │                 │ params full │ (RS + AG)      │                │
    │ ────────────────┼─────────────┼────────────────┼────────────────│
    │  ZeRO-3/FSDP   │ Everything/N│ 3P             │ Medium         │
    │                 │             │ (AG+AG+RS)     │                │
    │ ────────────────┼─────────────┼────────────────┼────────────────│
    │  Megatron TP    │ Params/N    │ 2 AR per layer │ High           │
    │  (tensor par.)  │ per layer   │ (latency-bound)│ (model changes)│
    │ ────────────────┼─────────────┼────────────────┼────────────────│
    │  Megatron PP    │ Layers/N    │ Point-to-point │ High           │
    │  (pipeline par.)│             │ (bubbles!)     │ (scheduling)   │
    └──────────────────────────────────────────────────────────────────┘

    When to use which:

    Small models (< 10B): ZeRO-2 or FSDP
        → Simple, good memory savings, no model changes

    Medium models (10-70B): FSDP (ZeRO-3)
        → Full sharding needed, still simple to use

    Large models (70B+): Megatron-LM (TP + PP) + FSDP
        → Need all three parallelism dimensions
        → Megatron reduces latency (TP within node)
        → FSDP reduces memory (across nodes)

    The modern recipe (Llama-3 training):
        Intra-node (8 GPUs): Tensor Parallelism (Megatron-style)
        Inter-node: FSDP (ZeRO-3 style)
        This combines the low-latency of TP with memory efficiency of FSDP.
"""


# ============================================================================
# PART 8: MIXED PRECISION TRAINING INTEGRATION
# ============================================================================

"""
FSDP + Mixed Precision:

    The standard approach for large model training:

    ┌─────────────────────────────────────────────────────────┐
    │  Mixed Precision + FSDP:                                │
    │                                                         │
    │  1. Parameters sharded in FP16/BF16 (2 bytes each)      │
    │  2. All-gather: reconstruct FP16 full params            │
    │  3. Forward pass: FP16 compute                          │
    │  4. Loss scaling (prevent underflow in FP16 gradients)  │
    │  5. Backward pass: FP16 compute, FP16 gradients         │
    │  6. Reduce-scatter: aggregate FP16 gradient shards      │
    │  7. Cast gradient shard to FP32                          │
    │  8. Adam step in FP32 (momentum, variance, update)      │
    │  9. Cast updated shard back to FP16                      │
    │  10. Ready for next iteration                            │
    │                                                         │
    │  Memory per GPU (7B model, 4 GPUs):                     │
    │    FP16 param shard:  14/4 = 3.5 GB × 0.5 = 1.75 GB    │
    │    Wait... FP16 is 2 bytes, so: 7B/4 × 2B = 3.5 GB     │
    │    FP32 optimizer:    7B/4 × (4+4+4) = 21 GB            │
    │    FP16 grad shard:   7B/4 × 2 = 3.5 GB                │
    │    Total model state: ~28 GB per GPU                     │
    │    + Activations: ~10-20 GB                              │
    │    Fits comfortably on 80 GB A100!                       │
    └─────────────────────────────────────────────────────────┘

    BF16 vs FP16 for training:

    FP16: 5 exponent bits, 10 mantissa bits
        Range: ±65504, precision: ~3 decimal digits
        Problem: often OVERFLOWS during training!
        Needs: loss scaling to avoid gradient underflow

    BF16: 8 exponent bits, 7 mantissa bits
        Range: ±3.4×10^38 (same as FP32!), precision: ~2 decimal digits
        No overflow problem! Less precision, but training is robust.
        Used by: Llama, GPT-4, most modern models

    Recommended: BF16 parameters + FP32 optimizer states
"""


# ============================================================================
# PART 9: DEMO
# ============================================================================

class ToyModel(nn.Module):
    """A simple model for demonstrating FSDP sharding."""
    def __init__(self, d_in: int = 512, d_hidden: int = 2048, d_out: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.fc3 = nn.Linear(d_hidden, d_out)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gelu(self.fc1(x))
        x = self.gelu(self.fc2(x))
        return self.fc3(x)


def demo():
    """
    Demonstrate ZeRO Stage 3 / FSDP memory savings with toy model.
    """
    print("=" * 70)
    print("FSDP / ZeRO STAGE 3 DEMO")
    print("=" * 70)

    # ── Model Setup ──
    torch.manual_seed(42)
    model = ToyModel(d_in=512, d_hidden=2048, d_out=10)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params:,} parameters")
    print(f"Full model size: {total_params * 4 / 1024**2:.1f} MB (FP32)")

    # ── Memory comparison across strategies ──
    print(f"\n{'─' * 70}")
    print("MEMORY COMPARISON (FP32 training, Adam optimizer)")
    print(f"{'─' * 70}")

    param_bytes = total_params * 4  # FP32
    grad_bytes = total_params * 4   # FP32
    opt_bytes = total_params * 4 * 3  # m + v + FP32 copy

    for num_gpus in [1, 2, 4, 8]:
        dp_mem = (param_bytes + grad_bytes + opt_bytes) / 1024**2
        z1_mem = (param_bytes + grad_bytes + opt_bytes / num_gpus) / 1024**2
        z2_mem = (param_bytes + grad_bytes / num_gpus + opt_bytes / num_gpus) / 1024**2
        z3_mem = (param_bytes / num_gpus + grad_bytes / num_gpus + opt_bytes / num_gpus) / 1024**2

        print(f"\n  {num_gpus} GPU(s):")
        print(f"    Standard DP:  {dp_mem:8.1f} MB  (no savings)")
        print(f"    ZeRO Stage 1: {z1_mem:8.1f} MB  ({(1-z1_mem/dp_mem)*100:.0f}% saved)")
        print(f"    ZeRO Stage 2: {z2_mem:8.1f} MB  ({(1-z2_mem/dp_mem)*100:.0f}% saved)")
        print(f"    ZeRO Stage 3: {z3_mem:8.1f} MB  ({(1-z3_mem/dp_mem)*100:.0f}% saved)")

    # ── Simulate ZeRO-3 Training Step ──
    print(f"\n{'─' * 70}")
    print("SIMULATING ZeRO-3 TRAINING STEP (4 GPUs)")
    print(f"{'─' * 70}")

    num_gpus = 4
    zero3 = ToyZeROStage3(model, num_gpus=num_gpus)

    mem = zero3.memory_per_gpu()
    print(f"\n  Memory per GPU (sharded):")
    print(f"    Parameters:     {mem['parameters_mb']:.2f} MB")
    print(f"    Gradients:      {mem['gradients_mb']:.2f} MB")
    print(f"    Optimizer:      {mem['optimizer_mb']:.2f} MB")
    print(f"    Total:          {mem['total_mb']:.2f} MB")
    print(f"    (vs {total_params * 4 * 5 / 1024**2:.2f} MB without sharding)")

    # Simulate a training step
    inputs = torch.randn(8, 512)
    targets = torch.randint(0, 10, (8,))

    for gpu_id in range(num_gpus):
        # Each GPU gets its slice of the batch
        batch_start = gpu_id * 2
        batch_end = batch_start + 2
        gpu_inputs = inputs[batch_start:batch_end]
        gpu_targets = targets[batch_start:batch_end]

        # Forward (with all-gather)
        gpu_model = copy.deepcopy(model)
        output = zero3.forward_pass(gpu_model, gpu_inputs, gpu_id)
        loss = F.cross_entropy(output, gpu_targets)

        # Backward (with reduce-scatter)
        zero3.backward_pass(loss, gpu_model, gpu_id)

        # Optimizer step (only on local shard)
        zero3.optimizer_step(gpu_id, lr=1e-3)

        if gpu_id == 0:
            print(f"\n  GPU {gpu_id}: loss = {loss.item():.4f}")

    comm_mb = zero3.communication_volume / 1024**2
    print(f"\n  Total communication: {comm_mb:.2f} MB")

    # ── Scale to real models ──
    print(f"\n{'─' * 70}")
    print("SCALING TO REAL MODELS")
    print(f"{'─' * 70}")

    models = [
        ("LLaMA-7B", 7e9),
        ("LLaMA-13B", 13e9),
        ("LLaMA-70B", 70e9),
        ("GPT-3 175B", 175e9),
    ]

    for name, params in models:
        # Mixed precision: FP16 params + FP32 optimizer
        full_mem = (params * 2 + params * 2 + params * 12) / 1e9  # FP16 p+g + FP32 opt
        for gpus in [8, 64, 256]:
            z3_mem = full_mem / gpus
            fits = "✓" if z3_mem < 70 else "✗"  # 70 GB usable on 80 GB A100
            print(f"  {name:15s} on {gpus:3d} GPUs (ZeRO-3): "
                  f"{z3_mem:6.1f} GB/GPU {fits}")

    print(f"\n{'=' * 70}")
    print("KEY TAKEAWAYS:")
    print("  1. Optimizer states are 75% of training memory — shard them first")
    print("  2. ZeRO-3/FSDP shards everything: N GPUs → 1/N memory per GPU")
    print("  3. Communication overhead: ~1.5× vs standard DP, mostly overlapped")
    print("  4. FSDP = simple API, works with any model (unlike Megatron TP)")
    print("  5. Modern recipe: TP intra-node + FSDP inter-node")
    print("=" * 70)


if __name__ == "__main__":
    demo()

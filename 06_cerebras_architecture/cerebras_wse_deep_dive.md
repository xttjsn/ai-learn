# Cerebras Wafer-Scale Engine: A Deep Dive

## From First Principles — Why Rethink the GPU?

Based on:
- Cerebras WSE-1 (2019), WSE-2 (2021), WSE-3 (2024)
- CS-3 System Architecture
- Weight Streaming: An Efficient Execution Model for Large Language Models

---

## PART 1: THE FUNDAMENTAL PROBLEM WITH GPU CLUSTERS

Let's start with what happens when you train a large model on GPUs:

```
A single NVIDIA H100 GPU:
    Compute:    ~990 TFLOPS (FP16 with sparsity)
    Memory:     80 GB HBM3
    Bandwidth:  3.35 TB/s (HBM)
    On-chip:    ~50 MB SRAM total (across all SMs)

The memory hierarchy problem:

    ┌──────────────────────────────┐
    │   SRAM (registers + shared)  │
    │   ~50 MB total               │
    │   ~19 TB/s bandwidth         │  ← WHERE COMPUTE HAPPENS
    └──────────────┬───────────────┘
                   │  bottleneck #1: HBM bandwidth
    ┌──────────────┴───────────────┐
    │   HBM3 (on-chip but off-die) │
    │   80 GB                      │
    │   3.35 TB/s                  │  ← WHERE WEIGHTS LIVE
    └──────────────┬───────────────┘
                   │  bottleneck #2: NVLink / network
    ┌──────────────┴───────────────┐
    │   Other GPUs (via NVLink)    │
    │   900 GB/s per link          │  ← WHERE OTHER SHARDS LIVE
    └──────────────┬───────────────┘
                   │  bottleneck #3: network fabric
    ┌──────────────┴───────────────┐
    │   Other NODES (via InfiniBand)│
    │   400 Gb/s = 50 GB/s         │  ← WHERE MOST SHARDS LIVE
    └──────────────────────────────┘

Key insight: MOST of the GPU's time is spent WAITING for data.

    Arithmetic Intensity = FLOPs / Bytes moved

    Matrix multiply:   high arithmetic intensity → compute-bound ✓
    Attention softmax:  low arithmetic intensity  → memory-bound ✗
    LayerNorm:          low arithmetic intensity  → memory-bound ✗
    Activation funcs:   low arithmetic intensity  → memory-bound ✗

And for multi-GPU training, add communication overhead:
    - Tensor parallelism: all-reduce after every layer
    - Pipeline parallelism: bubble overhead, complex scheduling
    - Data parallelism: gradient all-reduce every step

For a 70B model on 8×H100:
    ~30-40% of time is communication overhead
    ~20-30% is memory-bound operations
    ~30-40% is actual useful compute

    MFU (Model FLOPs Utilization) ≈ 35-45% typically
```

The GPU was designed as a general-purpose parallel processor. It's amazing at
what it does, but its architecture has fundamental constraints for LLM workloads:

1. **Limited on-chip memory**: 50 MB SRAM vs 80 GB HBM — a 1600× gap
2. **HBM bandwidth wall**: 3.35 TB/s sounds fast but can't keep up with 990 TFLOPS
3. **Multi-GPU communication**: splitting a model across GPUs adds latency at every layer
4. **Complex parallelism**: you need TP + PP + DP + expert parallelism = engineering nightmare

What if you could build a chip where ALL the memory was on-chip SRAM?

---

## PART 2: THE WAFER-SCALE ENGINE (WSE-3) ARCHITECTURE

### The Radical Idea: Use the Entire Wafer

Normal chip manufacturing:

```
    Silicon wafer (300mm / ~12 inches diameter)
    ┌─────────────────────────────────────┐
    │  ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐   │
    │  │H │ │H │ │H │ │H │ │H │ │H │   │
    │  │100│ │100│ │100│ │100│ │100│ │100│   │
    │  └──┘ └──┘ └──┘ └──┘ └──┘ └──┘   │
    │  ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐   │
    │  │H │ │H │ │H │ │H │ │H │ │H │   │
    │  │100│ │100│ │100│ │100│ │100│ │100│   │
    │  └──┘ └──┘ └──┘ └──┘ └──┘ └──┘   │
    │         ... more chips ...          │
    │  Step: dice wafer → individual chips │
    └─────────────────────────────────────┘

    An H100 die: 814 mm²   (one of the largest GPU dies)
    Wafer area:  ~70,000 mm²
    Yield: cut ~60-80 good chips per wafer
```

Cerebras approach: DON'T cut the wafer. Use the ENTIRE thing as one chip.

```
    Cerebras WSE-3:
    ┌─────────────────────────────────────┐
    │                                     │
    │          ENTIRE WAFER               │
    │          = ONE CHIP                 │
    │                                     │
    │    46,225 mm² die area              │
    │    (56× larger than H100)           │
    │                                     │
    │    900,000 AI-optimized cores       │
    │    44 GB on-chip SRAM               │
    │    21 PB/s memory bandwidth         │
    │                                     │
    └─────────────────────────────────────┘

    Comparison:
                        H100            WSE-3
    ─────────────────────────────────────────────
    Die area            814 mm²         46,225 mm²     (57×)
    Cores               16,896 CUDA     900,000        (53×)
    On-chip SRAM        50 MB           44 GB          (880×)
    Memory bandwidth    3.35 TB/s       21 PB/s        (6,268×)
    HBM                 80 GB           NONE (0 GB)
    Transistors         80B             4 trillion
    Process node        4nm (TSMC)      5nm (TSMC)
```

### How Do They Handle Defects?

On a wafer this large, defects are guaranteed. Cerebras solves this with:

```
    Redundancy Architecture:
    ┌────┬────┬────┬────┬────┐
    │core│core│core│SPARE│core│   ← spare cores replace defective ones
    ├────┼────┼────┼────┼────┤
    │core│ XX │core│core│core│   ← XX = defective core (bypassed)
    ├────┼────┼────┼────┼────┤
    │core│core│core│core│core│
    └────┴────┴────┴────┴────┘

    - ~1-2% of cores are spares
    - Routing fabric works around dead cores
    - Each core is independent: no global synchronization needed
    - The fabric is designed so that any core can be disabled
      without affecting its neighbors
```

### Core Architecture

Each of the 900,000 cores is a simple, efficient processor:

```
    Single WSE-3 Core:
    ┌─────────────────────────────┐
    │   48 KB SRAM                │   ← ALL memory is SRAM (no cache hierarchy!)
    │   ┌─────────┐ ┌─────────┐  │
    │   │ Tensor   │ │ Scalar  │  │
    │   │ compute  │ │ compute │  │
    │   │ (FMAC)   │ │         │  │
    │   └─────────┘ └─────────┘  │
    │   ┌─────────────────────┐  │
    │   │  Router (4 dirs)    │  │   ← connects to N/S/E/W neighbors
    │   └─────────────────────┘  │
    └─────────────────────────────┘

    Key properties:
    - 48 KB of SRAM per core (not cache — software-managed memory)
    - 900,000 cores × 48 KB ≈ 44 GB total on-chip
    - No cache hierarchy, no HBM, no DRAM
    - Direct core-to-core communication via 2D mesh fabric
    - Each core has its own tensor compute unit
    - Bandwidth between adjacent cores: ~21 PB/s aggregate

    The 2D mesh fabric:
    ┌────┐   ┌────┐   ┌────┐   ┌────┐
    │ C  │───│ C  │───│ C  │───│ C  │
    └─┬──┘   └─┬──┘   └─┬──┘   └─┬──┘
      │        │        │        │
    ┌─┴──┐   ┌─┴──┐   ┌─┴──┐   ┌─┴──┐
    │ C  │───│ C  │───│ C  │───│ C  │
    └─┬──┘   └─┬──┘   └─┬──┘   └─┬──┘
      │        │        │        │
    ┌─┴──┐   ┌─┴──┐   ┌─┴──┐   ┌─┴──┐
    │ C  │───│ C  │───│ C  │───│ C  │
    └────┘   └────┘   └────┘   └────┘

    Each link: very high bandwidth, very low latency
    Total fabric bandwidth: 21 PB/s
    (compare: 8×H100 NVLink total = 7.2 TB/s — that's 2,900× less)
```

### Why 44 GB SRAM >> 80 GB HBM

The WSE-3 has 44 GB of SRAM vs the H100's 80 GB of HBM.
Less total memory, but MUCH more effective:

```
    SRAM vs HBM:
                            SRAM (WSE-3)        HBM3 (H100)
    ─────────────────────────────────────────────────────────
    Capacity                44 GB               80 GB
    Bandwidth               21 PB/s             3.35 TB/s
    Bandwidth per byte      477 GB/s per GB     42 GB/s per GB
    Latency                 ~1 ns               ~100 ns
    Energy per access       ~10x lower          baseline

    The WSE has 6,268× MORE bandwidth to its memory.

    This means:
    - Memory-bound ops (softmax, LayerNorm, etc.) run at near-compute speed
    - No need for kernel fusion tricks to avoid HBM round-trips
    - Activations never leave the chip — they stay in local SRAM
```

---

## PART 3: THE WEIGHT STREAMING EXECUTION MODEL

The biggest architectural insight from Cerebras isn't the hardware — it's
the execution model. GPUs use a "weight-stationary" model. Cerebras uses
"weight streaming."

### GPU Approach: Weight-Stationary

```
    On a GPU, weights live in HBM. For each layer:

    Step 1: Load weights from HBM → SRAM
    Step 2: Stream activations through, computing results
    Step 3: Store output activations back to HBM
    Step 4: Move to next layer, repeat

    ┌─────────────────────────────────────────────┐
    │  HBM                                        │
    │  ┌──────────┐ ┌──────────┐ ┌──────────┐    │
    │  │ Layer 1   │ │ Layer 2   │ │ Layer 3   │   │
    │  │ weights   │ │ weights   │ │ weights   │   │
    │  └─────┬────┘ └─────┬────┘ └─────┬────┘    │
    │        │            │            │          │
    │        ▼            ▼            ▼          │
    │  Load → Compute → Store → Load → Compute → │
    │                                             │
    │  Problem: HBM bandwidth limits throughput   │
    └─────────────────────────────────────────────┘

    For a 70B model:
    Weights: 140 GB (FP16)
    HBM bandwidth: 3.35 TB/s
    Time to load all weights once: 140/3350 = 42 ms
    → Maximum throughput: ~24 forward passes/second
      (this is the THEORETICAL max, ignoring compute!)
```

### Cerebras Approach: Weight Streaming

```
    On WSE, activations stay local. Weights flow through.

    ┌─────────────────────────────────────────────┐
    │  WSE Chip (2D grid of cores)                │
    │                                             │
    │  Activations pinned to cores:               │
    │  ┌────┐ ┌────┐ ┌────┐ ┌────┐              │
    │  │act │ │act │ │act │ │act │  ← STAY PUT  │
    │  │ 0  │ │ 1  │ │ 2  │ │ 3  │              │
    │  └────┘ └────┘ └────┘ └────┘              │
    │    ↑      ↑      ↑      ↑                  │
    │    │      │      │      │                  │
    │  Weights stream IN from external memory:    │
    │  ════════════════════════════════════════    │
    │  Layer 1 weights → compute → discard        │
    │  Layer 2 weights → compute → discard        │
    │  Layer 3 weights → compute → discard        │
    │                                             │
    │  Weights are used once and discarded.        │
    │  Activations never leave the chip.           │
    └─────────────────────────────────────────────┘

    Why this is better:
    1. Activations (which are accessed many times) stay in fast SRAM
    2. Weights (which are used once per layer) stream through
    3. No HBM bottleneck for activations
    4. Inter-core communication for activations uses the 21 PB/s fabric
```

### Why No Tensor/Pipeline Parallelism Needed

On GPU clusters, you MUST split the model:

```
    GPU Cluster (8×H100 for 70B model):

    Tensor Parallelism (split each layer across GPUs):
    ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
    │GPU 0│ │GPU 1│ │GPU 2│ │GPU 3│
    │1/4  │ │1/4  │ │1/4  │ │1/4  │
    │of   │ │of   │ │of   │ │of   │
    │each │ │each │ │each │ │each │
    │layer│ │layer│ │layer│ │layer│
    └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘
       │       │       │       │
       └───────┴───┬───┴───────┘
                   │
              ALL-REDUCE after every layer
              (synchronization point!)

    Cost: 2 all-reduces per transformer layer
           × 80 layers = 160 all-reduces per forward pass
           Each all-reduce: ~100 μs on NVLink
           Total: ~16 ms of pure communication overhead

    Pipeline Parallelism (split layers across GPUs):
    ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐
    │GPU 0│──→│GPU 1│──→│GPU 2│──→│GPU 3│
    │L1-20│   │L21-40│  │L41-60│  │L61-80│
    └─────┘   └─────┘   └─────┘   └─────┘

    Cost: pipeline bubbles (GPUs idle while waiting)
           Bubble fraction ≈ (p-1)/(p-1+m) where p=stages, m=microbatches

    Cerebras WSE: NONE OF THIS.

    The entire model executes on ONE chip.
    - No tensor parallelism needed (no all-reduces)
    - No pipeline parallelism needed (no bubbles)
    - No complex 3D parallelism scheduling
    - Activations flow naturally through the 2D mesh

    Result: Near-100% compute utilization
```

---

## PART 4: THE CS-3 SYSTEM — MemoryX AND SwarmX

The WSE has 44 GB of SRAM. But a 70B model has 140 GB of weights (FP16).
Where do the weights come from?

### MemoryX: External Weight Storage

```
    CS-3 System Architecture:
    ┌──────────────────────────────────────────────────┐
    │                                                  │
    │   ┌──────────────────────────────┐               │
    │   │         WSE-3 Chip           │               │
    │   │   900K cores, 44 GB SRAM     │               │
    │   │   (activations live here)    │               │
    │   └──────────────┬───────────────┘               │
    │                  │                               │
    │         High-bandwidth links                     │
    │         (terabits/sec)                           │
    │                  │                               │
    │   ┌──────────────┴───────────────┐               │
    │   │        MemoryX               │               │
    │   │   External weight storage    │               │
    │   │   Up to 12 TB capacity       │               │
    │   │   (supports 24T param model) │               │
    │   │   Uses DRAM + Flash          │               │
    │   └──────────────────────────────┘               │
    │                                                  │
    └──────────────────────────────────────────────────┘

    How it works:
    1. All model weights stored in MemoryX
    2. Weights streamed to WSE layer-by-layer
    3. WSE computes on activations (already in SRAM)
    4. Weights discarded after use (or gradients sent back)
    5. MemoryX handles weight updates (optimizer step)

    The key: MemoryX bandwidth is matched to WSE compute rate.
    Weights arrive just in time for computation.
    No stalls, no waiting.
```

### SwarmX: Multi-System Scaling

For the largest models or highest throughput, multiple CS-3 systems
can be linked:

```
    SwarmX: Linking Multiple CS-3 Systems
    ┌────────────┐     ┌────────────┐     ┌────────────┐
    │   CS-3 #1  │     │   CS-3 #2  │     │   CS-3 #3  │
    │  ┌──────┐  │     │  ┌──────┐  │     │  ┌──────┐  │
    │  │ WSE-3│  │     │  │ WSE-3│  │     │  │ WSE-3│  │
    │  └──┬───┘  │     │  └──┬───┘  │     │  └──┬───┘  │
    │     │      │     │     │      │     │     │      │
    │  ┌──┴───┐  │     │  ┌──┴───┐  │     │  ┌──┴───┐  │
    │  │MemX  │  │     │  │MemX  │  │     │  │MemX  │  │
    │  └──────┘  │     │  └──────┘  │     │  └──────┘  │
    └─────┬──────┘     └─────┬──────┘     └─────┬──────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
                        SwarmX Fabric
                   (high-speed interconnect)

    SwarmX provides:
    - Data parallelism across CS-3 systems (simplest form)
    - Weight sharding for models larger than one MemoryX
    - Near-linear scaling: add more CS-3 = more throughput
    - NO tensor parallelism complexity — each WSE sees the full model
```

---

## PART 5: HARDWARE SPARSITY SUPPORT

One of the WSE's most unique features: native sparse computation.

### Why Sparsity Matters

```
    In a typical LLM:
    - ReLU activations: ~50% of values are zero
    - Attention weights: often >90% near-zero
    - Model weights: can be pruned to 50-80% sparsity with minimal quality loss

    On a GPU:
    - Sparse operations are HARD to accelerate
    - NVIDIA's 2:4 structured sparsity: requires exactly 2 zeros per 4 elements
    - Unstructured sparsity: no speedup on GPUs (irregular memory access)

    On WSE:
    - EVERY core checks for zeros BEFORE computing
    - Zero × anything = zero → skip the multiply entirely
    - Works with ANY sparsity pattern (unstructured)
    - Speedup scales linearly with sparsity level

    Example:
    Matrix multiply with 50% sparse activations:
        GPU (dense):    100% of multiplies executed
        GPU (2:4):      50% of multiplies (but strict pattern required)
        WSE:            50% of multiplies (any pattern!)

    Matrix multiply with 80% sparse activations:
        GPU (dense):    100% of multiplies executed
        GPU (2:4):      still 50% (can't exploit beyond 2:4)
        WSE:            20% of multiplies → 5× speedup!
```

### Training with Sparsity

```
    Cerebras has demonstrated training with sparsity:

    1. Activation Sparsity (free speedup):
       - ReLU naturally produces ~50% zeros
       - WSE skips these automatically
       - No accuracy impact (it's exact — zeros ARE zero)

    2. Weight Sparsity (pruning during training):
       - Gradually prune weights during training
       - WSE accelerates forward/backward proportionally
       - Can train a model that's 80% sparse from the start
       - Same final quality with 5× less compute

    3. Sparsity-Aware Training Pipeline:
       ┌─────────┐    ┌──────────┐    ┌─────────┐
       │ Dense   │───→│ Gradual  │───→│ Sparse  │
       │ warmup  │    │ pruning  │    │ training│
       │ (10%)   │    │ (20%)    │    │ (70%)   │
       └─────────┘    └──────────┘    └─────────┘

       Result: ~3× faster total training at same quality
```

---

## PART 6: CEREBRAS FOR INFERENCE

Cerebras has demonstrated remarkable inference performance, particularly
for long sequences.

### The GPU Inference Bottleneck

```
    Autoregressive LLM inference on GPU:

    For each token generated:
    1. Load ALL model weights from HBM → compute units
    2. Multiply by ONE token's activations
    3. Produce ONE output token
    4. Repeat

    For a 70B model (FP16):
    Weights: 140 GB
    HBM bandwidth: 3.35 TB/s (H100)
    Time to load weights: 140 / 3350 = 41.8 ms
    → Maximum: ~24 tokens/second PER USER (regardless of batch size 1)

    This is the "memory bandwidth wall" of LLM inference.
    The compute units are >95% idle during token generation!

    Batch size helps amortize:
    Batch=32: same weight loads, 32× more useful work
    But: batch=32 needs 32× more KV cache memory
    And: latency per request stays the same (or gets worse)
```

### Cerebras Inference Advantage

```
    WSE-3 for inference:

    Weights stream from MemoryX → WSE
    Bandwidth: matched to compute, so no bottleneck

    For a 70B model:
    WSE memory bandwidth: 21 PB/s (internal)
    Weight streaming: matched to compute rate

    Result:
    - ~20× faster per-token latency vs H100
    - Scales to very long sequences (activations stay on-chip)
    - No KV cache in HBM (it's in SRAM!)

    Concrete benchmark numbers (Cerebras published):
    Llama 3.1 70B inference:
        Cerebras CS-3:  ~2,100 tokens/sec (output)
        H100 (single):  ~24 tokens/sec (output, batch=1)
        H100 (8×, TP):  ~100 tokens/sec (output, batch=1)

    For long contexts (128K tokens):
    The KV cache for 70B at 128K tokens:
        MHA: 80 layers × 2 × 8192 × 128K × 2 bytes ≈ 320 GB
        → Doesn't fit in ONE H100! Need tensor parallelism.
        → On WSE: activations spread across 900K cores, fits in 44 GB SRAM
           (using GQA, the actual KV cache is much smaller)
```

### Why Low Latency for Long Sequences

```
    GPU approach to long sequences:
    ┌─────────────────────────────────────────┐
    │  Sequence length: 128K tokens           │
    │  KV cache: 320 GB (doesn't fit!)        │
    │  Solution: split across 8 GPUs          │
    │  Cost: all-reduce at every attention     │
    │  Latency: dominated by communication    │
    └─────────────────────────────────────────┘

    WSE approach:
    ┌─────────────────────────────────────────┐
    │  Sequence length: 128K tokens           │
    │  KV cache: distributed across cores     │
    │  All in SRAM, all on one chip           │
    │  Communication: 2D mesh at 21 PB/s      │
    │  Latency: dominated by compute (ideal!) │
    └─────────────────────────────────────────┘

    The key: on WSE, attention computation over long sequences
    is a LOCAL operation. Each core holds a chunk of the KV cache
    and computes its portion of attention. Results are reduced
    across cores using the mesh fabric.

    No PCIe. No NVLink. No InfiniBand. Just on-chip wires.
```

---

## PART 7: PERFORMANCE COMPARISONS

### Training Throughput

```
    Training a 1.3B parameter model:

    System                  Throughput (tokens/sec)    Hardware
    ──────────────────────────────────────────────────────────────
    1× H100                 ~45,000                    1 GPU
    8× H100 (DGX)           ~320,000                   8 GPUs
    1× CS-3                 ~480,000                   1 WSE-3

    Training a 70B parameter model:

    System                  Throughput (tokens/sec)    Hardware
    ──────────────────────────────────────────────────────────────
    64× H100 (TP=8, DP=8)  ~95,000                    64 GPUs
    256× H100               ~350,000                   256 GPUs
    4× CS-3 (SwarmX)        ~400,000                   4 WSE-3

    Key: Cerebras achieves similar throughput with FAR fewer systems
    and MUCH simpler software (no 3D parallelism!)
```

### Inference Latency

```
    Time to first token (TTFT) — Llama 70B, 2K input:

    System              TTFT        Tokens/sec (output)
    ───────────────────────────────────────────────────
    H100 (1×, BS=1)    ~420 ms     ~24 tok/s
    H100 (8×, TP=8)    ~85 ms      ~100 tok/s
    CS-3                ~12 ms      ~2,100 tok/s

    For long context (32K input):

    System              TTFT        Tokens/sec (output)
    ───────────────────────────────────────────────────
    H100 (8×, TP=8)    ~850 ms     ~95 tok/s
    CS-3                ~45 ms      ~1,800 tok/s
```

### Cost Efficiency

```
    The cost picture is more nuanced:

    CS-3 system:        ~$2-3M (estimated)
    8× H100 DGX:        ~$300K

    But per-token cost for inference:
    CS-3 serves ~20× more tokens/sec than 8× H100
    So cost per token can be LOWER on Cerebras
    (depends heavily on utilization)

    For training:
    CS-3 replaces ~16-64 H100s for many workloads
    Total cost of ownership includes:
    - Power (WSE: ~23 kW, DGX H100: ~10 kW)
    - Networking equipment (GPUs need expensive switches)
    - Engineering time (Cerebras: much simpler software)
    - Time to solution (faster training = earlier results)
```

---

## PART 8: LIMITATIONS AND TRADEOFFS

Cerebras isn't perfect for every workload. Here are the honest tradeoffs:

```
    Strengths:
    ✓ Single-chip simplicity (no distributed training complexity)
    ✓ Massive memory bandwidth (21 PB/s)
    ✓ Native sparsity support
    ✓ Excellent for inference latency
    ✓ Linear scaling with SwarmX
    ✓ Great for long sequences

    Limitations:
    ✗ 44 GB SRAM limits activation memory
      (batch size constrained for very large models)
    ✗ Expensive upfront cost ($2-3M per CS-3)
    ✗ Ecosystem: NVIDIA has CUDA, cuDNN, PyTorch native support
      Cerebras requires their SDK
    ✗ Availability: can't rent on-demand from most clouds
      (Cerebras offers their own cloud)
    ✗ Not great for workloads that need huge batch sizes
      (e.g., recommendation systems with millions of items)
    ✗ Model must fit Cerebras's supported op set
      (custom CUDA kernels can't be ported)

    Best suited for:
    - LLM training (especially with sparsity)
    - LLM inference (especially long-context, low-latency)
    - Scientific computing (molecular dynamics, etc.)
    - Organizations that want simple scaling

    Less suited for:
    - Computer vision (batch size matters more)
    - Recommendation systems
    - Workloads needing custom CUDA kernels
    - Budget-constrained teams
```

---

## PART 9: THE BIGGER PICTURE — ARCHITECTURE COMPARISON

```
    ┌─────────────────────────────────────────────────────────────┐
    │                   HARDWARE COMPARISON                       │
    ├──────────────┬──────────────┬──────────────┬───────────────┤
    │              │  NVIDIA H100 │  Google TPUv5│  Cerebras WSE │
    ├──────────────┼──────────────┼──────────────┼───────────────┤
    │ Architecture │  SIMT        │  Systolic    │  Dataflow     │
    │              │  (GPU)       │  array       │  (2D mesh)    │
    ├──────────────┼──────────────┼──────────────┼───────────────┤
    │ Memory model │  HBM + SRAM  │  HBM + SRAM  │  SRAM only    │
    ├──────────────┼──────────────┼──────────────┼───────────────┤
    │ Scaling      │  Multi-GPU   │  Multi-chip  │  Single chip  │
    │              │  (complex)   │  (ICI links) │  (simple)     │
    ├──────────────┼──────────────┼──────────────┼───────────────┤
    │ Parallelism  │  TP+PP+DP    │  TP+DP       │  DP only      │
    │  required    │              │              │  (via SwarmX) │
    ├──────────────┼──────────────┼──────────────┼───────────────┤
    │ Sparsity     │  2:4 only    │  Limited     │  Unstructured │
    ├──────────────┼──────────────┼──────────────┼───────────────┤
    │ Ecosystem    │  Massive     │  Growing     │  Niche        │
    │              │  (CUDA)      │  (JAX/XLA)   │  (Cerebras SDK│
    ├──────────────┼──────────────┼──────────────┼───────────────┤
    │ Availability │  Everywhere  │  GCP only    │  Cerebras     │
    │              │              │              │  Cloud        │
    └──────────────┴──────────────┴──────────────┴───────────────┘
```

---

## SUMMARY

```
    The Cerebras WSE represents a fundamentally different approach to AI compute:

    1. WAFER-SCALE: Use the entire silicon wafer as one chip
       → 900K cores, 44 GB SRAM, 21 PB/s bandwidth

    2. WEIGHT STREAMING: Weights flow through; activations stay local
       → Eliminates the HBM bandwidth bottleneck

    3. NO PARALLELISM COMPLEXITY: One chip, one model
       → No tensor/pipeline parallelism needed

    4. NATIVE SPARSITY: Hardware skips zeros automatically
       → Free speedup proportional to sparsity level

    5. MemoryX + SwarmX: External memory + multi-system scaling
       → Handles models of any size

    The GPU approach optimizes within constraints (HBM, multi-chip).
    The Cerebras approach removes the constraints entirely.

    Whether this wins long-term depends on:
    - Cost scaling (can wafer-scale manufacturing get cheaper?)
    - Ecosystem (can they match CUDA's tooling?)
    - Model trends (do sparse models become dominant?)
    - Competition (NVIDIA, Google, AMD all improving rapidly)
```

"""
Distributed LLM Training: Megatron-LM and the Three Dimensions of Parallelism

This module explains how Megatron-LM trains models with billions to trillions
of parameters across thousands of GPUs. Based on:
  - Megatron-LM v1 (2019): "Megatron-LM: Training Multi-Billion Parameter Language Models
    Using Model Parallelism" (Shoeybi et al.)
  - Megatron-LM v2 (2021): "Efficient Large-Scale Language Model Training on GPU Clusters"
    (Narayanan et al.)

================================================================================
PART 1: THE PROBLEM - WHY ONE GPU ISN'T ENOUGH
================================================================================

Let's do some math on GPT-3 (175 billion parameters):

    Model size in memory:
        175B params × 2 bytes (FP16) = 350 GB   (just the weights!)

    Training state (with Adam optimizer):
        Weights:             350 GB  (FP16)
        Gradients:           350 GB  (FP16)
        Optimizer states:    700 GB  (FP32 copy of weights + FP32 momentum + FP32 variance)
        Total:             ~1,400 GB = 1.4 TB

    Largest GPU (A100):      80 GB

    So you need AT LEAST 18 GPUs just to STORE the model state.
    Plus activations, plus batch data...

And even if you COULD fit it on one GPU:

    FLOPs to train GPT-3:   ~3.14 × 10^23
    A100 peak:               312 TFLOPS
    Time at 100% utilization: 3.14e23 / 312e12 = 1.0e9 seconds = ~32 YEARS

    With 1000 GPUs at 50% efficiency: ~23 DAYS

So we need parallelism for TWO reasons:
    1. Memory: model doesn't fit on one GPU
    2. Speed: training would take decades otherwise

There are three dimensions of parallelism. Let's build them up one by one.


================================================================================
PART 2: DATA PARALLELISM (DP) - THE EASY ONE
================================================================================

The simplest form of parallelism: replicate the entire model on N GPUs,
split the batch across them.

    Global batch: [x0, x1, x2, x3, x4, x5, x6, x7]

    GPU 0: full model copy, processes [x0, x1]
    GPU 1: full model copy, processes [x2, x3]
    GPU 2: full model copy, processes [x4, x5]
    GPU 3: full model copy, processes [x6, x7]

    Each GPU does:
        1. Forward pass on its mini-batch
        2. Compute gradients
        3. ALL-REDUCE: average gradients across all GPUs
        4. Update weights (identical update since gradients are averaged)

    After step 4, all GPUs have identical weights again.

The all-reduce operation:

    Before:  GPU0 has grad_0, GPU1 has grad_1, GPU2 has grad_2, GPU3 has grad_3
    After:   ALL GPUs have (grad_0 + grad_1 + grad_2 + grad_3) / 4

    This is implemented as a ring all-reduce: each GPU sends/receives
    to its neighbor in a ring, taking 2(N-1)/N × data_size of communication.

Pros:
    + Simple to implement (PyTorch DDP does it automatically)
    + Near-linear scaling for compute
    + No model changes needed

Cons:
    - Each GPU must hold the ENTIRE model
    - For GPT-3: need 1.4 TB per GPU? Impossible.
    - Gradient all-reduce becomes a bottleneck at large scale

Data parallelism alone works for models up to ~1-2B parameters.
Beyond that, you need to split the model itself.


================================================================================
PART 3: TENSOR PARALLELISM (TP) - MEGATRON'S KEY INNOVATION
================================================================================

Tensor parallelism splits individual LAYERS across GPUs.
This is Megatron-LM's core contribution.

The key insight: a matrix multiply can be split column-wise or row-wise,
and the transformer's structure lets us do this with MINIMAL communication.

Let's trace through a single transformer layer:

    Input X → [MLP] → [Attention] → Output

    The MLP is:  Y = GeLU(X @ A) @ B     (two linear layers with GeLU)
    Let's split it.

--- Splitting the MLP ---

    The first linear layer: Y = GeLU(X @ A)

    Split A by COLUMNS across 2 GPUs:

        A = [A_1 | A_2]    (column split)

        GPU 0 computes: Y_1 = GeLU(X @ A_1)
        GPU 1 computes: Y_2 = GeLU(X @ A_2)

    Why columns? Because GeLU is nonlinear:

        GeLU([X@A_1, X@A_2]) = [GeLU(X@A_1), GeLU(X@A_2)]  ✓ WORKS!

        GeLU is applied element-wise, so splitting columns means each GPU
        can apply GeLU independently. No communication needed!

    If we split by ROWS instead:

        X @ A = X @ [A_1; A_2] = X_1 @ A_1 + X_2 @ A_2   (need to SUM)

        GeLU(X_1 @ A_1 + X_2 @ A_2) ≠ GeLU(X_1@A_1) + GeLU(X_2@A_2)  ✗ BROKEN!

        We'd need to communicate BEFORE applying GeLU. Bad.

    The second linear layer: Z = Y @ B

    Now Y is split across GPUs: GPU 0 has Y_1, GPU 1 has Y_2.
    Split B by ROWS (to match the column split of Y):

        B = [B_1]    (row split)
            [B_2]

        GPU 0 computes: Z_1 = Y_1 @ B_1
        GPU 1 computes: Z_2 = Y_2 @ B_2

        Final result: Z = Z_1 + Z_2   ← requires ALL-REDUCE

    So the full MLP with 2-GPU tensor parallelism:

        GPU 0                          GPU 1
        ─────                          ─────
        Y_1 = GeLU(X @ A_1)           Y_2 = GeLU(X @ A_2)
        Z_1 = Y_1 @ B_1               Z_2 = Y_2 @ B_2
                    ↘                ↙
                      ALL-REDUCE (sum)
                    ↙                ↘
               Z = Z_1 + Z_2         Z = Z_1 + Z_2

    Communication: ONE all-reduce per MLP. That's it!

--- Splitting Multi-Head Attention ---

    This is even easier. Multi-head attention is INHERENTLY parallel:

        MultiHead(Q, K, V) = Concat(head_1, head_2, ..., head_h) @ W_O

    With 16 heads and 2 GPUs:
        GPU 0: compute heads 1-8
        GPU 1: compute heads 9-16

    Each GPU has its portion of Q, K, V weight matrices (column-split)
    and computes its attention heads independently.

    Then the output projection W_O is row-split (like B above).
    One all-reduce to combine.

    GPU 0                              GPU 1
    ─────                              ─────
    heads 1-8:                         heads 9-16:
      Q_1 = X @ W_Q1                    Q_2 = X @ W_Q2
      K_1 = X @ W_K1                    K_2 = X @ W_K2
      V_1 = X @ W_V1                    V_2 = X @ W_V2
      attn_1 = Attention(Q1,K1,V1)       attn_2 = Attention(Q2,K2,V2)
      out_1 = attn_1 @ W_O1             out_2 = attn_2 @ W_O2
                    ↘                  ↙
                      ALL-REDUCE (sum)
                    ↙                  ↘
                out = out_1 + out_2    out = out_1 + out_2

--- Full Transformer Layer with TP ---

    Per transformer layer, the communication is:

        1. All-reduce after MLP         (forward)
        2. All-reduce after Attention   (forward)
        3. All-reduce after MLP         (backward)
        4. All-reduce after Attention   (backward)

    = 4 all-reduce operations per layer per training step

    The data transferred per all-reduce: batch_size × seq_len × hidden_dim × 2 bytes
    For GPT-3: 4 × 2048 × 12288 × 2 = ~200 MB per all-reduce

    This is why TP MUST use fast interconnects (NVLink: 600 GB/s).
    Over network (InfiniBand: ~25 GB/s), TP would be 24x slower → unusable.

    Rule of thumb: TP within a single node (8 GPUs with NVLink).
                   Never across nodes.


================================================================================
PART 4: WHY THE COLUMN-THEN-ROW SPLIT IS GENIUS
================================================================================

Let's make sure the math is crystal clear. Consider a simple MLP:

    Z = dropout(GeLU(X @ A) @ B)

    X:  [b, s, h]      (batch, sequence, hidden)
    A:  [h, 4h]         (expand by 4x, standard transformer MLP)
    B:  [4h, h]          (project back down)

With TP degree 2:

    A split into columns: A_1 [h, 2h], A_2 [h, 2h]
    B split into rows:    B_1 [2h, h], B_2 [2h, h]

    GPU 0:
        Y_1 = GeLU(X @ A_1)    # [b,s,h] @ [h,2h] → [b,s,2h]
        Z_1 = Y_1 @ B_1        # [b,s,2h] @ [2h,h] → [b,s,h]

    GPU 1:
        Y_2 = GeLU(X @ A_2)    # [b,s,h] @ [h,2h] → [b,s,2h]
        Z_2 = Y_2 @ B_2        # [b,s,2h] @ [2h,h] → [b,s,h]

    All-reduce: Z = Z_1 + Z_2  # [b,s,h]

Verify correctness:

    Z_1 + Z_2 = Y_1 @ B_1 + Y_2 @ B_2
              = GeLU(X@A_1) @ B_1 + GeLU(X@A_2) @ B_2

    Standard (no split):
    Z = GeLU(X @ A) @ B
      = GeLU(X @ [A_1|A_2]) @ [B_1; B_2]
      = [GeLU(X@A_1) | GeLU(X@A_2)] @ [B_1; B_2]    ← GeLU is element-wise
      = GeLU(X@A_1) @ B_1 + GeLU(X@A_2) @ B_2        ← block matrix multiply

    ✓ Identical! The split is mathematically exact.

What about dropout?

    In the standard case: Z = dropout(GeLU(X@A) @ B)

    With TP: Each GPU applies dropout to its Z_i BEFORE the all-reduce,
    using DIFFERENT random seeds. This is equivalent to applying dropout
    to the concatenated result (since dropout is also element-wise on
    independent elements).

What needs to be identical across GPUs?

    - Input X: must be the same (broadcast or replicated)
    - Random seeds for dropout: must be DIFFERENT per GPU per layer
    - Everything else: naturally partitioned

This is what makes Megatron's approach so elegant:
    - No compiler or framework changes needed
    - Just insert all-reduce at two points per layer
    - Native PyTorch: a few lines of code


================================================================================
PART 5: PIPELINE PARALLELISM (PP) - SPLITTING LAYERS ACROSS NODES
================================================================================

TP works great within a node (8 GPUs, NVLink). But for bigger models,
we need to split across nodes. That's where pipeline parallelism comes in.

Pipeline parallelism splits the model by LAYERS:

    Node 0 (GPUs 0-7):   Layers 1-12    ← "Pipeline Stage 0"
    Node 1 (GPUs 8-15):  Layers 13-24   ← "Pipeline Stage 1"
    Node 2 (GPUs 16-23): Layers 25-36   ← "Pipeline Stage 2"
    Node 3 (GPUs 24-31): Layers 37-48   ← "Pipeline Stage 3"

    (Within each node, layers are further split via TP)

--- The Naive Approach (Bad) ---

    Stage 0 processes the full batch → sends activations to Stage 1 →
    Stage 1 processes → sends to Stage 2 → ... → backward same path

    Timeline:
        Stage 0: [FFFFF][idle ][idle ][idle ][BBBBB][idle ][idle ][idle ]
        Stage 1: [idle ][FFFFF][idle ][idle ][idle ][BBBBB][idle ][idle ]
        Stage 2: [idle ][idle ][FFFFF][idle ][idle ][idle ][BBBBB][idle ]
        Stage 3: [idle ][idle ][idle ][FFFFF][idle ][idle ][idle ][BBBBB]

    Only 1 out of 4 stages is active at any time. 75% idle. Terrible!

--- Microbatching (GPipe) ---

    Split the batch into M microbatches and pipeline them:

    Batch of 8 samples → 8 microbatches of 1

    Stage 0: [F0][F1][F2][F3][F4][F5][F6][F7][  ][  ][  ][B7][B6][B5][B4][B3][B2][B1][B0]
    Stage 1: [  ][F0][F1][F2][F3][F4][F5][F6][F7][  ][B7][B6][B5][B4][B3][B2][B1][B0][  ]
    Stage 2: [  ][  ][F0][F1][F2][F3][F4][F5][F6][F7][B7][B6][B5][B4][B3][B2][B1][B0][  ]
    Stage 3: [  ][  ][  ][F0][F1][F2][F3][F4][F5][F6][B7][B6][B5][B4][B3][B2][B1][B0][  ]

    Better! But still has a "bubble" at the start and end.

    GPipe bubble fraction = (P - 1) / M
    Where P = number of pipeline stages, M = number of microbatches

    With P=4, M=8: bubble = 3/8 = 37.5%
    With P=4, M=32: bubble = 3/32 = 9.4%    ← much better with more microbatches

    Problem: GPipe stores activations for ALL M microbatches simultaneously.
    Memory: O(M) intermediate activations. With large M, this blows up.

--- 1F1B Schedule (PipeDream-Flush / Megatron-LM) ---

    Megatron uses the "one forward, one backward" (1F1B) schedule:

    Phase 1 - Warmup: each stage does forward passes to fill the pipeline
    Phase 2 - Steady state: alternate 1 forward, 1 backward
    Phase 3 - Cooldown: drain remaining backward passes

    Stage 0: [F0][F1][F2][F3][B0][F4][B1][F5][B2][F6][B3][F7][B4][B5][B6][B7]
    Stage 1: [  ][F0][F1][F2][F3][B0][F4][B1][F5][B2][F6][B3][F7][B4][B5][B6][B7]
    Stage 2: [  ][  ][F0][F1][F2][F3][B0][F4][B1][F5][B2][F6][B3][F7][B4][B5][B6][B7]
    Stage 3: [  ][  ][  ][F0][F1][F2][F3][B0][F4][B1][B2][B3][B4][B5][B6][B7]

    Same bubble size as GPipe: (P-1)/M
    But MUCH less memory: only P microbatches' activations at peak (not M).

    With P=4: store at most 4 sets of activations, regardless of M.

--- Interleaved Schedule (Megatron-LM v2) ---

    The idea: make each pipeline stage do LESS work so the pipeline
    fills up and drains FASTER, shrinking the bubble.

    How? Give each device MULTIPLE non-contiguous chunks of layers
    instead of one big contiguous block.

    Non-interleaved (normal): 24 layers, 4 devices, 6 layers each

        Device 0: Layers  1-6     (one chunk)
        Device 1: Layers  7-12    (one chunk)
        Device 2: Layers 13-18    (one chunk)
        Device 3: Layers 19-24    (one chunk)

        Microbatch path: D0 → D1 → D2 → D3  (4 hops)

    Interleaved (v=2): same 24 layers, same 4 devices, but 2 chunks of 3 each

        Device 0: Layers 1-3   AND  Layers 13-15
        Device 1: Layers 4-6   AND  Layers 16-18
        Device 2: Layers 7-9   AND  Layers 19-21
        Device 3: Layers 10-12 AND  Layers 22-24

        Microbatch path (must execute layers 1→24 in order):
            Layers  1-3:  Device 0   (chunk 1)
            Layers  4-6:  Device 1   (chunk 1)
            Layers  7-9:  Device 2   (chunk 1)
            Layers 10-12: Device 3   (chunk 1)
            Layers 13-15: Device 0   (chunk 2)  ← BACK to Device 0!
            Layers 16-18: Device 1   (chunk 2)  ← back to Device 1!
            Layers 19-21: Device 2   (chunk 2)
            Layers 22-24: Device 3   (chunk 2)

        Path: D0 → D1 → D2 → D3 → D0 → D1 → D2 → D3  (8 hops)

    Why does this shrink the bubble? Each stage now does 3 layers
    instead of 6, so each step takes HALF as long. The pipeline
    fills up in half the time:

    Non-interleaved (each block = T time):

        D0: [====F0====][··········][··········][··········][====F1====]...
        D1: [··········][====F0====][··········][··········]...
        D2: [··········][··········][====F0====][··········]...
        D3: [··········][··········][··········][====F0====]...

        D0 idle for 3T before it can do more work. Bubble = 3T.

    Interleaved (each block = T/2 time, "a"=chunk1, "b"=chunk2):

        D0: [==F0a==][·······][·······][·······][==F0b==][·······][·······][·······][==F1a==]
        D1: [·······][==F0a==][·······][·······][·······][==F0b==][·······][·······]...
        D2: [·······][·······][==F0a==][·······][·······][·······][==F0b==]...
        D3: [·······][·······][·······][==F0a==][·······][·······][·······][==F0b==]...

        D0 idle for 3 × T/2 = 1.5T. Bubble HALVED!

    Bubble formula: (P-1) / (M × v)
    Where v = number of chunks per device

    With P=4, M=8, v=1 (normal):  bubble = 3/8  = 37.5%
    With P=4, M=8, v=2 (interleaved): bubble = 3/16 = 18.75%  ← half!

    Trade-off: 2x more inter-device communication (8 hops instead of 4),
    but the bubble is halved. Worth it when the interconnect is fast enough.

    The intuition: each stage is like a worker on an assembly line.
    If you give each worker LESS to do per step, items flow through
    the line faster. There's still warmup/cooldown time, but it's
    proportionally smaller because each step is quicker.


================================================================================
PART 6: COMBINING ALL THREE - 3D PARALLELISM
================================================================================

For truly large models, Megatron-LM combines all three:

    TP (Tensor Parallelism):   Split layers WITHIN a node     (NVLink)
    PP (Pipeline Parallelism): Split layers ACROSS nodes       (InfiniBand)
    DP (Data Parallelism):     Replicate pipeline across groups (InfiniBand)

Example: Training a 1 trillion parameter model on 3072 A100 GPUs:

    TP degree: 8  (within each DGX A100 node: 8 GPUs connected by NVLink)
    PP degree: 8  (8 nodes form one pipeline: layers split across 8 stages)
    DP degree: 48 (48 copies of the pipeline, each getting different data)

    Total GPUs: 8 × 8 × 48 = 3,072

    Visual layout:

    DP group 0:
    ┌──────────────────────────────────────────────────────────┐
    │ Node 0 (8 GPUs, TP)  → Node 1 (8 GPUs, TP) → ... → Node 7 │
    │ [Layers 1-N/8]         [Layers N/8+1-2N/8]     [Last layers]│
    │      PP stage 0             PP stage 1           PP stage 7  │
    └──────────────────────────────────────────────────────────┘

    DP group 1:  (same structure, different data)
    ┌──────────────────────────────────────────────────────────┐
    │ Node 8 (8 GPUs, TP) → Node 9 (8 GPUs, TP) → ... → Node 15  │
    └──────────────────────────────────────────────────────────┘

    ... (48 such groups total)

Communication patterns:

    TP (within node):
        - All-reduce after each attention + MLP
        - NVLink: ~600 GB/s bandwidth
        - Latency: microseconds

    PP (between adjacent stages):
        - Point-to-point: send activations forward, gradients backward
        - InfiniBand: ~25 GB/s per link
        - Data per transfer: batch × seq_len × hidden_dim × 2 bytes
          (small relative to TP communication)

    DP (across all pipeline replicas):
        - All-reduce of gradients after each training step
        - InfiniBand, but overlapped with backward computation
        - Largest communication volume, but amortized over micro-batches

Why this specific mapping?

    TP MUST be within a node:
        - 4 all-reduces per layer, high frequency
        - Needs NVLink bandwidth (600 GB/s)
        - Over InfiniBand (25 GB/s): 24x slower → kills throughput

    PP should be between nodes:
        - Only point-to-point sends between adjacent stages
        - Low bandwidth requirement (just one activation tensor)
        - Higher latency is okay (pipelined over many microbatches)

    DP is most flexible:
        - Gradient all-reduce happens once per training step
        - Can overlap with backward pass computation
        - Works fine over InfiniBand


================================================================================
PART 7: THE SCATTER/GATHER OPTIMIZATION
================================================================================

Megatron-LM has a clever trick for PP communication between nodes.

The problem: with TP degree 8, each of the 8 GPUs in a pipeline stage
has the SAME activation output (after the all-reduce). When sending to
the next pipeline stage, all 8 GPUs send identical data. That's 8x redundancy!

    Node 0 (Stage 0):    GPU0, GPU1, ..., GPU7 each have identical output X
    Node 1 (Stage 1):    GPU0, GPU1, ..., GPU7 each need X

    Naive: all 8 GPUs on Node 0 send X over InfiniBand to Node 1
           = 8 × |X| bytes over the network  (wasteful!)

    Scatter/Gather optimization:

    SEND SIDE (Node 0):
        Split X into 8 chunks: [X_0, X_1, ..., X_7]
        GPU_i sends only chunk X_i to GPU_i on Node 1
        Each GPU sends |X|/8 bytes  (8x less per link)
        Total bandwidth: same |X|, but across 8 PARALLEL IB links

    RECEIVE SIDE (Node 1):
        Each GPU_i receives chunk X_i from Node 0
        All-gather over NVLink (fast!) to reconstruct full X
        NVLink is ~24x faster than IB, so this is nearly free

    Result:
        - Uses all 8 InfiniBand cards in parallel (instead of 1)
        - Effective bandwidth: 8 × 25 GB/s = 200 GB/s (8x improvement!)
        - The NVLink all-gather cost is negligible


================================================================================
PART 8: ZeRO - SHARDING THE OPTIMIZER STATE
================================================================================

Even with TP + PP, Data Parallelism still replicates the FULL optimizer
state on every DP rank. For large models, this is wasteful.

ZeRO (Zero Redundancy Optimizer) from DeepSpeed shards optimizer state:

    Standard DP with 4 GPUs:
        GPU 0: all params + all grads + all optimizer state  (1.4 TB)
        GPU 1: all params + all grads + all optimizer state  (1.4 TB)
        GPU 2: all params + all grads + all optimizer state  (1.4 TB)
        GPU 3: all params + all grads + all optimizer state  (1.4 TB)
        Total: 5.6 TB                                       (4x redundant!)

    ZeRO Stage 1 (shard optimizer states):
        GPU 0: all params + all grads + 1/4 optimizer state
        GPU 1: all params + all grads + 1/4 optimizer state
        GPU 2: all params + all grads + 1/4 optimizer state
        GPU 3: all params + all grads + 1/4 optimizer state
        Saves: ~4x on optimizer memory

    ZeRO Stage 2 (+ shard gradients):
        GPU 0: all params + 1/4 grads + 1/4 optimizer state
        GPU 1: all params + 1/4 grads + 1/4 optimizer state
        ...
        Saves: ~8x

    ZeRO Stage 3 (+ shard parameters):
        GPU 0: 1/4 params + 1/4 grads + 1/4 optimizer state
        ...
        Saves: ~linear with number of GPUs
        Cost: must all-gather params before each forward/backward pass

    With 3D parallelism, typically only ZeRO Stage 1 is used
    (sharding optimizer state within each DP group).
    Stage 2/3 add too much communication when combined with PP.


================================================================================
PART 9: ACTIVATION CHECKPOINTING / RECOMPUTATION
================================================================================

During training, you must store intermediate activations for the backward pass.

    Forward:  X → Layer 1 → a_1 → Layer 2 → a_2 → ... → Layer N → loss
    Backward: Need a_1, a_2, ..., a_N to compute gradients

    Memory for activations: O(N × batch × seq_len × hidden_dim)
    For GPT-3: this can be 10s of GB per microbatch

Activation checkpointing trades compute for memory:

    Instead of storing ALL activations:
        Only store activations at certain "checkpoint" layers
        (e.g., every 3rd layer)

    During backward pass:
        When you need a_5 but only stored a_3:
        Re-run forward from a_3 through layers 4-5 to recompute a_5

    Memory:  O(N/k × batch × seq × hidden)  where k = checkpoint interval
    Compute: ~33% more (one extra forward pass for non-checkpointed layers)

    This is essential for training large models. Megatron-LM checkpoints
    every transformer layer by default.

Selective activation recomputation (Megatron-LM v2):

    Not all activations are equally expensive to store.

    Attention scores: [batch, heads, seq, seq] = can be HUGE for long sequences
    Linear layer outputs: [batch, seq, hidden] = relatively small

    Megatron selectively recomputes only the expensive attention activations
    and stores the cheap linear outputs. This gets most of the memory savings
    with much less recomputation overhead.


================================================================================
PART 10: PUTTING IT ALL TOGETHER - A CONCRETE EXAMPLE
================================================================================

Training GPT-3 (175B parameters) on 1024 A100 GPUs:

    Configuration:
        TP = 8    (within each DGX A100 node)
        PP = 8    (8 nodes per pipeline, with interleaved schedule)
        DP = 16   (16 copies of the full pipeline)
        Total: 8 × 8 × 16 = 1,024 GPUs

    Model layout:
        175B params / 96 layers
        PP = 8 stages → 12 layers per stage
        TP = 8 → each layer split across 8 GPUs within a node
        Each GPU holds: 12 layers / 8 TP shards ≈ 2.7 GB of params

    Training step:
        Global batch size: 1536 sequences of 2048 tokens
        Per DP rank: 1536 / 16 = 96 sequences
        Microbatch size: 1 sequence
        Number of microbatches: 96

    Pipeline bubble:
        With interleaved schedule (v=2 chunks):
        Bubble = (P-1) / (M×v) = 7 / (96×2) = 3.6%  ← very small!

    Communication per training step:
        TP: 4 all-reduces per layer × 96 layers = 384 all-reduces (NVLink)
        PP: ~192 point-to-point sends (microbatches through pipeline)
        DP: 1 gradient all-reduce (overlapped with backward pass)

    Achieved performance:
        ~163 TFLOPS per GPU (52% of peak 312 TFLOPS)
        Aggregate: 167 PFLOPS across 1024 GPUs
        Time to train: ~34 days


================================================================================
PART 11: SUMMARY - THE MENTAL MODEL
================================================================================

    Dimension   │ What it splits    │ Where        │ Communication
    ────────────┼───────────────────┼──────────────┼────────────────
    Tensor (TP) │ Each layer        │ Within node  │ All-reduce (NVLink)
    Pipeline(PP)│ Groups of layers  │ Across nodes │ Point-to-point (IB)
    Data (DP)   │ The batch         │ Across groups│ Gradient all-reduce (IB)

    Why this order?

    TP needs the most bandwidth (many all-reduces per layer)
    → Use fastest link: NVLink (600 GB/s)

    PP needs moderate bandwidth (activation tensors between stages)
    → InfiniBand is sufficient (25 GB/s per link, 8 links per node)

    DP needs least frequent communication (once per training step)
    → Can overlap with compute; InfiniBand is fine

    The hierarchy mirrors the hardware:

    ┌─ GPU ←─ NVLink ─→ GPU ─┐
    │        (TP domain)      │
    │      DGX A100 Node      │
    └────────────┬────────────┘
                 │ InfiniBand (PP)
    ┌────────────┴────────────┐
    │      DGX A100 Node      │
    │        (TP domain)      │
    └────────────┬────────────┘
                 │ InfiniBand (PP)
                ...
                 │ InfiniBand (DP across pipeline replicas)
    ┌────────────┴────────────┐
    │   Another full pipeline  │
    │   (different data shard) │
    └─────────────────────────┘

Key takeaway: Megatron-LM's genius is mapping the three types of
parallelism to the three tiers of hardware interconnect, minimizing
communication bottlenecks at each level.

Papers:
    - Megatron-LM v1: Shoeybi et al., "Megatron-LM: Training Multi-Billion
      Parameter Language Models Using Model Parallelism", 2019
    - Megatron-LM v2: Narayanan et al., "Efficient Large-Scale Language Model
      Training on GPU Clusters", 2021
    - ZeRO: Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training
      Trillion Parameter Models", 2020
    - GPipe: Huang et al., "GPipe: Efficient Training of Giant Neural Networks
      using Pipeline Parallelism", 2019
    - PipeDream: Narayanan et al., "PipeDream: Generalized Pipeline Parallelism
      for DNN Training", SOSP 2019
"""

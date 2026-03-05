"""
FlashAttention: Making Attention Fast by Respecting the Hardware

This module explains FlashAttention (v1 and v2) from the ground up.
Based on:
  - FlashAttention (2022): "FlashAttention: Fast and Memory-Efficient Exact Attention
    with IO-Awareness" (Tri Dao et al.)
  - FlashAttention-2 (2023): "FlashAttention-2: Faster Attention with Better Parallelism
    and Work Partitioning" (Tri Dao)

================================================================================
PART 1: THE PROBLEM - ATTENTION IS SLOW FOR THE WRONG REASON
================================================================================

Standard attention:

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V

For sequence length N and head dimension d:

    Q @ K^T:    N×d  @  d×N  =  N×N     (matrix multiply)
    softmax:    N×N  →  N×N              (elementwise + reduction)
    × V:        N×N  @  N×d  =  N×d     (matrix multiply)

The N×N matrix is the bottleneck:

    N = 2048, d = 128:
    Q@K^T:    2048 × 2048 = 4M entries = 8 MB (FP16)
    That's per head, per layer. With 32 heads, 96 layers:
    ~24 GB just for attention score matrices!

But here's the surprise:

    ATTENTION IS NOT SLOW BECAUSE OF COMPUTE.
    ATTENTION IS SLOW BECAUSE OF MEMORY ACCESS.

Let's look at what the GPU actually spends time on:

    Operation      FLOPs      Time on A100
    ────────────   ────────   ───────────
    Q @ K^T        ~8M        fast (matrix multiply, compute-bound)
    Softmax        ~12M       SLOW (elementwise, memory-bound!)
    Masking        ~4M        SLOW (elementwise, memory-bound!)
    Dropout        ~4M        SLOW (elementwise, memory-bound!)
    × V            ~8M        fast (matrix multiply, compute-bound)

    The non-matmul ops (softmax, masking, dropout) dominate runtime
    even though they do FEWER FLOPs. Why?

Because of the GPU memory hierarchy.


================================================================================
PART 2: THE GPU MEMORY HIERARCHY - WHY IO MATTERS
================================================================================

A GPU has two levels of memory that matter:

    ┌─────────────────────────────────┐
    │           SRAM (on-chip)         │
    │   Size:     ~20 MB total         │  (192 KB per streaming multiprocessor
    │   Speed:    ~19 TB/s             │   × 108 SMs on A100)
    │   Latency:  ~nanoseconds         │
    └────────────────┬────────────────┘
                     │ ← this transfer is the bottleneck
    ┌────────────────┴────────────────┐
    │           HBM (off-chip)         │
    │   Size:     80 GB                │
    │   Speed:    ~2 TB/s              │
    │   Latency:  ~hundreds of ns      │
    └─────────────────────────────────┘

    SRAM is ~10x FASTER than HBM, but ~4000x SMALLER.

    "HBM" = "High Bandwidth Memory" — ironic name, because the whole
    problem is that HBM is NOT fast enough relative to compute.

Every GPU operation works like this:

    1. Load data from HBM → SRAM          (SLOW)
    2. Compute on data in SRAM             (FAST)
    3. Write results from SRAM → HBM       (SLOW)

For memory-bound ops (softmax, dropout, masking):
    The compute in step 2 is trivial.
    Steps 1 and 3 dominate.

Now look at standard attention implementation:

    Step 1: Load Q, K from HBM    → compute S = Q@K^T   → write S to HBM
    Step 2: Load S from HBM       → compute P = softmax(S) → write P to HBM
    Step 3: Load P from HBM       → compute P = dropout(P) → write P to HBM
    Step 4: Load P, V from HBM    → compute O = P@V     → write O to HBM

    The N×N matrix S gets written to HBM, then read back, then written
    again as P, then read back AGAIN. That's 4 trips through the slow bus
    for data that could have stayed on-chip!

    Total HBM reads/writes: O(N² + N·d) — dominated by the N² terms.

This is what "IO-aware" means: design the algorithm to minimize
HBM ↔ SRAM transfers, not just minimize FLOPs.


================================================================================
PART 3: THE CORE IDEA - TILING + KERNEL FUSION
================================================================================

FlashAttention's insight: NEVER materialize the N×N attention matrix.

Instead:
    1. Load BLOCKS of Q, K, V into SRAM
    2. Compute attention for that block entirely in SRAM
    3. Write only the final output O back to HBM
    4. Repeat for next block

    Standard:  Q,K → [HBM] S → [HBM] P → [HBM] → O
               ↕ HBM  ↕ HBM  ↕ HBM  ↕ HBM      (many round trips)

    Flash:     Q,K,V → [SRAM: compute S,P,O all at once] → O
               ↕ HBM                                  ↕ HBM  (two trips!)

This is "kernel fusion": instead of separate GPU kernels for matmul,
softmax, dropout, and another matmul, fuse them into ONE kernel
that does everything in SRAM.

But there's a problem:

    The N×N attention matrix doesn't fit in SRAM!

    N = 2048, SRAM ≈ 192 KB per SM
    N×N in FP16 = 8 MB  ≫  192 KB

    We can't compute the full softmax in SRAM because softmax needs
    ALL scores in a row to compute the denominator:

        softmax(z_i) = exp(z_i) / Σ_j exp(z_j)
                                   ^^^^^^^^^^^^^
                                   needs ALL z values!

    If we only have a block of scores, we can't compute the correct softmax.

    ...or can we?


================================================================================
PART 4: ONLINE SOFTMAX - THE MATHEMATICAL TRICK
================================================================================

The key enabler: you CAN compute softmax incrementally, one block at a time,
and get the EXACT same result.

--- Standard softmax ---

    Given scores z = [z_1, z_2, ..., z_N]:

    softmax(z_i) = exp(z_i - m) / Σ_j exp(z_j - m)

    where m = max(z)  (for numerical stability)

    This requires seeing ALL of z before computing any output.

--- Online (incremental) softmax ---

    Process z in blocks of size B:

    Block 1: z[1:B]
    Block 2: z[B+1:2B]
    ...

    After each block, maintain two running statistics:
        m = running maximum (across all blocks seen so far)
        l = running sum of exp(z - m) (the denominator)

    When a new block arrives:

        m_new = max(m_old, max(z_block))
        l_new = l_old × exp(m_old - m_new) + Σ exp(z_block - m_new)
                        ^^^^^^^^^^^^^^^^
                        correction factor: old partial sums were computed
                        with a different max, so we rescale them

Let's trace through a concrete example:

    z = [2, 4, 1, 3, 5, 2]    block size B = 3

    Block 1: [2, 4, 1]
        m_1 = 4
        l_1 = exp(2-4) + exp(4-4) + exp(1-4)
            = exp(-2) + exp(0) + exp(-3)
            = 0.135 + 1.0 + 0.050
            = 1.185

        Partial softmax (WRONG for now, but will be corrected):
            p_1 = [exp(2-4), exp(4-4), exp(1-4)] / l_1
                = [0.135, 1.0, 0.050] / 1.185
                = [0.114, 0.844, 0.042]

    Block 2: [3, 5, 2]
        m_new = max(m_1=4, max(3,5,2)=5) = 5

        Correction: old l was computed with max=4, now max=5:
            l_new = l_1 × exp(4-5) + exp(3-5) + exp(5-5) + exp(2-5)
                  = 1.185 × 0.368 + 0.135 + 1.0 + 0.050
                  = 0.436 + 0.135 + 1.0 + 0.050
                  = 1.621

        Correct the old partial output:
            p_1_corrected = p_1 × (l_1 / l_new) × exp(m_1 - m_new)
                          = p_1 × (1.185 / 1.621) × exp(4 - 5)
                          = p_1 × 0.731 × 0.368
                          = p_1 × 0.269

        New partial softmax:
            p_2 = [exp(3-5), exp(5-5), exp(2-5)] / l_new
                = [0.135, 1.0, 0.050] / 1.621
                = [0.083, 0.617, 0.031]

    Final result: [0.114×0.269, 0.844×0.269, 0.042×0.269, 0.083, 0.617, 0.031]
                = [0.031, 0.227, 0.011, 0.083, 0.617, 0.031]

    Verify: these sum to 1.0 ✓ and match standard softmax ✓

The critical point: at NO stage did we need all 6 scores in memory at once.
We only ever needed 3 (one block) plus two scalars (m and l).


================================================================================
PART 5: THE FLASHATTENTION ALGORITHM
================================================================================

Now combine tiling with online softmax to compute attention without
ever materializing the N×N matrix:

    Inputs: Q, K, V in HBM    (each N × d)
    Output: O in HBM           (N × d)

    Block sizes: B_r rows of Q, B_c columns of K/V
    (chosen to maximize SRAM usage)

    # Initialize output accumulator and softmax statistics
    O = zeros(N, d)        # in HBM
    l = zeros(N)           # running softmax denominator, in HBM
    m = full(N, -inf)      # running max, in HBM

    # Outer loop: iterate over K, V blocks (columns of attention matrix)
    for j in range(0, N, B_c):
        # Load K_j and V_j from HBM to SRAM
        K_j = K[j : j+B_c]                    # [B_c, d]
        V_j = V[j : j+B_c]                    # [B_c, d]

        # Inner loop: iterate over Q blocks (rows of attention matrix)
        for i in range(0, N, B_r):
            # Load Q_i, O_i, l_i, m_i from HBM to SRAM
            Q_i = Q[i : i+B_r]                # [B_r, d]
            O_i = O[i : i+B_r]                # [B_r, d]
            l_i = l[i : i+B_r]                # [B_r]
            m_i = m[i : i+B_r]                # [B_r]

            # Compute attention scores for this block (STAYS IN SRAM)
            S_ij = Q_i @ K_j.T / sqrt(d)      # [B_r, B_c]  ← never written to HBM!

            # Online softmax update
            m_new = max(m_i, rowmax(S_ij))     # [B_r]
            P_ij = exp(S_ij - m_new)           # [B_r, B_c]  ← never written to HBM!
            l_new = l_i * exp(m_i - m_new) + rowsum(P_ij)  # [B_r]

            # Update output accumulator
            # Rescale old output (was computed with old max) + add new contribution
            O_i = O_i * (l_i * exp(m_i - m_new) / l_new)  +  P_ij @ V_j / l_new
            #     ↑ correct previous partial result              ↑ new block's contribution

            # Write updated O_i, l_new, m_new back to HBM
            O[i : i+B_r] = O_i
            l[i : i+B_r] = l_new
            m[i : i+B_r] = m_new

    # After all blocks: O contains the exact attention output!

What's happening visually:

    The N×N attention matrix (never fully created):

    K blocks →   j=0     j=B_c   j=2B_c
    Q blocks  ┌────────┬────────┬────────┐
    ↓         │        │        │        │
    i=0       │ S_00   │ S_01   │ S_02   │  ← computed in SRAM, immediately used
              │        │        │        │     for O update, then discarded
              ├────────┼────────┼────────┤
    i=B_r     │ S_10   │ S_11   │ S_12   │
              │        │        │        │
              ├────────┼────────┼────────┤
    i=2B_r    │ S_20   │ S_21   │ S_22   │
              │        │        │        │
              └────────┴────────┴────────┘

    Each block S_ij is computed, used, and FORGOTTEN.
    Only O (N×d) lives in HBM. The N×N matrix never exists!


================================================================================
PART 6: WHY THIS IS FASTER - IO COMPLEXITY ANALYSIS
================================================================================

HBM accesses for standard attention:

    Step 1: Read Q,K (2Nd), write S (N²)     →  2Nd + N²
    Step 2: Read S (N²), write P (N²)         →  2N²
    Step 3: Read P,V (N² + Nd), write O (Nd)  →  N² + 2Nd
    Total: 4N² + 4Nd  ≈  O(N²)               (N² dominates since N >> d)

HBM accesses for FlashAttention:

    Outer loop: T_c = N/B_c iterations
    Inner loop: T_r = N/B_r iterations

    Per inner iteration:
        Read Q_i (B_r×d), K_j (B_c×d), V_j (B_c×d), O_i (B_r×d)
        Write O_i (B_r×d)
        Total per iteration: O(B_r×d + B_c×d)

    Total: T_c × T_r × O(B_r×d + B_c×d)
         = (N/B_c) × (N/B_r) × O(B_r×d + B_c×d)
         = O(N²d / B_c + N²d / B_r)
         = O(N²d² / M)     where M = SRAM size (determines block sizes)

    Since d² / M is typically much less than 1 (d=128, M=192KB):

    Standard:  O(N²)     HBM accesses
    Flash:     O(N²d²/M) HBM accesses

    With typical values: d=128, M=100KB (usable):
    Ratio: d²/M = 16384 / 100000 ≈ 0.16

    FlashAttention does ~6x fewer HBM accesses!

Memory usage:

    Standard:  O(N²)   — must store the full N×N attention matrix
    Flash:     O(N)    — only store O (N×d), l (N), m (N). No N×N matrix!

    This is why FlashAttention enables longer sequences.
    N=16K with standard attention: 16K × 16K × 2 bytes = 512 MB per head
    N=16K with FlashAttention: just the O matrix + small bookkeeping


================================================================================
PART 7: THE BACKWARD PASS - RECOMPUTATION
================================================================================

For training, we need gradients. The backward pass of attention needs
the N×N attention probability matrix P.

Standard approach: save P during forward pass (O(N²) memory).

FlashAttention approach: DON'T save P. Instead:
    - Save only Q, K, V, O, l, m from forward pass
    - During backward, RECOMPUTE P from Q, K block by block (same tiling)

    This is just activation checkpointing applied to attention.

    Trade-off:
        Extra compute:  recompute P (one extra matmul per block)
        Memory saved:   O(N²) → O(N)

    Since attention is memory-bound, the extra compute is essentially free —
    the GPU cores would have been idle waiting for memory anyway!

    This is the counterintuitive insight: doing MORE computation can make
    things FASTER because the bottleneck was never compute to begin with.


================================================================================
PART 8: FLASHATTENTION-2 - SQUEEZING OUT MORE PERFORMANCE
================================================================================

FlashAttention v1 achieved 25-40% of theoretical max FLOPS on A100.
FlashAttention-2 pushes this to 50-73%. How?

Three improvements, all about better GPU utilization:

--- Improvement 1: Reduce non-matmul FLOPs ---

    FA-1 rescales the output O_i at every step:
        O_i = diag(l_i)^{-1} × exp(m_old - m_new) × l_old × O_i_old + ...

    These diag, exp, multiply operations are non-matmul FLOPs.
    On a GPU, matmul runs on specialized Tensor Cores (super fast).
    Non-matmul runs on regular CUDA cores (~16x slower!).

    FA-2 delays the rescaling: keep O_i un-normalized during the inner loop.
    Only divide by l at the VERY END after all blocks are processed.

    Before (FA-1): rescale O every block    → many non-matmul ops
    After  (FA-2): rescale O once at end    → fewer non-matmul ops

--- Improvement 2: Better parallelism across thread blocks ---

    FA-1: outer loop over K/V blocks, inner loop over Q blocks.
          For each K/V block, iterate over ALL Q blocks sequentially.

    Problem: if N is small or batch×heads is small, there aren't enough
    thread blocks to keep all 108 SMs busy.

    FA-2: SWAP the loops. Outer loop over Q blocks, inner loop over K/V.

    Why this helps:

        FA-1 (outer K, inner Q):
            Each thread block processes one K/V block against all Q blocks.
            Number of parallel thread blocks = N / B_c (number of K blocks)
            For N=2048, B_c=256: only 8 thread blocks. Many SMs idle!

        FA-2 (outer Q, inner K):
            Each thread block processes one Q block against all K/V blocks.
            Number of parallel thread blocks = N / B_r (number of Q blocks)
            Same parallelism, BUT each thread block writes to a DIFFERENT
            output row — no synchronization needed between thread blocks!

    Also: in FA-1, different thread blocks write to the SAME output rows
    and must coordinate (atomic adds or barriers). FA-2 avoids this.

--- Improvement 3: Better work partitioning within a thread block ---

    A thread block contains multiple "warps" (groups of 32 threads).
    FA-1 splits work so that warps must communicate through shared memory
    (SRAM) to combine their partial results.

    FA-2 restructures the computation so each warp handles a different
    portion of the output independently, eliminating warp-to-warp
    communication through shared memory.

    FA-1: "split-K" — warps split the K dimension, must reduce results
    FA-2: "split-Q" — warps split the Q dimension, write independently

    Result: fewer shared memory reads/writes within each thread block.

--- Combined effect ---

    FlashAttention-1:  25-40% of A100 peak TFLOPS
    FlashAttention-2:  50-73% of A100 peak TFLOPS  (~2x speedup)

    For context: optimized GEMM (matrix multiply) achieves ~80-90%.
    FlashAttention-2 gets close to that, which is remarkable for
    a fused kernel that does matmul + softmax + dropout together.

    End-to-end: training GPT-style models at 225 TFLOPS/s per A100
    (72% model FLOPs utilization).


================================================================================
PART 9: CAUSAL MASKING - HANDLING THE TRIANGULAR MASK
================================================================================

For autoregressive models (GPT), attention is CAUSAL: token i can only
attend to tokens 1..i, not future tokens.

    Standard: apply a mask to the N×N score matrix:
        S[i][j] = -inf  if j > i  (mask future positions)

    With FlashAttention tiling, some blocks are:
        - Fully unmasked (below the diagonal)  → compute normally
        - Fully masked (above the diagonal)    → skip entirely!
        - Partially masked (on the diagonal)   → apply mask within block

    ┌────────┬────────┬────────┐
    │ FULL   │ skip   │ skip   │
    ├────────┼────────┼────────┤
    │ FULL   │partial │ skip   │
    ├────────┼────────┼────────┤
    │ FULL   │ FULL   │partial │
    └────────┴────────┴────────┘

    Skipping upper-triangle blocks saves ~50% of computation!
    This is why FlashAttention is even faster for causal (GPT-style) models.


================================================================================
PART 10: SUMMARY - WHAT FLASHATTENTION ACTUALLY DOES
================================================================================

    Standard attention:
        1. Compute full N×N score matrix S       (write to HBM)
        2. Apply softmax to get P                (read/write HBM)
        3. Apply dropout                         (read/write HBM)
        4. Multiply by V to get O                (read/write HBM)
        Memory: O(N²)    HBM accesses: O(N²)

    FlashAttention:
        1. Load blocks of Q, K, V into SRAM      (one HBM read)
        2. Compute S, P, O for that block IN SRAM (no HBM access!)
        3. Use online softmax to maintain correct running totals
        4. Write only the output O back to HBM    (one HBM write)
        Memory: O(N)     HBM accesses: O(N²d²/M) ≈ O(N²/6)

    The N×N attention matrix NEVER EXISTS in HBM.
    It's computed block-by-block in SRAM and immediately consumed.

    Key ideas:
    ┌──────────────────────────────────────────────────────────────┐
    │  Idea                │ What it enables          │ Impact     │
    ├──────────────────────────────────────────────────────────────┤
    │  Tiling              │ Fit in SRAM              │ Foundation │
    │  Online softmax      │ Correct results w/tiles  │ Correctness│
    │  Kernel fusion       │ No intermediate HBM I/O  │ Speed (2-4x)│
    │  Recomputation (bwd) │ Don't save N×N matrix    │ Memory (O(N))│
    │  Loop reordering(v2) │ Better GPU utilization   │ Speed (2x) │
    │  Warp partitioning(v2)│ Less shared mem traffic │ Speed      │
    └──────────────────────────────────────────────────────────────┘

    FlashAttention is now the DEFAULT attention implementation in:
        - PyTorch 2.0+ (torch.nn.functional.scaled_dot_product_attention)
        - HuggingFace Transformers
        - All major LLM training frameworks

    It's not optional anymore. It's the standard.

Papers:
    - FlashAttention: Dao et al., "FlashAttention: Fast and Memory-Efficient
      Exact Attention with IO-Awareness", NeurIPS 2022
    - FlashAttention-2: Dao, "FlashAttention-2: Faster Attention with Better
      Parallelism and Work Partitioning", ICLR 2024
    - Online softmax: Milakov & Gimelshein, "Online normalizer calculation
      for softmax", 2018
"""

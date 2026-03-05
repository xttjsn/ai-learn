"""
LLM Serving: Batching, Scheduling, and PagedAttention

This module explains how LLM serving systems batch and schedule requests,
from naive approaches to vLLM's PagedAttention. Based on:
  - Orca (OSDI 2022): "Orca: A Distributed Serving System for Transformer-Based Generative Models"
  - vLLM (SOSP 2023): "Efficient Memory Management for LLM Serving with PagedAttention"

================================================================================
PART 1: WHY SERVING IS HARD - THE MEMORY WALL
================================================================================

You've learned that LLM inference has two phases:

    Prefill:  Process all prompt tokens in one big matrix multiply
    Decode:   Generate tokens one at a time, each needing all previous KV cache

Here's the key insight for serving:

    LLM inference is MEMORY-BANDWIDTH BOUND, not compute bound.

What does that mean? On an A100 GPU:

    Compute:  312 TFLOPS (FP16)
    Memory:   2 TB/s bandwidth, 80 GB capacity

    A 13B model has ~26 GB of parameters.
    Loading those parameters takes:  26 GB / 2 TB/s = 13 ms
    Computing on them takes:         ~0.2 ms  (for a single token)

    So the GPU spends 98% of its time LOADING weights, not computing.

This means: if you're loading the weights anyway, you might as well process
MANY tokens with those same weights. That's batching.

    Batch size 1:   Load 26 GB, compute 1 token    → 13 ms
    Batch size 32:  Load 26 GB, compute 32 tokens   → ~13.5 ms  (barely slower!)

    Throughput goes from 77 tokens/s to 2,370 tokens/s. ~30x improvement.

So the goal of a serving system is simple:

    MAXIMIZE BATCH SIZE → MAXIMIZE THROUGHPUT

But there's a catch...


================================================================================
PART 2: THE KV CACHE PROBLEM - WHY BATCHING LLMs IS DIFFERENT
================================================================================

In normal deep learning (image classification, etc.), batching is trivial:
stack inputs into a tensor, run forward pass, done.

LLMs are different because of the KV CACHE:

    Every token that has been processed creates K and V vectors
    that must be kept in GPU memory for ALL future tokens to attend to.

How big is this cache?

    For LLaMA-13B:
        - 40 layers
        - 40 heads per layer
        - 128 dimensions per head
        - 2 bytes per value (FP16)
        - 2 matrices (K and V)

    Per token: 40 × 40 × 128 × 2 × 2 = 819,200 bytes ≈ 800 KB

    For a sequence of 2048 tokens:
        2048 × 800 KB ≈ 1.6 GB  (per sequence!)

    On an A100 with 80 GB, after the model (26 GB), you have 54 GB left.
    That's only ~33 sequences of length 2048.

And here's the REAL problem:

    You DON'T KNOW how long each sequence will be.

    Request A might generate 10 tokens and finish.
    Request B might generate 2000 tokens.
    Request C might generate 50 tokens.

    They all arrive at different times and finish at different times.


================================================================================
PART 3: STATIC BATCHING - THE NAIVE APPROACH
================================================================================

The simplest approach: collect a batch of requests, run them all until
every request has finished, then collect the next batch.

    Time →
    ┌──────────────────────────────────┐
    │ Seq 1: [prompt] [tok1] [tok2] ■  │   ■ = end of sequence
    │ Seq 2: [prompt] [tok1] ... [tok8] ■ │
    │ Seq 3: [prompt] [tok1] ■ □ □ □ □ □ │   □ = wasted GPU cycles
    │ Seq 4: [prompt] [tok1] [tok2] [tok3] ■ □ □│
    └──────────────────────────────────┘
                                         ↑
                              Must wait for Seq 2 to finish
                              before starting ANY new requests

Problems:

    1. GPU UNDERUTILIZATION: Seq 3 finished after 2 decode steps but the
       GPU keeps "processing" it (padding with nothing) until Seq 2 finishes.

    2. WASTED MEMORY: Seq 3's KV cache sits in memory doing nothing, blocking
       new requests from being processed.

    3. HIGH LATENCY: New requests must wait for the entire batch to complete,
       even if there's free capacity.

    4. OVER-RESERVATION: Since you don't know output lengths in advance,
       you must reserve max_seq_len of KV cache per sequence. If max is 2048
       but average output is 100 tokens, you waste 95% of reserved memory.

Studies show existing systems waste 60-80% of KV cache memory due to
fragmentation and over-reservation. That directly limits batch size
and therefore throughput.


================================================================================
PART 4: CONTINUOUS BATCHING (ITERATION-LEVEL SCHEDULING)
================================================================================

This is the key insight from the Orca paper (OSDI 2022):

    Instead of scheduling at the REQUEST level (whole batch at once),
    schedule at the ITERATION level (every single decode step).

What does that mean concretely?

    At EVERY decode iteration:
        1. Check if any sequence just finished (emitted end-of-sequence token)
        2. If yes: remove it from the batch, free its memory
        3. Check if any new requests are waiting
        4. If yes AND there's memory: add them to the batch

    Time →
    ┌────────────────────────────────────────┐
    │ Seq 1: [prompt] [tok1] [tok2] ■         │
    │ Seq 2: [prompt] [tok1] ... [tok8] ■     │
    │ Seq 3: [prompt] [tok1] ■                │
    │         Seq 5: [prompt] [tok1] [tok2] ■ │  ← inserted when Seq 3 finished
    │ Seq 4: [prompt] [tok1] [tok2] [tok3] ■  │
    │         Seq 6: [prompt] [tok1] ■        │  ← inserted when Seq 4 finished
    │                  Seq 7: [prompt] [tok1]  │  ← inserted when Seq 1 finished
    └────────────────────────────────────────┘

    No wasted slots! As soon as a sequence finishes, a new one takes its place.

This is also called "cellular batching" or "dynamic batching" in some literature,
though "continuous batching" is the most common term.

The improvement is dramatic:

    Static batching:     ~100 tokens/s  (on typical workloads)
    Continuous batching: ~800 tokens/s  (8x improvement!)

But continuous batching alone doesn't solve the MEMORY problem. Each sequence
still pre-allocates a contiguous block of GPU memory for its full max KV cache.
That's where PagedAttention comes in.


================================================================================
PART 5: THE MEMORY FRAGMENTATION PROBLEM
================================================================================

Even with continuous batching, KV cache memory management is wasteful.

Traditional systems allocate KV cache like this:

    GPU Memory Layout:
    ┌─────────────────────────────────────────────────────┐
    │ Model Weights (fixed)                                │
    ├─────────────────────────────────────────────────────┤
    │ Seq 1 KV Cache: [████████████░░░░░░░░░░░░░░░░░░░]  │  █ = used, ░ = reserved but empty
    │ Seq 2 KV Cache: [██████░░░░░░░░░░░░░░░░░░░░░░░░░]  │
    │ Seq 3 KV Cache: [██████████████████████████████░░]  │
    │                        (gap - unusable!)             │
    │ Seq 4 KV Cache: [████░░░░░░░░░░░░░░░░░░░░░░░░░░░]  │
    └─────────────────────────────────────────────────────┘

Three kinds of waste:

    1. INTERNAL FRAGMENTATION: Each sequence reserves max_seq_len slots,
       but most are empty (the ░ regions).

    2. EXTERNAL FRAGMENTATION: When sequences finish and new ones start,
       they leave gaps that are too small for new allocations.

    3. RESERVATION WASTE: You must guess how much memory each request needs
       at arrival time. Guess too high → waste. Guess too low → crash.

Real-world measurements show:
    - Only 20-38% of KV cache memory is actually used for stored tokens
    - The rest is wasted on fragmentation and over-reservation

This is the EXACT same problem operating systems faced with physical memory
in the 1960s. And the solution is the same: VIRTUAL MEMORY + PAGING.


================================================================================
PART 6: PAGEDATTENTION - VIRTUAL MEMORY FOR KV CACHE
================================================================================

The core insight of vLLM:

    Treat KV cache like an OS treats process memory.
    Use paging to eliminate fragmentation.

Key analogy:

    OS Concept          →  vLLM Concept
    ─────────────────      ─────────────────
    Process             →  Sequence (request)
    Virtual address     →  Logical KV block index
    Physical page       →  Physical KV block in GPU memory
    Page table          →  Block table
    Page fault          →  Allocate new block on demand
    Byte                →  Token

How it works:

    1. Divide KV cache into fixed-size BLOCKS (e.g., 16 tokens per block)

    2. Each sequence has a BLOCK TABLE that maps logical blocks to physical blocks:

        Sequence 1 Block Table:
            Logical Block 0 → Physical Block 7
            Logical Block 1 → Physical Block 3
            Logical Block 2 → Physical Block 12

    3. Physical blocks are allocated ON DEMAND (not pre-reserved):

        Token 1-16:   Allocate physical block 7
        Token 17-32:  Allocate physical block 3   (not contiguous with 7!)
        Token 33-48:  Allocate physical block 12  (wherever there's space!)

    GPU Memory Layout with PagedAttention:
    ┌─────────────────────────────────────────────────────┐
    │ Model Weights (fixed)                                │
    ├─────────────────────────────────────────────────────┤
    │ Block  0: [Seq2, tokens 1-16 ]                       │
    │ Block  1: [Seq4, tokens 1-16 ]                       │
    │ Block  2: [Seq1, tokens 33-48]                       │
    │ Block  3: [Seq1, tokens 17-32]                       │
    │ Block  4: [Seq3, tokens 1-16 ]                       │
    │ Block  5: [Seq2, tokens 17-32]                       │
    │ Block  6: [Seq3, tokens 17-32]                       │
    │ Block  7: [Seq1, tokens 1-16 ]                       │
    │ Block  8: [FREE]                                     │
    │ Block  9: [Seq4, tokens 17-23] ← partially filled    │
    │ Block 10: [FREE]                                     │
    │ ...                                                  │
    └─────────────────────────────────────────────────────┘

    No contiguous allocation needed!
    No pre-reservation needed!
    Waste only happens in the LAST block of each sequence (< 4% waste).


Why this is brilliant:

    Traditional:
        - Allocate 2048 × 800 KB = 1.6 GB per sequence upfront
        - Even if sequence only uses 50 tokens (40 KB actual, 1.6 GB reserved)

    PagedAttention:
        - Allocate one block (16 × 800 KB = 12.5 KB) at a time
        - 50 tokens = 4 blocks = 50 KB allocated (only 12.5 KB wasted in last block)
        - Memory savings: 99.97% less waste!


================================================================================
PART 7: THE PAGEDATTENTION KERNEL - HOW ATTENTION WORKS WITH BLOCKS
================================================================================

Normal attention is simple:

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V

    Where K and V are contiguous tensors in memory.

With PagedAttention, K and V are scattered across non-contiguous blocks.
The kernel must:

    1. Look up the block table for the current sequence
    2. For each block: fetch the K and V data from the physical location
    3. Compute partial attention scores per block
    4. Combine the partial results correctly

Pseudocode for PagedAttention kernel (simplified):

    def paged_attention(query, block_table, k_cache, v_cache):
        '''
        query:       [num_heads, head_dim]        - current token's query
        block_table: [max_blocks]                  - maps logical → physical block ids
        k_cache:     [num_blocks, block_size, num_heads, head_dim]  - all K blocks
        v_cache:     [num_blocks, block_size, num_heads, head_dim]  - all V blocks
        '''
        output = zeros(num_heads, head_dim)

        for head in range(num_heads):
            q = query[head]                       # [head_dim]
            scores = []
            values = []

            for logical_idx in range(num_used_blocks):
                physical_idx = block_table[logical_idx]

                # Fetch this block's keys and values from wherever they live
                k_block = k_cache[physical_idx]   # [block_size, head_dim]
                v_block = v_cache[physical_idx]   # [block_size, head_dim]

                # Compute attention scores for this block
                block_scores = q @ k_block.T / sqrt(head_dim)   # [block_size]
                scores.append(block_scores)
                values.append(v_block)

            # Combine all blocks: softmax over ALL scores, then weighted sum of V
            all_scores = concatenate(scores)       # [total_tokens]
            all_values = concatenate(values)        # [total_tokens, head_dim]
            weights = softmax(all_scores)
            output[head] = weights @ all_values

        return output

    In practice, this is a fused CUDA kernel that does the block lookups
    and attention in a single GPU pass for efficiency.


================================================================================
PART 8: MEMORY SHARING - COPY-ON-WRITE FOR KV CACHE
================================================================================

PagedAttention enables another powerful optimization: MEMORY SHARING.

Use case: Parallel sampling (generate N completions from the same prompt).

    Without sharing:
        "Tell me a joke" → 3 parallel completions
        Prompt KV cache duplicated 3 times: 3 × (prompt_len × 800 KB)

    With PagedAttention sharing:
        All 3 sequences share the SAME physical blocks for the prompt!

    Sequence A block table: [Block 5, Block 9, Block 2, Block 11, ...]
    Sequence B block table: [Block 5, Block 9, Block 2, Block 14, ...]
    Sequence C block table: [Block 5, Block 9, Block 2, Block 8,  ...]
                             ^^^^^^^^^^^^^^^^^^^^^^^^
                             Same physical blocks!   Different only after
                                                     completions diverge

    This is exactly like UNIX fork() with copy-on-write pages:
        - Parent and child share pages
        - When either writes, copy the page first

    vLLM tracks REFERENCE COUNTS per physical block:
        Block 5: refcount = 3  (shared by A, B, C)
        Block 11: refcount = 1 (only A)

    When sequence A generates a new token that fills Block 2:
        - refcount(Block 2) > 1? YES → copy Block 2 to new Block 17
        - Update A's block table: logical block 2 → physical Block 17
        - Write new KV to Block 17
        - Decrement refcount of Block 2: 3 → 2

    Memory savings: up to 55% reduction for parallel sampling / beam search.
    Throughput improvement: up to 2.2x for these workloads.


================================================================================
PART 9: THE vLLM SCHEDULER - PUTTING IT ALL TOGETHER
================================================================================

vLLM's scheduler runs a loop every iteration:

    while True:
        # 1. CHECK FINISHED SEQUENCES
        for seq in running_batch:
            if seq.last_token == EOS or seq.length >= max_length:
                free_blocks(seq)
                return_result(seq)
                running_batch.remove(seq)

        # 2. TRY TO ADD NEW SEQUENCES
        while waiting_queue and can_allocate_blocks():
            new_seq = waiting_queue.pop()
            # Prefill phase: process all prompt tokens
            prefill(new_seq)
            # Allocate initial KV cache blocks
            allocate_blocks(new_seq, num_prompt_blocks)
            running_batch.add(new_seq)

        # 3. HANDLE MEMORY PRESSURE (preemption)
        while not enough_memory_for_one_decode_step():
            # Evict the lowest-priority sequence
            victim = running_batch.lowest_priority()
            # Option A: SWAP to CPU memory
            swap_to_cpu(victim)
            # Option B: RECOMPUTE - discard KV, re-prefill later
            discard_kv(victim)
            running_batch.remove(victim)
            swapped_queue.add(victim)

        # 4. DECODE ONE TOKEN for all running sequences
        for seq in running_batch:
            new_token = decode_step(seq)
            if needs_new_block(seq):
                allocate_block(seq)  # on-demand!

        # 5. TRY TO RESUME SWAPPED SEQUENCES
        while swapped_queue and can_allocate_blocks():
            victim = swapped_queue.pop()
            swap_from_cpu(victim)  # or re-prefill
            running_batch.add(victim)

Key design decisions:

    PREEMPTION: When memory is full and a new high-priority request arrives,
    vLLM can evict (preempt) running sequences. Two strategies:

        Swapping:    Copy KV blocks to CPU RAM. Resume later by copying back.
        Recomputation: Discard KV entirely. When resumed, re-run prefill.

        Swapping is faster to resume but uses CPU memory.
        Recomputation uses no extra memory but costs compute on resume.

    FIRST-COME-FIRST-SERVED scheduling: Simple but effective.
    Prevents starvation (long sequences don't block short ones forever
    because continuous batching lets short ones finish quickly).


================================================================================
PART 10: THE NUMBERS - WHY THIS MATTERS
================================================================================

Benchmark results from the vLLM paper (vs. state-of-the-art systems):

    Workload: ShareGPT traces (real chat conversations)
    Model: LLaMA-13B on A100 (40GB)

    System                          Throughput (req/s)
    ────────────────────────────    ──────────────────
    HuggingFace Transformers        ~1x  (baseline)
    FasterTransformer (NVIDIA)      ~1.5-2x
    Orca (continuous batching)      ~2-4x
    vLLM (PagedAttention)           ~14-24x

    Where does the improvement come from?

    Continuous batching alone:       ~2-4x  (no wasted slots)
    PagedAttention memory savings:   ~2-4x  (bigger batches via less waste)
    Memory sharing (parallel):       ~1.5-2x (for multi-output workloads)
    Combined:                        ~14-24x

    The improvement is MORE PRONOUNCED with:
        - Longer sequences (more KV cache waste to eliminate)
        - Larger models (tighter memory constraints)
        - Complex decoding (beam search, parallel sampling)

Memory efficiency comparison:

    System                KV Cache Utilization
    ────────────────      ────────────────────
    Static allocation     20-38%
    Orca                  ~50-60%  (continuous batching but still contiguous alloc)
    vLLM (PagedAttention) 96-98%   (near-zero waste, only last-block waste)


================================================================================
PART 11: SUMMARY - THE FULL PICTURE
================================================================================

Let's trace a request through the full serving stack:

    1. REQUEST ARRIVES
       "Explain quantum computing in simple terms"
       → Tokenized: [15496, 14821, 9145, 287, 2829, 2846]

    2. SCHEDULER PICKS IT UP (next iteration boundary)
       → Checks: is there memory for initial KV blocks? YES
       → Adds to running batch

    3. PREFILL PHASE
       → Process all 6 prompt tokens in one matrix multiply
       → Store K, V vectors in allocated blocks:
           Block table: [logical 0 → physical 42]  (6 tokens fit in one block)

    4. DECODE LOOP (repeats until done)
       → Generate next token
       → Append its K, V to the current block
       → If block full (16 tokens), allocate new physical block on demand
       → Block table grows: [phys 42, phys 17, phys 91, ...]
       → After each step, scheduler checks for finished/new sequences

    5. COMPLETION
       → Model outputs EOS token
       → Free all physical blocks (refcounts decremented)
       → Return generated text to user

    6. MEANWHILE: Other sequences in the batch are doing steps 3-5
       concurrently. The batch size fluctuates every iteration as
       sequences arrive and finish.

The key innovations, in order of impact:

    ┌─────────────────────────────────────────────────────────────┐
    │  Innovation              │ What it solves        │ Gain     │
    ├─────────────────────────────────────────────────────────────┤
    │  Continuous batching      │ GPU slot waste        │ 2-4x    │
    │  PagedAttention          │ Memory fragmentation  │ 2-4x    │
    │  On-demand allocation    │ Over-reservation      │ (part of above)│
    │  Copy-on-write sharing   │ Duplicate KV cache    │ 1.5-2x  │
    │  Preemption (swap/recomp)│ Memory deadlocks      │ robustness│
    └─────────────────────────────────────────────────────────────┘

    Combined effect: 14-24x throughput improvement over naive serving.

This is why vLLM became the de facto standard for LLM serving in production.

Papers:
    - Orca: Yu et al., "Orca: A Distributed Serving System for
      Transformer-Based Generative Models", OSDI 2022
    - vLLM: Kwon et al., "Efficient Memory Management for Large Language
      Model Serving with PagedAttention", SOSP 2023
"""

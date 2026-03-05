"""
KV Cache and Attention Variants: From MHA to GQA to Sliding Window

This module explains KV cache optimization strategies with working code.
Based on:
  - Multi-Query Attention (Shazeer, 2019)
  - Grouped-Query Attention / GQA (Ainslie et al., 2023)
  - Sliding Window Attention / Mistral (Jiang et al., 2023)
  - Ring Attention (Liu et al., 2023)

================================================================================
PART 1: THE KV CACHE PROBLEM
================================================================================

During autoregressive generation, we cache the Key and Value tensors
from all previous tokens to avoid recomputing them.

    For each new token, attention needs:
        Q: just the new token's query     (1 × d)
        K: ALL previous tokens' keys      (seq_len × d)
        V: ALL previous tokens' values    (seq_len × d)

    Without cache: recompute K,V for all previous tokens every step
        → O(seq_len²) total compute for generating seq_len tokens

    With cache: store K,V, only compute new token's K,V
        → O(seq_len) per step, but need to STORE the cache

KV cache size for standard Multi-Head Attention (MHA):

    For Llama-70B:
        Layers: 80
        Heads: 64
        Head dim: 128
        Sequence length: 4096

    KV cache per token per layer:
        K: num_heads × head_dim = 64 × 128 = 8192 values
        V: num_heads × head_dim = 64 × 128 = 8192 values
        Total: 16,384 values × 2 bytes (FP16) = 32 KB

    KV cache per token (all layers):
        80 layers × 32 KB = 2.56 MB PER TOKEN

    KV cache for one request at 4096 tokens:
        4096 × 2.56 MB = 10.5 GB

    KV cache for batch of 32 requests:
        32 × 10.5 GB = 336 GB  ← MORE than the model weights!

    ┌─────────────────────────────────────────────────────┐
    │  Memory breakdown for Llama-70B serving (batch=32):  │
    │                                                     │
    │  Model weights:  140 GB (FP16)                      │
    │  KV cache:       336 GB                             │
    │  Total:          476 GB → needs 6× H100s            │
    │                                                     │
    │  The KV cache is 2.4× the model weights!             │
    │  And it scales linearly with:                        │
    │    - batch size                                      │
    │    - sequence length                                 │
    │    - number of heads                                 │
    │    - head dimension                                  │
    └─────────────────────────────────────────────────────┘

Every optimization in this module reduces the KV cache size.


================================================================================
PART 2: MULTI-HEAD ATTENTION (MHA) — THE BASELINE
================================================================================

Standard transformer attention with separate K,V per head:

    Q = x @ W_Q    → split into num_heads heads
    K = x @ W_K    → split into num_heads heads
    V = x @ W_V    → split into num_heads heads

    Each head h:
        attn_h = softmax(Q_h @ K_h^T / √d) @ V_h

    All heads have INDEPENDENT K and V projections.
    → KV cache stores num_heads × head_dim × 2 values per token per layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention (MHA).
    Every head has its own K and V — maximum expressiveness, maximum memory.
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, S, D = x.shape

        Q = self.W_Q(x).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Append to KV cache
        if kv_cache is not None:
            K = torch.cat([kv_cache[0], K], dim=2)
            V = torch.cat([kv_cache[1], V], dim=2)

        new_cache = (K, V)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).reshape(B, S, D)
        return self.W_O(out), new_cache

    def kv_cache_size(self, seq_len: int, batch: int = 1) -> int:
        """KV cache size in bytes (FP16)."""
        return batch * 2 * self.num_heads * seq_len * self.head_dim * 2


# ============================================================================
# PART 3: MULTI-QUERY ATTENTION (MQA) — SINGLE KV HEAD
# ============================================================================

"""
Multi-Query Attention (Shazeer, 2019):

    Instead of num_heads K,V projections, use just ONE K and ONE V.
    All query heads share the same K and V.

    ┌─────────────────────────────────────────────────────┐
    │  MHA:  Q has 64 heads, K has 64 heads, V has 64 heads│
    │  MQA:  Q has 64 heads, K has 1 head,  V has 1 head  │
    │                                                     │
    │  KV cache reduction: 64× !!                          │
    │                                                     │
    │  MHA for Llama-70B:  2.56 MB/token → 10.5 GB@4K     │
    │  MQA for Llama-70B:  0.04 MB/token → 0.16 GB@4K     │
    │                                                     │
    │  But: quality drops because one KV head can't express│
    │       as much information as 64 independent heads.   │
    └─────────────────────────────────────────────────────┘
"""


class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention: One K head, One V head, multiple Q heads.
    Massive KV cache savings at the cost of some quality.
    Used by: PaLM, Falcon, StarCoder.
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        # Only one head for K and V!
        self.W_K = nn.Linear(d_model, self.head_dim, bias=False)
        self.W_V = nn.Linear(d_model, self.head_dim, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, S, D = x.shape

        Q = self.W_Q(x).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        # K, V: single head
        K = self.W_K(x).reshape(B, S, 1, self.head_dim).transpose(1, 2)
        V = self.W_V(x).reshape(B, S, 1, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            K = torch.cat([kv_cache[0], K], dim=2)
            V = torch.cat([kv_cache[1], V], dim=2)

        new_cache = (K, V)

        # Broadcast K, V across all heads
        K_expanded = K.expand(-1, self.num_heads, -1, -1)
        V_expanded = V.expand(-1, self.num_heads, -1, -1)

        scores = torch.matmul(Q, K_expanded.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V_expanded)

        out = out.transpose(1, 2).reshape(B, S, D)
        return self.W_O(out), new_cache

    def kv_cache_size(self, seq_len: int, batch: int = 1) -> int:
        """KV cache size in bytes (FP16). Just 1 head for K and V."""
        return batch * 2 * 1 * seq_len * self.head_dim * 2


# ============================================================================
# PART 4: GROUPED-QUERY ATTENTION (GQA) — THE MIDDLE GROUND
# ============================================================================

"""
GQA: Instead of 1 KV head (MQA) or num_heads KV heads (MHA),
     use num_kv_heads where 1 < num_kv_heads < num_heads.

    ┌─────────────────────────────────────────────────────┐
    │  Llama-2 70B:  num_heads=64, num_kv_heads=8 (GQA-8)│
    │                                                     │
    │  Group size = 64/8 = 8 query heads per KV head      │
    │                                                     │
    │  Q heads: [0,1,2,3,4,5,6,7 | 8,9,...,15 | ... ]    │
    │  K heads: [     KV_0       |    KV_1    | ... ]     │
    │  V heads: [     KV_0       |    KV_1    | ... ]     │
    │                                                     │
    │  8 query heads share 1 KV head                       │
    │                                                     │
    │  KV cache: 8× smaller than MHA, 8× larger than MQA  │
    │  Quality: much better than MQA, close to MHA         │
    │                                                     │
    │  MHA:  64 KV heads → 2.56 MB/token                   │
    │  GQA:  8  KV heads → 0.32 MB/token (8× savings)     │
    │  MQA:  1  KV head  → 0.04 MB/token (64× savings)    │
    │                                                     │
    │  GQA is the sweet spot: good quality + good savings   │
    │  Used by: Llama 2/3 (70B), Mistral, Gemma            │
    └─────────────────────────────────────────────────────┘
"""


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA).
    Multiple Q heads share each KV head.
    """
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_heads // num_kv_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model

        self.W_Q = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.W_K = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.W_V = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, S, D = x.shape

        Q = self.W_Q(x).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).reshape(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).reshape(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            K = torch.cat([kv_cache[0], K], dim=2)
            V = torch.cat([kv_cache[1], V], dim=2)

        new_cache = (K, V)

        # Expand KV heads to match Q heads
        # Each KV head is shared by group_size Q heads
        K_expanded = K.repeat_interleave(self.group_size, dim=1)
        V_expanded = V.repeat_interleave(self.group_size, dim=1)

        scores = torch.matmul(Q, K_expanded.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V_expanded)

        out = out.transpose(1, 2).reshape(B, S, D)
        return self.W_O(out), new_cache

    def kv_cache_size(self, seq_len: int, batch: int = 1) -> int:
        return batch * 2 * self.num_kv_heads * seq_len * self.head_dim * 2


# ============================================================================
# PART 5: SLIDING WINDOW ATTENTION (SWA)
# ============================================================================

"""
Sliding Window Attention (used in Mistral, Mixtral):

    Instead of attending to ALL previous tokens, attend only to
    the last W tokens (the "window").

    Standard attention:  token at position T attends to [0, 1, ..., T]
    Sliding window:      token at position T attends to [T-W, ..., T]

    ┌─────────────────────────────────────────────────────┐
    │  Standard causal attention (seq_len=8):              │
    │                                                     │
    │  Token 0: [X . . . . . . .]                         │
    │  Token 1: [X X . . . . . .]                         │
    │  Token 2: [X X X . . . . .]                         │
    │  Token 3: [X X X X . . . .]                         │
    │  Token 4: [X X X X X . . .]                         │
    │  Token 5: [X X X X X X . .]                         │
    │  Token 6: [X X X X X X X .]                         │
    │  Token 7: [X X X X X X X X]                         │
    │                                                     │
    │  KV cache grows linearly with sequence length!       │
    │                                                     │
    │  Sliding window attention (W=3):                     │
    │                                                     │
    │  Token 0: [X . . . . . . .]                         │
    │  Token 1: [X X . . . . . .]                         │
    │  Token 2: [X X X . . . . .]                         │
    │  Token 3: [. X X X . . . .]                         │
    │  Token 4: [. . X X X . . .]                         │
    │  Token 5: [. . . X X X . .]                         │
    │  Token 6: [. . . . X X X .]                         │
    │  Token 7: [. . . . . X X X]                         │
    │                                                     │
    │  KV cache is BOUNDED at W tokens!                    │
    │  No matter how long the sequence.                    │
    └─────────────────────────────────────────────────────┘

    But wait — doesn't this lose long-range information?

    With STACKED layers, information propagates further:

    Layer 1: each token sees W tokens
    Layer 2: each token sees 2W tokens (through layer 1's outputs)
    Layer L: each token sees L×W tokens!

    Mistral: W=4096, 32 layers → effective range = 131,072 tokens
    So the model CAN use long-range info, it just takes multiple layers.

    KV cache savings:
    Llama-70B at 128K tokens: 128K × 2.56 MB = 327 GB per request
    Mistral (W=4096) at 128K: 4096 × 2.56 MB = 10.5 GB per request
    Savings: 31× !
"""


class SlidingWindowAttention(nn.Module):
    """Attention with a sliding window — bounded KV cache."""
    def __init__(self, d_model: int, num_heads: int, window_size: int = 4096):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        self.d_model = d_model

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, S, D = x.shape

        Q = self.W_Q(x).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            K = torch.cat([kv_cache[0], K], dim=2)
            V = torch.cat([kv_cache[1], V], dim=2)

        # Evict old entries beyond window size
        if K.shape[2] > self.window_size:
            K = K[:, :, -self.window_size:, :]
            V = V[:, :, -self.window_size:, :]

        new_cache = (K, V)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).reshape(B, S, D)
        return self.W_O(out), new_cache

    def kv_cache_size(self, seq_len: int, batch: int = 1) -> int:
        effective_len = min(seq_len, self.window_size)
        return batch * 2 * self.num_heads * effective_len * self.head_dim * 2


# ============================================================================
# PART 6: QUANTIZED KV CACHE
# ============================================================================

"""
Orthogonal optimization: quantize the KV cache from FP16 to INT8 or FP8.

    FP16 KV cache:  2 bytes per value
    INT8 KV cache:  1 byte per value (+ small scale/zero-point overhead)
    FP8  KV cache:  1 byte per value

    Savings: 2× memory reduction → 2× more requests or 2× longer sequences

    Quality impact: minimal (0.1-0.3% perplexity increase)

    ┌──────────────────────────────────────────────┐
    │  Quantization per head per layer:             │
    │                                              │
    │  FP16 values: [0.123, -0.456, 0.789, ...]   │
    │                                              │
    │  INT8 quantization:                           │
    │    scale = (max - min) / 255                  │
    │    zero_point = round(-min / scale)           │
    │    quantized = round(value / scale) + zp      │
    │                                              │
    │  Per-token quantization: one scale per token  │
    │  Per-head quantization: one scale per head    │
    │  Per-channel: one scale per channel (best)    │
    └──────────────────────────────────────────────┘
"""


class QuantizedKVCache:
    """
    INT8 quantized KV cache for memory savings.
    Stores K,V in INT8 with per-token scale factors.
    """
    def __init__(self):
        self.k_quantized: Optional[torch.Tensor] = None  # INT8
        self.v_quantized: Optional[torch.Tensor] = None  # INT8
        self.k_scales: Optional[torch.Tensor] = None     # FP16
        self.v_scales: Optional[torch.Tensor] = None     # FP16
        self.k_zeros: Optional[torch.Tensor] = None      # FP16
        self.v_zeros: Optional[torch.Tensor] = None      # FP16

    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize FP16 tensor to INT8 per-token."""
        # Per-token min/max (last dim = head_dim)
        t_min = tensor.amin(dim=-1, keepdim=True)
        t_max = tensor.amax(dim=-1, keepdim=True)

        scale = (t_max - t_min) / 255.0
        scale = torch.clamp(scale, min=1e-8)
        zero_point = torch.round(-t_min / scale)

        quantized = torch.clamp(torch.round(tensor / scale) + zero_point, 0, 255)
        return quantized.to(torch.uint8), scale, zero_point

    def dequantize(self, quantized: torch.Tensor, scale: torch.Tensor,
                   zero_point: torch.Tensor) -> torch.Tensor:
        """Dequantize INT8 tensor back to FP16."""
        return (quantized.float() - zero_point) * scale

    def append(self, k: torch.Tensor, v: torch.Tensor):
        """Append new K,V (in FP16) to the quantized cache."""
        k_q, k_s, k_z = self.quantize(k)
        v_q, v_s, v_z = self.quantize(v)

        if self.k_quantized is None:
            self.k_quantized, self.k_scales, self.k_zeros = k_q, k_s, k_z
            self.v_quantized, self.v_scales, self.v_zeros = v_q, v_s, v_z
        else:
            self.k_quantized = torch.cat([self.k_quantized, k_q], dim=2)
            self.k_scales = torch.cat([self.k_scales, k_s], dim=2)
            self.k_zeros = torch.cat([self.k_zeros, k_z], dim=2)
            self.v_quantized = torch.cat([self.v_quantized, v_q], dim=2)
            self.v_scales = torch.cat([self.v_scales, v_s], dim=2)
            self.v_zeros = torch.cat([self.v_zeros, v_z], dim=2)

    def get_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get dequantized K,V tensors."""
        k = self.dequantize(self.k_quantized, self.k_scales, self.k_zeros)
        v = self.dequantize(self.v_quantized, self.v_scales, self.v_zeros)
        return k, v

    def memory_bytes(self) -> int:
        if self.k_quantized is None:
            return 0
        # INT8 values + FP16 scales + FP16 zero points
        kv_bytes = (self.k_quantized.numel() + self.v_quantized.numel()) * 1
        scale_bytes = (self.k_scales.numel() + self.v_scales.numel()) * 2
        zero_bytes = (self.k_zeros.numel() + self.v_zeros.numel()) * 2
        return kv_bytes + scale_bytes + zero_bytes


# ============================================================================
# PART 7: RING ATTENTION FOR ULTRA-LONG SEQUENCES
# ============================================================================

"""
Ring Attention: distribute a long sequence across GPUs in a ring.

    Problem: sequence of 1M tokens → KV cache doesn't fit on one GPU.

    Solution: split the sequence across N GPUs.
    Each GPU holds seq_len/N tokens.
    Attention is computed in a ring pattern:

    ┌───────────────────────────────────────────────────────┐
    │  Ring Attention with 4 GPUs, seq_len=1M:              │
    │                                                       │
    │  GPU 0: tokens [0, 250K)     → Q chunk 0              │
    │  GPU 1: tokens [250K, 500K)  → Q chunk 1              │
    │  GPU 2: tokens [500K, 750K)  → Q chunk 2              │
    │  GPU 3: tokens [750K, 1M)    → Q chunk 3              │
    │                                                       │
    │  Round 1: Each GPU computes attention with LOCAL KV   │
    │    GPU 0: Q0 × K0,V0                                  │
    │    GPU 1: Q1 × K1,V1                                  │
    │    GPU 2: Q2 × K2,V2                                  │
    │    GPU 3: Q3 × K3,V3                                  │
    │                                                       │
    │  Round 2: Pass KV to next GPU in ring                  │
    │    GPU 0: Q0 × K3,V3 (received from GPU 3)            │
    │    GPU 1: Q1 × K0,V0 (received from GPU 0)            │
    │    GPU 2: Q2 × K1,V1 (received from GPU 1)            │
    │    GPU 3: Q3 × K2,V2 (received from GPU 2)            │
    │                                                       │
    │  Round 3: Pass KV again...                             │
    │  Round 4: Pass KV again... (all pairs computed)        │
    │                                                       │
    │  After N rounds: each GPU has attended to ALL tokens!  │
    │  But NEVER needs more than seq_len/N tokens in memory. │
    │                                                       │
    │  Communication: overlapped with computation            │
    │  While computing attention for current KV chunk,       │
    │  the NEXT KV chunk is being transferred.               │
    └───────────────────────────────────────────────────────┘

    Memory: O(seq_len / N) per GPU instead of O(seq_len)
    Communication: O(seq_len / N) per round × N rounds = O(seq_len) total
    But with overlap: communication is mostly hidden behind compute.

    This enables training with 1M+ token sequences!
"""


# ============================================================================
# PART 8: DEMO — COMPARING ALL VARIANTS
# ============================================================================

def demo():
    print("=" * 70)
    print("KV CACHE AND ATTENTION VARIANTS COMPARISON")
    print("=" * 70)

    d_model = 512
    num_heads = 8
    head_dim = d_model // num_heads
    batch_size = 1
    seq_len = 64

    # Create all variants
    mha = MultiHeadAttention(d_model, num_heads)
    mqa = MultiQueryAttention(d_model, num_heads)
    gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads=2)
    swa = SlidingWindowAttention(d_model, num_heads, window_size=16)

    x = torch.randn(batch_size, seq_len, d_model)

    print(f"\nConfiguration: d_model={d_model}, heads={num_heads}, seq={seq_len}")

    # ── KV cache size comparison ──
    print(f"\n{'─' * 70}")
    print("KV CACHE SIZE COMPARISON")
    print(f"{'─' * 70}")

    variants = [
        ("MHA (8 KV heads)", mha),
        ("MQA (1 KV head)", mqa),
        ("GQA (2 KV heads)", gqa),
        ("SWA (window=16)", swa),
    ]

    for name, module in variants:
        size = module.kv_cache_size(seq_len, batch_size)
        print(f"  {name:25s}: {size:8,} bytes ({size/1024:.1f} KB)")

    # ── Forward pass verification ──
    print(f"\n{'─' * 70}")
    print("FORWARD PASS (verify all produce valid output)")
    print(f"{'─' * 70}")

    for name, module in variants:
        out, cache = module(x)
        print(f"  {name:25s}: output shape {list(out.shape)}, "
              f"K cache shape {list(cache[0].shape)}")

    # ── Autoregressive generation with KV cache ──
    print(f"\n{'─' * 70}")
    print("AUTOREGRESSIVE GENERATION (token by token)")
    print(f"{'─' * 70}")

    for name, module in variants:
        # Process initial prompt
        prompt = torch.randn(1, 4, d_model)
        _, cache = module(prompt)

        # Generate 10 tokens autoregressively
        for step in range(10):
            new_token = torch.randn(1, 1, d_model)
            _, cache = module(new_token, kv_cache=cache)

        k_shape = list(cache[0].shape)
        cache_bytes = cache[0].numel() * 2 + cache[1].numel() * 2
        print(f"  {name:25s}: K shape after 14 tokens: {k_shape}, "
              f"cache: {cache_bytes:,} bytes")

    # ── Quantized KV cache demo ──
    print(f"\n{'─' * 70}")
    print("QUANTIZED KV CACHE (INT8)")
    print(f"{'─' * 70}")

    # Simulate KV cache entries
    q_cache = QuantizedKVCache()
    fp16_cache_bytes = 0

    for _ in range(seq_len):
        k = torch.randn(1, num_heads, 1, head_dim, dtype=torch.float16)
        v = torch.randn(1, num_heads, 1, head_dim, dtype=torch.float16)
        q_cache.append(k.float(), v.float())
        fp16_cache_bytes += (k.numel() + v.numel()) * 2

    int8_bytes = q_cache.memory_bytes()
    print(f"  FP16 KV cache: {fp16_cache_bytes:,} bytes")
    print(f"  INT8 KV cache: {int8_bytes:,} bytes")
    print(f"  Savings: {(1 - int8_bytes/fp16_cache_bytes)*100:.0f}%")

    # Verify dequantization accuracy
    k_deq, v_deq = q_cache.get_kv()
    print(f"  Dequantized shape: K={list(k_deq.shape)}, V={list(v_deq.shape)}")

    # ── Scale to real models ──
    print(f"\n{'─' * 70}")
    print("REAL MODEL KV CACHE SIZES (batch=32, seq=4096)")
    print(f"{'─' * 70}")

    models = [
        ("Llama-70B MHA", 80, 64, 64, 128, None),
        ("Llama-70B GQA-8", 80, 64, 8, 128, None),
        ("Llama-70B MQA", 80, 64, 1, 128, None),
        ("Mistral-7B SWA-4K", 32, 32, 8, 128, 4096),
    ]

    batch, seq = 32, 4096

    print(f"\n  {'Model':25s} {'KV Cache':>12s} {'Per Request':>12s}")
    print(f"  {'─' * 25} {'─' * 12} {'─' * 12}")

    for name, layers, q_heads, kv_heads, hdim, window in models:
        effective_seq = min(seq, window) if window else seq
        kv_bytes = batch * layers * 2 * kv_heads * effective_seq * hdim * 2
        per_req = kv_bytes / batch
        print(f"  {name:25s} {kv_bytes / 1e9:9.1f} GB {per_req / 1e6:9.1f} MB")

    # Long sequence comparison
    print(f"\n  At 128K tokens (batch=1):")
    seq = 131072
    for name, layers, q_heads, kv_heads, hdim, window in models:
        effective_seq = min(seq, window) if window else seq
        kv_bytes = 1 * layers * 2 * kv_heads * effective_seq * hdim * 2
        print(f"  {name:25s} {kv_bytes / 1e9:9.1f} GB")

    print(f"\n{'=' * 70}")
    print("KEY TAKEAWAYS:")
    print("  1. MHA: max quality, max memory (64 KV heads)")
    print("  2. MQA: 64× KV savings, but quality drops (1 KV head)")
    print("  3. GQA: best tradeoff — 8× savings, near-MHA quality (Llama 2/3)")
    print("  4. SWA: bounded KV cache regardless of seq length (Mistral)")
    print("  5. INT8 KV cache: 2× savings orthogonal to all above")
    print("  6. Ring Attention: distribute long sequences across GPUs")
    print("=" * 70)


if __name__ == "__main__":
    demo()

"""
Quantization: Making Models Smaller and Faster

This module explains quantization techniques from the ground up with
working implementations.
Based on:
  - GPTQ (Frantar et al., 2023)
  - AWQ (Lin et al., 2024)
  - SmoothQuant (Xiao et al., 2023)
  - FP8 training (Micikevicius et al., 2022)

================================================================================
PART 1: WHY QUANTIZE?
================================================================================

A 70B parameter model in FP16:
    Memory: 70B × 2 bytes = 140 GB
    H100 GPU: 80 GB
    → Doesn't fit on one GPU!

    Quantize to INT4:
    Memory: 70B × 0.5 bytes = 35 GB
    → Fits on ONE GPU!

    Plus: INT4 compute is faster (less data to move through HBM).

    ┌─────────────────────────────────────────────────────┐
    │  QUANTIZATION IMPACT FOR LLAMA-70B:                  │
    │                                                     │
    │  Precision   Memory    GPUs needed   Tok/s (approx) │
    │  ─────────   ──────    ──────────    ────────────── │
    │  FP32        280 GB    4× H100       ~12            │
    │  FP16        140 GB    2× H100       ~24            │
    │  INT8         70 GB    1× H100       ~45            │
    │  INT4         35 GB    1× H100       ~60            │
    │  FP8          70 GB    1× H100       ~50 (native!)  │
    │                                                     │
    │  FP16→INT4: 4× smaller, ~2.5× faster, ~1% quality   │
    └─────────────────────────────────────────────────────┘


================================================================================
PART 2: NUMBER FORMAT BASICS
================================================================================

    FP32 (32 bits): 1 sign + 8 exponent + 23 mantissa
        Range: ±3.4 × 10^38
        Precision: ~7 decimal digits
        Size: 4 bytes

    FP16 (16 bits): 1 sign + 5 exponent + 10 mantissa
        Range: ±65504
        Precision: ~3.3 decimal digits
        Size: 2 bytes
        Problem: limited range → overflow during training!

    BF16 (16 bits): 1 sign + 8 exponent + 7 mantissa
        Range: ±3.4 × 10^38 (same as FP32!)
        Precision: ~2.4 decimal digits
        Size: 2 bytes
        Used by: most modern LLM training (Llama, GPT-4)

    FP8 E4M3 (8 bits): 1 sign + 4 exponent + 3 mantissa
        Range: ±448
        Precision: ~1.7 decimal digits
        Size: 1 byte
        Used for: weights, activations (H100 Tensor Cores)

    FP8 E5M2 (8 bits): 1 sign + 5 exponent + 2 mantissa
        Range: ±57344
        Precision: ~1.2 decimal digits
        Size: 1 byte
        Used for: gradients (wider range needed)

    INT8 (8 bits): signed integer [-128, 127]
        256 uniformly spaced values
        Size: 1 byte

    INT4 (4 bits): signed integer [-8, 7]
        16 uniformly spaced values
        Size: 0.5 bytes

    ┌──────────────────────────────────────────────┐
    │  Visual comparison (values representable):    │
    │                                              │
    │  FP32: .............................(millions)│
    │  FP16: ................(~65K unique values)   │
    │  BF16: ........(~65K, but wider range)        │
    │  FP8:  ....(256 unique values)                │
    │  INT8: ||||||||||||||||  (256 uniform steps)  │
    │  INT4: ||||  (16 uniform steps)               │
    └──────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


# ============================================================================
# PART 3: BASIC QUANTIZATION — SYMMETRIC AND ASYMMETRIC
# ============================================================================

def symmetric_quantize(tensor: torch.Tensor, bits: int = 8
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric quantization: maps [-max_abs, +max_abs] to [-2^(b-1), 2^(b-1)-1]

    The zero point is always 0 (symmetric around zero).

        scale = max(|tensor|) / (2^(b-1) - 1)
        quantized = round(tensor / scale)
        dequantized = quantized × scale

    ┌──────────────────────────────────────────────┐
    │  Example (INT8, symmetric):                   │
    │                                              │
    │  FP values:  [-1.5, -0.5, 0.0, 0.5, 1.2]   │
    │  max_abs = 1.5                                │
    │  scale = 1.5 / 127 = 0.01181                 │
    │                                              │
    │  Quantized: [-127, -42, 0, 42, 102]          │
    │  Dequant:   [-1.5, -0.496, 0.0, 0.496, 1.204]│
    │                                              │
    │  Error: small! (rounding error only)          │
    └──────────────────────────────────────────────┘
    """
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1

    max_abs = tensor.abs().max()
    scale = max_abs / qmax
    scale = torch.clamp(scale, min=1e-10)

    quantized = torch.clamp(torch.round(tensor / scale), qmin, qmax).to(torch.int8)
    return quantized, scale


def symmetric_dequantize(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return quantized.float() * scale


def asymmetric_quantize(tensor: torch.Tensor, bits: int = 8
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Asymmetric quantization: maps [min, max] to [0, 2^b - 1]

    Uses a zero-point offset for distributions not centered at zero.

        scale = (max - min) / (2^b - 1)
        zero_point = round(-min / scale)
        quantized = round(tensor / scale) + zero_point

    Better for activations (often non-negative after ReLU).

    ┌──────────────────────────────────────────────┐
    │  Example (INT8, asymmetric):                  │
    │                                              │
    │  FP values:  [0.0, 0.5, 1.0, 1.5, 2.0]     │
    │  All non-negative! Symmetric would waste      │
    │  half the range on unused negative values.    │
    │                                              │
    │  min=0.0, max=2.0                             │
    │  scale = 2.0 / 255 = 0.00784                 │
    │  zero_point = 0                               │
    │                                              │
    │  Quantized: [0, 64, 128, 191, 255]           │
    │  Full range used!                             │
    └──────────────────────────────────────────────┘
    """
    qmin = 0
    qmax = 2 ** bits - 1

    t_min = tensor.min()
    t_max = tensor.max()
    scale = (t_max - t_min) / (qmax - qmin)
    scale = torch.clamp(scale, min=1e-10)
    zero_point = torch.clamp(torch.round(-t_min / scale), qmin, qmax)

    quantized = torch.clamp(torch.round(tensor / scale) + zero_point, qmin, qmax)
    return quantized.to(torch.uint8), scale, zero_point


def asymmetric_dequantize(quantized: torch.Tensor, scale: torch.Tensor,
                          zero_point: torch.Tensor) -> torch.Tensor:
    return (quantized.float() - zero_point) * scale


# ============================================================================
# PART 4: QUANTIZATION GRANULARITY
# ============================================================================

"""
Where to compute the scale factor:

    Per-tensor:  ONE scale for the entire weight matrix
        + Simplest, fastest
        - Least accurate (outliers in one row affect all rows)

    Per-channel: ONE scale per output channel (row of weight matrix)
        + Much better accuracy
        - Slightly more complex
        Used by: most PTQ methods

    Per-group:   ONE scale per group of G consecutive values
        + Best accuracy (G=128 is common)
        - Most storage overhead (one scale per group)
        Used by: GPTQ, AWQ (G=128)

    ┌──────────────────────────────────────────────────────┐
    │  Weight matrix W: [out_features × in_features]       │
    │                                                      │
    │  Per-tensor (1 scale):                                │
    │  ┌──────────────────────┐                            │
    │  │ scale_1               │ → entire matrix            │
    │  └──────────────────────┘                            │
    │                                                      │
    │  Per-channel (out_features scales):                   │
    │  ┌──────────────────────┐                            │
    │  │ scale_1 ─────────── │ → row 1                     │
    │  │ scale_2 ─────────── │ → row 2                     │
    │  │ scale_3 ─────────── │ → row 3                     │
    │  └──────────────────────┘                            │
    │                                                      │
    │  Per-group (groups of 128):                           │
    │  ┌──────────────────────┐                            │
    │  │ s1│ s2│ s3│ s4│ ... │ → row 1 (multiple scales)   │
    │  │ s1│ s2│ s3│ s4│ ... │ → row 2                     │
    │  └──────────────────────┘                            │
    └──────────────────────────────────────────────────────┘
"""


def per_channel_quantize(weight: torch.Tensor, bits: int = 8
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize per output channel (per row)."""
    qmax = 2 ** (bits - 1) - 1
    max_abs = weight.abs().amax(dim=-1, keepdim=True)
    scales = max_abs / qmax
    scales = torch.clamp(scales, min=1e-10)
    quantized = torch.clamp(torch.round(weight / scales), -qmax - 1, qmax)
    return quantized.to(torch.int8), scales


def per_group_quantize(weight: torch.Tensor, group_size: int = 128,
                       bits: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize per group of consecutive values.
    Most accurate but most overhead.
    """
    out_features, in_features = weight.shape
    assert in_features % group_size == 0

    qmax = 2 ** (bits - 1) - 1
    weight_grouped = weight.reshape(out_features, -1, group_size)
    max_abs = weight_grouped.abs().amax(dim=-1, keepdim=True)
    scales = max_abs / qmax
    scales = torch.clamp(scales, min=1e-10)

    quantized = torch.clamp(
        torch.round(weight_grouped / scales), -qmax - 1, qmax
    )
    return quantized.reshape(out_features, in_features).to(torch.int8), scales.squeeze(-1)


# ============================================================================
# PART 5: POST-TRAINING QUANTIZATION (PTQ)
# ============================================================================

"""
PTQ: Quantize a pre-trained model WITHOUT retraining.

    Simplest approach: Round-to-Nearest (RTN)
    Just quantize each weight to the nearest integer.

    Better approach: Calibration
    Run a small dataset through the model to find the range of
    activations, then set quantization ranges accordingly.

    For weights: use min/max of weight values
    For activations: need calibration data to find typical ranges
"""


def round_to_nearest_quantize(model: nn.Module, bits: int = 8) -> dict:
    """
    Simplest PTQ: quantize all weights by rounding to nearest.

    Returns dict of {name: (quantized_weight, scale)} for each layer.
    """
    quantized_weights = {}
    for name, param in model.named_parameters():
        if param.dim() >= 2:  # Only quantize weight matrices
            q, s = symmetric_quantize(param.data, bits=bits)
            quantized_weights[name] = (q, s)
    return quantized_weights


# ============================================================================
# PART 6: GPTQ — SECOND-ORDER WEIGHT QUANTIZATION
# ============================================================================

"""
GPTQ: Use second-order (Hessian) information to quantize more carefully.

    Insight: some weights are MORE IMPORTANT than others.
    The Hessian (second derivative of loss w.r.t. weights) tells us
    which weights are most sensitive.

    Instead of independently rounding each weight:
    - Quantize one column at a time (left to right)
    - After quantizing a column, ADJUST remaining columns to
      compensate for the quantization error
    - Use the Hessian to determine optimal adjustments

    ┌──────────────────────────────────────────────────────┐
    │  GPTQ Algorithm (simplified):                        │
    │                                                      │
    │  For each column j (left to right):                   │
    │    1. Quantize column j: w_q[j] = quant(w[j])        │
    │    2. Compute error: δ = w[j] - dequant(w_q[j])      │
    │    3. Adjust remaining columns:                       │
    │       w[j+1:] -= δ × H_inv[j, j+1:] / H_inv[j,j]   │
    │                                                      │
    │  H_inv = inverse of Hessian (computed from calibration│
    │  data: H = X^T X where X is layer input activations) │
    │                                                      │
    │  The adjustment step distributes quantization error   │
    │  across remaining columns in the way that minimizes   │
    │  the overall output error (second-order optimal).     │
    └──────────────────────────────────────────────────────┘

    GPTQ with group_size=128, 4-bit:
    - Llama-70B: 140 GB → 35 GB (fits on 1 GPU!)
    - Quality: ~0.5% perplexity increase (very small)
    - Calibration: ~10 minutes on a single GPU
"""


def gptq_quantize_layer(weight: torch.Tensor, hessian: torch.Tensor,
                         bits: int = 4, group_size: int = 128
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simplified GPTQ quantization for one linear layer.

    Args:
        weight: (out_features, in_features) weight matrix
        hessian: (in_features, in_features) H = X^T X / n_samples
        bits: quantization bits
        group_size: group size for per-group quantization

    Returns:
        quantized weights and scales
    """
    out_features, in_features = weight.shape
    qmax = 2 ** (bits - 1) - 1

    # Add small diagonal for numerical stability
    H = hessian.clone()
    damp = 0.01 * torch.diag(H).mean()
    H += damp * torch.eye(in_features, device=H.device)

    # Cholesky decomposition for efficient inverse
    try:
        H_inv = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(H_inv)
    except:
        H_inv = torch.linalg.inv(H)

    W = weight.clone().float()
    Q = torch.zeros_like(W)
    scales = torch.zeros(out_features, in_features // group_size, device=weight.device)

    for j in range(in_features):
        group_idx = j // group_size

        # Get scale for current group
        if j % group_size == 0:
            group_end = min(j + group_size, in_features)
            group_max = W[:, j:group_end].abs().amax(dim=1)
            scales[:, group_idx] = group_max / qmax

        s = scales[:, group_idx].clamp(min=1e-10)

        # Quantize this column
        w_col = W[:, j]
        q_col = torch.clamp(torch.round(w_col / s), -qmax - 1, qmax)
        Q[:, j] = q_col

        # Compute quantization error
        error = w_col - q_col * s  # (out_features,)

        # Adjust remaining columns (the key GPTQ step!)
        if j < in_features - 1:
            h_inv_jj = H_inv[j, j].clamp(min=1e-10)
            adjustment = error.unsqueeze(1) * H_inv[j, j+1:].unsqueeze(0) / h_inv_jj
            W[:, j+1:] -= adjustment

    return Q.to(torch.int8), scales


# ============================================================================
# PART 7: AWQ — ACTIVATION-AWARE WEIGHT QUANTIZATION
# ============================================================================

"""
AWQ: Don't treat all weights equally — some channels carry more
important activations.

    Key insight: 1% of weight channels correspond to LARGE activations.
    Quantizing these channels coarsely causes disproportionate error.

    Solution: SCALE UP important weight channels BEFORE quantization.
    This protects them from quantization error.

    ┌──────────────────────────────────────────────────────┐
    │  AWQ Algorithm:                                      │
    │                                                      │
    │  1. Run calibration data to find activation magnitudes│
    │     per channel: s_x[j] = mean(|X[:, j]|)           │
    │                                                      │
    │  2. For channels with large activations, scale UP    │
    │     the corresponding weights:                       │
    │     W_scaled[:, j] = W[:, j] × α^s_x[j]            │
    │     (α > 1, searched for optimal value)              │
    │                                                      │
    │  3. Scale DOWN the activations to compensate:         │
    │     X_scaled[:, j] = X[:, j] / α^s_x[j]            │
    │     (mathematically equivalent: W×X = W_s × X_s)     │
    │                                                      │
    │  4. Quantize W_scaled — the important channels now   │
    │     have larger values, so quantization error is      │
    │     relatively smaller for them!                     │
    │                                                      │
    │  Before AWQ:                                         │
    │  Channel 5 (important): W=0.01, quant error=0.005   │
    │  Relative error: 50%!                                │
    │                                                      │
    │  After AWQ:                                          │
    │  Channel 5 (scaled up): W=1.0, quant error=0.005    │
    │  Relative error: 0.5%!                               │
    └──────────────────────────────────────────────────────┘
"""


def awq_scale_search(weight: torch.Tensor, activation_magnitudes: torch.Tensor,
                     bits: int = 4, n_grid: int = 20
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Search for optimal per-channel scaling factors.

    Args:
        weight: (out_features, in_features) weight matrix
        activation_magnitudes: (in_features,) mean |activation| per channel
        bits: quantization bits
        n_grid: number of alpha values to search

    Returns:
        best_scales: (in_features,) per-channel scales
        quantized_weight: scaled and quantized weight
    """
    qmax = 2 ** (bits - 1) - 1
    best_error = float('inf')
    best_scales = torch.ones_like(activation_magnitudes)

    # Normalize activation magnitudes to [0, 1]
    act_norm = activation_magnitudes / (activation_magnitudes.max() + 1e-10)

    for alpha_idx in range(n_grid):
        alpha = alpha_idx / n_grid  # search alpha in [0, 1)
        # Scale = act_magnitude^alpha
        scales = act_norm.pow(alpha).clamp(min=1e-4)

        # Scale weights up
        w_scaled = weight * scales.unsqueeze(0)

        # Quantize
        w_max = w_scaled.abs().amax(dim=0, keepdim=True)
        w_scale = w_max / qmax
        w_q = torch.clamp(torch.round(w_scaled / w_scale.clamp(min=1e-10)),
                          -qmax - 1, qmax)
        w_deq = w_q * w_scale

        # Scale back down
        w_deq = w_deq / scales.unsqueeze(0)

        # Compute reconstruction error (weighted by activation magnitudes)
        error = ((weight - w_deq) ** 2 * activation_magnitudes.unsqueeze(0)).sum()

        if error < best_error:
            best_error = error
            best_scales = scales.clone()

    # Apply best scales and quantize
    w_scaled = weight * best_scales.unsqueeze(0)
    w_max = w_scaled.abs().amax(dim=0, keepdim=True)
    w_scale = w_max / qmax
    w_q = torch.clamp(torch.round(w_scaled / w_scale.clamp(min=1e-10)),
                      -qmax - 1, qmax)

    return best_scales, w_q.to(torch.int8)


# ============================================================================
# PART 8: SMOOTHQUANT — SMOOTH THE QUANTIZATION DIFFICULTY
# ============================================================================

"""
SmoothQuant: For W8A8 (both weights AND activations in INT8).

    Problem: Activations have OUTLIERS — a few channels have very large
    values (10-100× larger than others). This makes activation quantization
    very lossy.

    ┌──────────────────────────────────────────────────────┐
    │  Activation distribution:                            │
    │                                                      │
    │  Channel 0: [0.1, 0.2, 0.15, 0.3, ...]  (normal)   │
    │  Channel 1: [0.05, 0.1, 0.08, 0.2, ...]  (normal)  │
    │  Channel 2: [50.0, 80.0, 60.0, 70.0, ...] OUTLIER! │
    │  Channel 3: [0.2, 0.1, 0.3, 0.15, ...]  (normal)   │
    │                                                      │
    │  Per-tensor INT8: scale = 80/127 = 0.63              │
    │  Normal channels: 0.1/0.63 = 0.16 → rounds to 0!    │
    │  → Massive precision loss for normal channels!        │
    │                                                      │
    │  SmoothQuant solution: MIGRATE difficulty to weights  │
    │                                                      │
    │  Y = X @ W                                           │
    │  Y = (X × diag(s)^-1) @ (diag(s) @ W)               │
    │  Y = X_smooth @ W_smooth                              │
    │                                                      │
    │  s = max(|X|, per channel)^α / max(|W|, per channel)^(1-α) │
    │  α = 0.5 (split difficulty evenly)                   │
    │                                                      │
    │  After smoothing:                                    │
    │  X_smooth: outliers reduced (÷ by large s)           │
    │  W_smooth: absorbs the difficulty (× by s)           │
    │  Both are now easier to quantize!                    │
    └──────────────────────────────────────────────────────┘
"""


def smooth_quant(weight: torch.Tensor, act_scales: torch.Tensor,
                 alpha: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply SmoothQuant transformation.

    Args:
        weight: (out_features, in_features)
        act_scales: (in_features,) max absolute activation per channel
        alpha: migration strength (0.5 = equal split)

    Returns:
        smoothed_weight, smooth_scales, activation_divisor
    """
    # Compute per-channel weight scales
    weight_scales = weight.abs().amax(dim=0)  # (in_features,)

    # Compute smoothing factor
    s = (act_scales.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-5)

    # Apply: divide activations by s, multiply weights by s
    smoothed_weight = weight * s.unsqueeze(0)  # W_smooth = W × diag(s)
    # Activations would be: X_smooth = X / s  (applied at runtime)

    return smoothed_weight, s, s  # second s is the activation divisor


# ============================================================================
# PART 9: FP8 TRAINING
# ============================================================================

"""
FP8 Training: use 8-bit floats for forward AND backward pass.

    Standard mixed-precision: FP16/BF16 compute + FP32 accumulation
    FP8 training: FP8 compute + FP16/FP32 accumulation

    Two FP8 formats:
    - E4M3: 4 exponent + 3 mantissa → better precision, less range
      Used for: weights and activations (forward pass)
    - E5M2: 5 exponent + 2 mantissa → more range, less precision
      Used for: gradients (need wider range)

    ┌──────────────────────────────────────────────────────┐
    │  FP8 Training Loop:                                  │
    │                                                      │
    │  Forward pass:                                       │
    │    Weights: FP8 E4M3 (cast from FP16 master copy)   │
    │    Activations: FP8 E4M3                             │
    │    Accumulation: FP32 (Tensor Core accumulates wider)│
    │                                                      │
    │  Backward pass:                                      │
    │    Gradients: FP8 E5M2                               │
    │    Accumulation: FP32                                 │
    │                                                      │
    │  Optimizer step: FP32 (full precision)                │
    │                                                      │
    │  Benefits:                                           │
    │    2× throughput vs FP16 (same as INT8 but flexible!) │
    │    No quantization/dequantization overhead            │
    │    Native Tensor Core support on H100+                │
    │    Minimal accuracy loss with per-tensor scaling      │
    └──────────────────────────────────────────────────────┘

    Per-tensor scaling for FP8:
    Each matmul gets a dynamic scale factor:
        scale = max(|tensor|) / max_fp8_value
        tensor_fp8 = tensor / scale   (then cast to FP8)
        result = matmul(A_fp8, B_fp8) * scale_A * scale_B

    The scale is updated each iteration using delayed scaling:
        new_scale = max(|tensor|) from PREVIOUS iteration
        (avoids the cost of computing max in the current iteration)
"""


def simulate_fp8_quantize(tensor: torch.Tensor, format: str = "E4M3"
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate FP8 quantization (actual FP8 needs H100 hardware).
    We simulate by clamping and reducing precision.
    """
    if format == "E4M3":
        max_val = 448.0       # FP8 E4M3 max
        min_val = -448.0
        # Simulate reduced precision: round to ~1.7 decimal digits
        precision_factor = 8.0  # simulate mantissa precision
    else:  # E5M2
        max_val = 57344.0     # FP8 E5M2 max
        min_val = -57344.0
        precision_factor = 4.0

    # Per-tensor scaling
    amax = tensor.abs().max()
    scale = amax / max_val if amax > 0 else torch.tensor(1.0)

    # Scale, clamp, simulate reduced precision, scale back
    scaled = tensor / scale.clamp(min=1e-10)
    clamped = torch.clamp(scaled, min_val, max_val)
    # Simulate precision loss
    quantized = torch.round(clamped * precision_factor) / precision_factor
    dequantized = quantized * scale

    return dequantized, scale


# ============================================================================
# PART 10: DEMO — COMPARING ALL TECHNIQUES
# ============================================================================

def demo():
    print("=" * 70)
    print("QUANTIZATION FROM SCRATCH DEMO")
    print("=" * 70)

    torch.manual_seed(42)

    # Create a toy weight matrix
    out_features, in_features = 256, 512
    weight = torch.randn(out_features, in_features) * 0.02
    # Add some outliers (realistic for LLM weights)
    weight[0, :10] *= 50  # outlier row
    weight[:, 42] *= 30   # outlier channel

    print(f"\nWeight matrix: {list(weight.shape)}")
    print(f"Weight range: [{weight.min():.3f}, {weight.max():.3f}]")
    print(f"Weight mean |w|: {weight.abs().mean():.4f}")

    # ── Symmetric vs Asymmetric ──
    print(f"\n{'─' * 70}")
    print("SYMMETRIC vs ASYMMETRIC QUANTIZATION (INT8)")
    print(f"{'─' * 70}")

    q_sym, s_sym = symmetric_quantize(weight, bits=8)
    deq_sym = symmetric_dequantize(q_sym, s_sym)
    error_sym = (weight - deq_sym).pow(2).mean().sqrt()

    q_asym, s_asym, zp_asym = asymmetric_quantize(weight, bits=8)
    deq_asym = asymmetric_dequantize(q_asym, s_asym, zp_asym)
    error_asym = (weight - deq_asym).pow(2).mean().sqrt()

    print(f"  Symmetric:   RMSE = {error_sym:.6f}, scale = {s_sym:.6f}")
    print(f"  Asymmetric:  RMSE = {error_asym:.6f}, scale = {s_asym:.6f}")

    # ── Granularity comparison ──
    print(f"\n{'─' * 70}")
    print("QUANTIZATION GRANULARITY (INT8)")
    print(f"{'─' * 70}")

    # Per-tensor
    q_pt, s_pt = symmetric_quantize(weight, bits=8)
    deq_pt = symmetric_dequantize(q_pt, s_pt)
    error_pt = (weight - deq_pt).pow(2).mean().sqrt()

    # Per-channel
    q_pc, s_pc = per_channel_quantize(weight, bits=8)
    deq_pc = q_pc.float() * s_pc
    error_pc = (weight - deq_pc).pow(2).mean().sqrt()

    # Per-group
    q_pg, s_pg = per_group_quantize(weight, group_size=128, bits=8)
    deq_pg = q_pg.float().reshape(out_features, -1, 128) * s_pg.unsqueeze(-1)
    deq_pg = deq_pg.reshape(out_features, in_features)
    error_pg = (weight - deq_pg).pow(2).mean().sqrt()

    print(f"  Per-tensor:    RMSE = {error_pt:.6f}")
    print(f"  Per-channel:   RMSE = {error_pc:.6f} ({(1-error_pc/error_pt)*100:.0f}% better)")
    print(f"  Per-group(128):RMSE = {error_pg:.6f} ({(1-error_pg/error_pt)*100:.0f}% better)")

    # ── Bit-width comparison ──
    print(f"\n{'─' * 70}")
    print("BIT-WIDTH COMPARISON (per-channel symmetric)")
    print(f"{'─' * 70}")

    for bits in [8, 4]:
        q, s = per_channel_quantize(weight, bits=bits)
        deq = q.float() * s
        error = (weight - deq).pow(2).mean().sqrt()
        size_bytes = weight.numel() * bits / 8 + s.numel() * 2  # + scale overhead
        orig_bytes = weight.numel() * 4
        print(f"  INT{bits}: RMSE = {error:.6f}, size = {size_bytes/1024:.1f} KB "
              f"({size_bytes/orig_bytes*100:.0f}% of FP32)")

    # ── GPTQ demo ──
    print(f"\n{'─' * 70}")
    print("GPTQ (second-order quantization, INT4)")
    print(f"{'─' * 70}")

    # Simulate calibration: compute Hessian from random activations
    n_samples = 256
    X = torch.randn(n_samples, in_features) * 0.1
    hessian = X.T @ X / n_samples

    q_gptq, s_gptq = gptq_quantize_layer(weight, hessian, bits=4, group_size=128)
    # Dequantize GPTQ
    deq_gptq = q_gptq.float()
    for j in range(in_features):
        g = j // 128
        deq_gptq[:, j] *= s_gptq[:, g]
    error_gptq = (weight - deq_gptq).pow(2).mean().sqrt()

    # Compare with naive INT4
    q_naive, s_naive = per_group_quantize(weight, group_size=128, bits=4)
    deq_naive = q_naive.float().reshape(out_features, -1, 128) * s_naive.unsqueeze(-1)
    deq_naive = deq_naive.reshape(out_features, in_features)
    error_naive = (weight - deq_naive).pow(2).mean().sqrt()

    print(f"  Naive INT4 (RTN):  RMSE = {error_naive:.6f}")
    print(f"  GPTQ INT4:         RMSE = {error_gptq:.6f} "
          f"({(1-error_gptq/error_naive)*100:.0f}% better)")

    # ── AWQ demo ──
    print(f"\n{'─' * 70}")
    print("AWQ (activation-aware quantization, INT4)")
    print(f"{'─' * 70}")

    act_magnitudes = X.abs().mean(dim=0)
    best_scales, q_awq = awq_scale_search(weight, act_magnitudes, bits=4)

    print(f"  Activation outlier channels: {(act_magnitudes > act_magnitudes.mean() * 3).sum().item()}")
    print(f"  Scale range: [{best_scales.min():.4f}, {best_scales.max():.4f}]")

    # ── SmoothQuant demo ──
    print(f"\n{'─' * 70}")
    print("SMOOTHQUANT (W8A8 quantization)")
    print(f"{'─' * 70}")

    act_scales = X.abs().amax(dim=0)
    print(f"  Activation max per channel: range [{act_scales.min():.3f}, {act_scales.max():.3f}]")
    print(f"  Outlier ratio (>5×mean): {(act_scales > act_scales.mean() * 5).sum().item()} channels")

    smoothed_w, smooth_s, _ = smooth_quant(weight, act_scales, alpha=0.5)

    # Compare quantization error before and after smoothing
    q_before, s_before = symmetric_quantize(weight, bits=8)
    error_before = (weight - symmetric_dequantize(q_before, s_before)).pow(2).mean().sqrt()

    q_after, s_after = symmetric_quantize(smoothed_w, bits=8)
    error_after = (smoothed_w - symmetric_dequantize(q_after, s_after)).pow(2).mean().sqrt()

    print(f"  Weight quant RMSE before smooth: {error_before:.6f}")
    print(f"  Weight quant RMSE after smooth:  {error_after:.6f}")
    print(f"  (Weights absorb activation difficulty, but activations become easier)")

    # ── FP8 demo ──
    print(f"\n{'─' * 70}")
    print("FP8 SIMULATION")
    print(f"{'─' * 70}")

    fp8_e4m3, scale_e4m3 = simulate_fp8_quantize(weight, "E4M3")
    fp8_e5m2, scale_e5m2 = simulate_fp8_quantize(weight, "E5M2")

    error_e4m3 = (weight - fp8_e4m3).pow(2).mean().sqrt()
    error_e5m2 = (weight - fp8_e5m2).pow(2).mean().sqrt()

    print(f"  FP8 E4M3 (weights/acts): RMSE = {error_e4m3:.6f}")
    print(f"  FP8 E5M2 (gradients):    RMSE = {error_e5m2:.6f}")

    # ── Summary table ──
    print(f"\n{'─' * 70}")
    print("SUMMARY: MEMORY SAVINGS FOR LLAMA-70B")
    print(f"{'─' * 70}")

    param_count = 70e9
    configs = [
        ("FP32", 4.0, "N/A"),
        ("FP16/BF16", 2.0, "~0%"),
        ("INT8 (PTQ)", 1.0, "~0.1%"),
        ("INT4 GPTQ (g128)", 0.5 + 0.02, "~0.5%"),
        ("INT4 AWQ (g128)", 0.5 + 0.02, "~0.3%"),
        ("FP8 E4M3", 1.0, "~0.1%"),
    ]

    print(f"\n  {'Method':25s} {'Size':>10s} {'GPUs (H100)':>12s} {'Quality Loss':>13s}")
    print(f"  {'─'*25} {'─'*10} {'─'*12} {'─'*13}")
    for name, bpp, quality in configs:
        size_gb = param_count * bpp / 1e9
        gpus = math.ceil(size_gb / 75)  # ~75 GB usable per H100
        print(f"  {name:25s} {size_gb:7.0f} GB {gpus:>8d}      {quality:>10s}")

    print(f"\n{'=' * 70}")
    print("KEY TAKEAWAYS:")
    print("  1. INT4 quantization: 4× memory reduction, ~1% quality loss")
    print("  2. Per-group quantization (g=128) much better than per-tensor")
    print("  3. GPTQ: uses Hessian info for optimal rounding (best INT4)")
    print("  4. AWQ: protects channels with large activations")
    print("  5. SmoothQuant: enables W8A8 by migrating difficulty to weights")
    print("  6. FP8: native HW support on H100+, best for training")
    print("=" * 70)


if __name__ == "__main__":
    demo()

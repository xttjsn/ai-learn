"""
Timing Comparison: Sequential vs Parallel Processing

This script compares the computational characteristics of:
1. Recurrent models (RNN, LSTM) - sequential processing
2. A simple parallel baseline - to show what parallel looks like

THE KEY INSIGHT:
================
In a recurrent model:
    h_t = f(h_{t-1}, x_t)

This creates a chain of dependencies:
    h_1 -> h_2 -> h_3 -> ... -> h_T

You CANNOT compute h_T until you have h_{T-1}, h_{T-2}, ..., h_1.

In contrast, a parallel model computes:
    y = g(x)

where g can be applied to all positions independently.

WHAT THIS MEANS FOR GPUs:
=========================
GPUs are massively parallel - they have thousands of cores.
- Recurrent: Can only use parallelism within each timestep
- Parallel: Can use parallelism across ALL positions AND within each

This is why Transformers can be 10-100x faster to train on GPUs,
despite doing "more work" (O(n^2) attention vs O(n) recurrence).

Author: Learning project
"""

import torch
import torch.nn as nn
import time
from typing import Tuple

from rnn_from_scratch import VanillaRNN
from lstm_from_scratch import LSTM


class ParallelBaseline(nn.Module):
    """
    A simple parallel baseline that processes all positions at once.

    This is NOT a good language model (it doesn't capture dependencies),
    but it demonstrates what parallel processing looks like.

    In a real Transformer:
    - Self-attention computes relationships between all positions
    - But each attention head processes all positions in parallel
    - This is fundamentally different from the sequential RNN
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Stack of feedforward layers (applied to each position independently)
        layers = []
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process all positions in parallel.

        Args:
            x: Input, shape (batch_size, seq_len, input_size)

        Returns:
            output: shape (batch_size, seq_len, hidden_size)
        """
        # This processes all (batch_size * seq_len) positions in ONE operation
        # No sequential dependency!
        return self.layers(x)


def benchmark_model(
    model: nn.Module,
    x: torch.Tensor,
    n_warmup: int = 5,
    n_iterations: int = 20,
    device: torch.device = torch.device("cpu")
) -> float:
    """Benchmark a model's forward pass."""
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            if isinstance(model, ParallelBaseline):
                _ = model(x)
            else:
                _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iterations):
            if isinstance(model, ParallelBaseline):
                _ = model(x)
            else:
                _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = (time.perf_counter() - start) / n_iterations
    return elapsed * 1000  # Convert to milliseconds


def run_comparison():
    """Compare sequential (RNN/LSTM) vs parallel processing."""
    print("=" * 70)
    print("TIMING COMPARISON: SEQUENTIAL vs PARALLEL PROCESSING")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Model parameters
    input_size = 128
    hidden_size = 256
    num_layers = 2
    batch_size = 32

    # Create models
    rnn = VanillaRNN(input_size, hidden_size, num_layers).to(device)
    lstm = LSTM(input_size, hidden_size, num_layers).to(device)
    parallel = ParallelBaseline(input_size, hidden_size, num_layers).to(device)

    # Count parameters
    rnn_params = sum(p.numel() for p in rnn.parameters())
    lstm_params = sum(p.numel() for p in lstm.parameters())
    parallel_params = sum(p.numel() for p in parallel.parameters())

    print(f"\nModel parameters:")
    print(f"  RNN:      {rnn_params:,}")
    print(f"  LSTM:     {lstm_params:,}")
    print(f"  Parallel: {parallel_params:,}")

    # Test with different sequence lengths
    seq_lengths = [32, 64, 128, 256, 512]

    print(f"\nBatch size: {batch_size}")
    print(f"Input size: {input_size}, Hidden size: {hidden_size}")
    print("-" * 70)
    print(f"{'Seq Len':<10} {'RNN (ms)':<12} {'LSTM (ms)':<12} {'Parallel (ms)':<14} {'RNN/Par':<10}")
    print("-" * 70)

    for seq_len in seq_lengths:
        x = torch.randn(batch_size, seq_len, input_size, device=device)

        rnn_time = benchmark_model(rnn, x, device=device)
        lstm_time = benchmark_model(lstm, x, device=device)
        parallel_time = benchmark_model(parallel, x, device=device)

        ratio = rnn_time / parallel_time if parallel_time > 0 else float('inf')

        print(f"{seq_len:<10} {rnn_time:<12.3f} {lstm_time:<12.3f} "
              f"{parallel_time:<14.3f} {ratio:<10.1f}x")

    print("-" * 70)

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print("""
    KEY OBSERVATIONS:

    1. SEQUENTIAL MODELS (RNN, LSTM):
       - Time increases ~linearly with sequence length
       - This is because each timestep MUST wait for the previous one
       - Cannot utilize GPU parallelism across time

    2. PARALLEL MODEL:
       - Time increases much more slowly (sublinearly) with sequence length
       - All positions computed in ONE matrix multiplication
       - Fully utilizes GPU parallelism

    3. THE RATIO GROWS:
       - As sequences get longer, the advantage of parallel processing grows
       - For seq_len=512, parallel can be 10-50x faster!

    4. LSTM vs RNN:
       - LSTM is slightly slower (more computations per step)
       - But both have the same sequential bottleneck
       - LSTM's advantage is gradient flow, not speed

    WHY THIS MATTERS FOR TRANSFORMERS:
    ==================================
    Transformers replace recurrence with self-attention:
    - Attention is O(n^2) in sequence length (more operations than RNN's O(n))
    - BUT attention is fully parallelizable
    - On GPUs, parallel O(n^2) often beats sequential O(n)!

    This is the fundamental insight that made modern LLMs possible.
    """)
    print("=" * 70)


def demonstrate_dependency_chain():
    """
    Visually demonstrate the dependency chain in RNNs.
    """
    print("\n" + "=" * 70)
    print("VISUALIZING THE SEQUENTIAL DEPENDENCY CHAIN")
    print("=" * 70)
    print("""
    RECURRENT MODEL (RNN/LSTM):
    ===========================

    Input:   x_1 -----> x_2 -----> x_3 -----> x_4 -----> x_5
              |          |          |          |          |
              v          v          v          v          v
    Hidden:  h_1 -----> h_2 -----> h_3 -----> h_4 -----> h_5
              \\________/  \\________/  \\________/  \\________/
              MUST WAIT   MUST WAIT   MUST WAIT   MUST WAIT

    To compute h_5, we need h_4.
    To compute h_4, we need h_3.
    ...
    This creates a chain of T sequential operations.


    PARALLEL MODEL (like Transformer layers):
    ==========================================

    Input:   x_1        x_2        x_3        x_4        x_5
              |          |          |          |          |
              v          v          v          v          v
    Output:  y_1        y_2        y_3        y_4        y_5
              (all computed in parallel!)

    In Transformers, attention does create dependencies, but they're
    computed via matrix multiplication, which is highly parallel.


    BACKPROPAGATION THROUGH TIME (BPTT):
    ====================================
    The sequential bottleneck also affects training:

    Forward:  h_1 --> h_2 --> h_3 --> h_4 --> h_5 --> Loss

    Backward: dL/dh_1 <-- dL/dh_2 <-- dL/dh_3 <-- dL/dh_4 <-- dL/dh_5

    Gradients must flow backward through the same chain.
    This causes:
    1. Slow training (sequential backward pass)
    2. Vanishing gradients (products of small numbers)
    3. Exploding gradients (products of large numbers)
    """)
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_dependency_chain()
    run_comparison()

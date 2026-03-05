"""
Mixture of Experts (MoE): Scaling Models Without Scaling Compute

This module explains MoE from the ground up with working implementations.
Based on:
  - "Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer"
    (Shazeer et al., 2017)
  - GShard (Lepikhin et al., 2020)
  - Switch Transformer (Fedus et al., 2021)
  - Mixtral 8x7B (Jiang et al., 2024)

================================================================================
PART 1: THE INSIGHT — SPARSE MODELS SCALE BETTER
================================================================================

The problem with dense models:

    A 70B dense model:
    - Every token activates ALL 70B parameters
    - FLOPs per token: ~140 GFLOPs (2 × param_count)
    - Memory: 140 GB (FP16)
    - Training cost: proportional to total parameters

    To make a model "smarter," we add more parameters...
    but EVERY parameter is used for EVERY token.

    Cost scales LINEARLY with parameters:
        2× parameters → 2× FLOPs → 2× memory → 2× cost

MoE insight: not every parameter needs to fire for every input.

    A 47B MoE model (Mixtral 8×7B):
    - Total parameters: ~47B
    - Parameters active per token: ~13B (2 of 8 experts)
    - FLOPs per token: ~26 GFLOPs (like a ~13B dense model!)
    - Quality: comparable to a ~34B dense model
    - Memory: 94 GB (need to store all experts)

    ┌─────────────────────────────────────────────────────────┐
    │  DENSE MODEL:                                           │
    │                                                         │
    │  Input → [FFN: 100% of params] → Output                │
    │          ████████████████████                            │
    │          All params used = expensive                     │
    │                                                         │
    │  MoE MODEL:                                             │
    │                                                         │
    │  Input → Router → [Expert 3: 12.5%] → Output            │
    │                   [Expert 7: 12.5%]                      │
    │          ██░░░░░░██░░░░░░░░                              │
    │          Only 2/8 experts used = cheap!                  │
    │                                                         │
    │  Same total params, but 4× less compute per token!       │
    └─────────────────────────────────────────────────────────┘

Where MoE fits in a transformer:

    Standard transformer block:
        Attention → FFN

    MoE transformer block:
        Attention → MoE(FFN)
        (Replace the FFN with multiple expert FFNs + a router)

    The attention layer stays the same (it's already good at routing
    information). Only the FFN is replicated into experts.

    Mixtral 8×7B architecture:
    ┌──────────────────────────────────────────────┐
    │  For each transformer block:                  │
    │                                              │
    │  x → LayerNorm → Multi-Head Attention → + ─┐ │
    │  ↑                                         │ │
    │  └─────────────────────────────────────────┘ │
    │                                              │
    │  x → LayerNorm → Router → Select top-2 ────┐ │
    │                     │                       │ │
    │              ┌──────┴──────┐                │ │
    │              ▼             ▼                │ │
    │         [Expert 3]   [Expert 7]             │ │
    │              │             │                │ │
    │              └──────┬──────┘                │ │
    │                     │ weighted sum          │ │
    │  x ←────────────────┴──────────────────── + │ │
    │  ↑                                         │ │
    │  └─────────────────────────────────────────┘ │
    └──────────────────────────────────────────────┘

================================================================================
PART 2: THE ROUTER / GATING MECHANISM
================================================================================

The router decides which experts process each token:

    router_logits = x @ W_gate    (W_gate: [d_model, num_experts])
    router_probs = softmax(router_logits)
    top_k_experts = topk(router_probs, k=2)

    For each token:
    - Compute affinity scores for all experts
    - Select top-k experts (k=1 for Switch, k=2 for Mixtral)
    - Weight expert outputs by their router probabilities

    Example with 8 experts, top-2:

    Token "The":  router_probs = [0.02, 0.05, 0.01, 0.35, 0.03, 0.04, 0.10, 0.40]
                  top-2: Expert 7 (0.40) + Expert 3 (0.35)
                  output = 0.53 × Expert7("The") + 0.47 × Expert3("The")
                  (normalized: 0.40/(0.40+0.35) ≈ 0.53)

    Token "cat":  router_probs = [0.30, 0.05, 0.25, 0.05, 0.03, 0.04, 0.10, 0.18]
                  top-2: Expert 0 (0.30) + Expert 2 (0.25)
                  output = 0.55 × Expert0("cat") + 0.45 × Expert2("cat")

    Different tokens go to different experts!
    Experts naturally SPECIALIZE (e.g., syntax vs semantics vs math).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


# ============================================================================
# PART 3: IMPLEMENTATION — MoE LAYER
# ============================================================================

class Expert(nn.Module):
    """
    A single expert = standard FFN (SwiGLU in modern models).

    Each expert has the SAME architecture but DIFFERENT weights.
    Through training, experts learn to specialize.

    For Mixtral 8×7B:
        d_model = 4096
        d_ff = 14336  (each expert)
        8 experts × 14336 × 4096 × 2 ≈ 3.7B params per layer (just experts)
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # SwiGLU-style FFN (used in Mixtral, Llama, etc.)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # gate projection
        self.w2 = nn.Linear(d_ff, d_model, bias=False)  # down projection
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # up projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: out = (SiLU(xW1) ⊙ xW3) W2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TopKRouter(nn.Module):
    """
    Router that selects top-k experts for each token.

    The router is a simple linear layer that maps from d_model to num_experts.
    We take the top-k scores as gating weights.

    ┌──────────────────────────────────────────────┐
    │  Router computation:                          │
    │                                              │
    │  x: (batch, seq, d_model)                    │
    │       │                                      │
    │       ▼                                      │
    │  logits = x @ W_gate  → (batch, seq, n_exp)  │
    │       │                                      │
    │       ▼                                      │
    │  probs = softmax(logits)                     │
    │       │                                      │
    │       ▼                                      │
    │  top_k_probs, top_k_indices = topk(probs, k) │
    │       │                                      │
    │       ▼                                      │
    │  Normalize: top_k_probs /= sum(top_k_probs)  │
    └──────────────────────────────────────────────┘
    """
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            top_k_weights: (batch * seq_len, top_k) normalized weights
            top_k_indices: (batch * seq_len, top_k) expert indices
            router_logits: (batch * seq_len, num_experts) for aux loss
        """
        batch, seq_len, d_model = x.shape
        x_flat = x.reshape(-1, d_model)  # (B*S, d_model)

        # Compute routing scores
        router_logits = self.gate(x_flat)  # (B*S, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # Normalize so weights sum to 1
        top_k_weights = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-10)

        return top_k_weights, top_k_indices, router_logits


class MoELayer(nn.Module):
    """
    Mixture of Experts layer — replaces the FFN in a transformer.

    Architecture:
        Input → Router → dispatch to top-k experts → weighted combine → output

    Implementation strategies:
    1. Loop over experts (simple, but sequential)
    2. Batched expert computation with token permutation (fast)
    3. Expert parallelism across GPUs

    We implement strategy 2 here (token permutation):

    ┌──────────────────────────────────────────────────────┐
    │  Token Permutation (batched expert execution):       │
    │                                                      │
    │  Input tokens: [t0, t1, t2, t3, t4, t5]             │
    │  Router says:                                        │
    │    t0 → Expert 0,2    t3 → Expert 1,2                │
    │    t1 → Expert 1,0    t4 → Expert 0,1                │
    │    t2 → Expert 2,1    t5 → Expert 2,0                │
    │                                                      │
    │  Group by expert:                                    │
    │    Expert 0: [t0, t1, t4, t5]  (batch of 4)         │
    │    Expert 1: [t1, t2, t3, t4]  (batch of 4)         │
    │    Expert 2: [t0, t2, t3, t5]  (batch of 4)         │
    │                                                      │
    │  Each expert processes its batch in ONE matmul!       │
    │  Then scatter results back to original positions.     │
    └──────────────────────────────────────────────────────┘
    """
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 8,
                 top_k: int = 2, capacity_factor: float = 1.25):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        self.router = TopKRouter(d_model, num_experts, top_k)
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
            aux_loss: scalar load-balancing loss
        """
        batch, seq_len, d_model = x.shape
        x_flat = x.reshape(-1, d_model)  # (N, d_model) where N = batch * seq_len
        N = x_flat.shape[0]

        # Route tokens to experts
        top_k_weights, top_k_indices, router_logits = self.router(x)
        # top_k_weights: (N, top_k), top_k_indices: (N, top_k)

        # Compute output by dispatching to experts
        output = torch.zeros_like(x_flat)

        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]  # (N,) which expert for each token
            weights = top_k_weights[:, k]         # (N,) weight for this expert

            for expert_id in range(self.num_experts):
                # Find tokens assigned to this expert
                mask = (expert_indices == expert_id)
                if not mask.any():
                    continue

                # Get tokens for this expert
                expert_input = x_flat[mask]  # (num_tokens, d_model)

                # Apply capacity factor (drop tokens if too many)
                capacity = int(N / self.num_experts * self.capacity_factor)
                if expert_input.shape[0] > capacity:
                    expert_input = expert_input[:capacity]
                    # In real implementations, dropped tokens use a residual connection

                # Expert forward pass
                expert_output = self.experts[expert_id](expert_input)

                # Weight and scatter back
                token_indices = mask.nonzero(as_tuple=True)[0]
                if len(token_indices) > capacity:
                    token_indices = token_indices[:capacity]

                output[token_indices] += weights[token_indices].unsqueeze(-1) * expert_output

        output = output.reshape(batch, seq_len, d_model)

        # Compute auxiliary load-balancing loss
        aux_loss = self._load_balance_loss(router_logits, top_k_indices, N)

        return output, aux_loss

    def _load_balance_loss(self, router_logits: torch.Tensor,
                           top_k_indices: torch.Tensor,
                           num_tokens: int) -> torch.Tensor:
        """
        Auxiliary loss to encourage balanced expert usage.

        Without this loss, the router tends to COLLAPSE — sending most
        tokens to just 1-2 "favorite" experts. The other experts get
        no gradient signal and become useless.

        The loss encourages:
            fraction of tokens to expert i ≈ 1/num_experts (uniform)

        Switch Transformer formulation:
            L_aux = num_experts × Σ_i (f_i × P_i)

            where:
            f_i = fraction of tokens routed to expert i
            P_i = mean routing probability for expert i

        This is minimized when f_i = P_i = 1/num_experts for all i.

        ┌──────────────────────────────────────────┐
        │  Without aux loss:                        │
        │  Expert 0: ████████████████ (80% tokens) │
        │  Expert 1: ██ (10%)                      │
        │  Expert 2: █ (5%)                        │
        │  Expert 3:   (2%)                        │
        │  Expert 4-7: (3% combined)               │
        │  → Most experts are WASTED                │
        │                                          │
        │  With aux loss:                           │
        │  Expert 0: ████ (14%)                    │
        │  Expert 1: ███ (13%)                     │
        │  Expert 2: ████ (14%)                    │
        │  Expert 3: ███ (12%)                     │
        │  Expert 4: ███ (13%)                     │
        │  Expert 5: ████ (14%)                    │
        │  Expert 6: ███ (11%)                     │
        │  Expert 7: ███ (9%)                      │
        │  → All experts utilized!                  │
        └──────────────────────────────────────────┘
        """
        router_probs = F.softmax(router_logits, dim=-1)  # (N, num_experts)

        # f_i: fraction of tokens assigned to expert i
        expert_counts = torch.zeros(self.num_experts, device=router_logits.device)
        for k in range(self.top_k):
            for i in range(self.num_experts):
                expert_counts[i] += (top_k_indices[:, k] == i).float().sum()
        f = expert_counts / (num_tokens * self.top_k)

        # P_i: mean routing probability for expert i
        P = router_probs.mean(dim=0)

        # Aux loss: num_experts × Σ(f_i × P_i)
        aux_loss = self.num_experts * (f * P).sum()

        return aux_loss


# ============================================================================
# PART 4: EXPERT PARALLELISM
# ============================================================================

"""
With many experts, we can distribute them across GPUs:

    ┌──────────────────────────────────────────────────────┐
    │  Expert Parallelism (EP) with 4 GPUs, 8 experts:     │
    │                                                      │
    │  GPU 0: Expert 0, Expert 1                           │
    │  GPU 1: Expert 2, Expert 3                           │
    │  GPU 2: Expert 4, Expert 5                           │
    │  GPU 3: Expert 6, Expert 7                           │
    │                                                      │
    │  For each token:                                     │
    │  1. All GPUs compute router (replicated, cheap)      │
    │  2. ALL-TO-ALL: send tokens to the GPU with their    │
    │     assigned expert                                  │
    │  3. Each GPU runs its experts on received tokens     │
    │  4. ALL-TO-ALL: send results back to source GPU      │
    │                                                      │
    │  Communication pattern:                               │
    │                                                      │
    │  GPU 0  ──t3→  GPU 1    GPU 1  ──t1→  GPU 0         │
    │  GPU 0  ──t5→  GPU 2    GPU 2  ──t2→  GPU 0         │
    │  GPU 1  ──t7→  GPU 3    GPU 3  ──t4→  GPU 1         │
    │  ...           ...      ...           ...            │
    │                                                      │
    │  This is an ALL-TO-ALL pattern (each GPU sends to    │
    │  every other GPU). Most efficient on NVSwitch.        │
    └──────────────────────────────────────────────────────┘

    Combining with Data Parallelism (DP) and Tensor Parallelism (TP):

    Mixtral 8×7B on 32 GPUs:
    - Expert Parallelism: 8 GPUs (1 expert per GPU)
    - Data Parallelism: 4 replicas
    - No tensor parallelism needed (each expert is small)

    Larger MoE models (e.g., Switch 1.6T):
    - Expert Parallelism: 64-128 GPUs
    - Data Parallelism: more replicas
    - Possibly TP within experts if experts are large
"""


# ============================================================================
# PART 5: CAPACITY FACTOR AND TOKEN DROPPING
# ============================================================================

"""
The capacity problem:

    With N tokens and E experts (top-1 routing):
    Expected tokens per expert = N/E

    But routing is NOT uniform — some experts get more tokens.
    We set a CAPACITY limit per expert:

        capacity = (N/E) × capacity_factor

    capacity_factor = 1.0: exactly average, many tokens dropped
    capacity_factor = 1.5: 50% buffer, fewer drops
    capacity_factor = 2.0: lots of buffer, no drops but wasted compute

    ┌──────────────────────────────────────────────┐
    │  Token dropping example (8 tokens, 4 experts) │
    │  Capacity factor = 1.0 → capacity = 2        │
    │                                              │
    │  Expert 0: [t0, t1]         (2/2, full)      │
    │  Expert 1: [t2, t3, t4] ← t4 DROPPED         │
    │  Expert 2: [t5]             (1/2)             │
    │  Expert 3: [t6, t7]        (2/2, full)       │
    │                                              │
    │  Token t4 gets dropped! Its output is the     │
    │  input (residual connection only).            │
    │                                              │
    │  With capacity_factor = 1.5 → capacity = 3:  │
    │  Expert 1: [t2, t3, t4]    (3/3, fits!)      │
    │  No tokens dropped.                           │
    └──────────────────────────────────────────────┘

    GShard approach: top-2 routing + capacity factor 2.0
    Switch approach: top-1 routing + capacity factor 1.5
    Mixtral approach: top-2 routing + no dropping (smaller expert count)
"""


# ============================================================================
# PART 6: EVOLUTION OF MoE — GShard → Switch → Mixtral
# ============================================================================

"""
    ┌──────────────────────────────────────────────────────────────────┐
    │  MoE EVOLUTION                                                   │
    ├──────────────┬────────────┬────────────────┬────────────────────┤
    │              │ GShard     │ Switch          │ Mixtral            │
    │              │ (2020)     │ Transformer     │ 8×7B (2024)       │
    │              │            │ (2021)          │                    │
    ├──────────────┼────────────┼────────────────┼────────────────────┤
    │ Top-k        │ 2          │ 1              │ 2                  │
    ├──────────────┼────────────┼────────────────┼────────────────────┤
    │ Experts      │ 2048       │ 128-2048       │ 8                  │
    ├──────────────┼────────────┼────────────────┼────────────────────┤
    │ Total params │ 600B       │ 1.6T           │ 47B               │
    ├──────────────┼────────────┼────────────────┼────────────────────┤
    │ Active params│ ~2B        │ ~200M          │ ~13B              │
    ├──────────────┼────────────┼────────────────┼────────────────────┤
    │ Expert size  │ Small      │ Very small     │ Large (7B-like)   │
    ├──────────────┼────────────┼────────────────┼────────────────────┤
    │ Training     │ Encoder-dec│ Encoder-decoder│ Decoder-only      │
    ├──────────────┼────────────┼────────────────┼────────────────────┤
    │ Capacity     │ Yes (2.0)  │ Yes (1.5)      │ No                │
    │ factor       │            │                │ (8 experts, 2 per │
    │              │            │                │  token → 25% each)│
    ├──────────────┼────────────┼────────────────┼────────────────────┤
    │ Key insight  │ Scale with │ Simpler is     │ Fewer, larger     │
    │              │ many tiny  │ better (top-1) │ experts work great │
    │              │ experts    │                │ for decoder-only   │
    └──────────────┴────────────┴────────────────┴────────────────────┘

    The trend: fewer, larger experts

    GShard/Switch: thousands of tiny experts
        Pro: massive parameter count, good for encoder-decoder
        Con: load balancing is hard, many experts underutilized

    Mixtral: 8 large experts
        Pro: easier to balance, each expert is capable
        Con: less total parameter scaling
        Result: much more practical for deployment

    DeepSeek-V2 (2024): combines MoE with other innovations
        64 routed experts + 2 shared experts
        Fine-grained expert segmentation
        Pushes MoE to even larger scale
"""


# ============================================================================
# PART 7: DEMO
# ============================================================================

def demo():
    """
    Demonstrate MoE routing, load balancing, and memory analysis.
    """
    print("=" * 70)
    print("MIXTURE OF EXPERTS (MoE) DEMO")
    print("=" * 70)

    torch.manual_seed(42)

    # ── Model Setup ──
    d_model = 512
    d_ff = 1024
    num_experts = 8
    top_k = 2

    moe = MoELayer(d_model, d_ff, num_experts, top_k)

    # Count parameters
    total_params = sum(p.numel() for p in moe.parameters())
    expert_params = sum(p.numel() for e in moe.experts for p in e.parameters())
    router_params = sum(p.numel() for p in moe.router.parameters())

    print(f"\nMoE Layer Configuration:")
    print(f"  d_model: {d_model}")
    print(f"  d_ff: {d_ff}")
    print(f"  Experts: {num_experts}")
    print(f"  Top-k: {top_k}")
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Expert parameters: {expert_params:,} ({expert_params/total_params*100:.1f}%)")
    print(f"  Router parameters: {router_params:,} ({router_params/total_params*100:.1f}%)")
    print(f"  Active per token: {expert_params * top_k / num_experts:,.0f} "
          f"({top_k/num_experts*100:.0f}% of expert params)")

    # ── Dense equivalent comparison ──
    dense_ffn = nn.Sequential(
        nn.Linear(d_model, d_ff, bias=False),
        nn.SiLU(),
        nn.Linear(d_ff, d_model, bias=False),
    )
    dense_params = sum(p.numel() for p in dense_ffn.parameters())

    print(f"\n  Dense FFN params: {dense_params:,}")
    print(f"  MoE total params: {total_params:,} ({total_params/dense_params:.1f}× dense)")
    print(f"  MoE active params: {expert_params * top_k / num_experts:,.0f} "
          f"({expert_params * top_k / num_experts / dense_params:.1f}× dense)")

    # ── Forward pass ──
    print(f"\n{'─' * 70}")
    print("FORWARD PASS WITH ROUTING")
    print(f"{'─' * 70}")

    batch_size = 2
    seq_len = 16
    x = torch.randn(batch_size, seq_len, d_model)

    output, aux_loss = moe(x)
    print(f"\n  Input shape:  {list(x.shape)}")
    print(f"  Output shape: {list(output.shape)}")
    print(f"  Aux loss: {aux_loss.item():.4f}")

    # ── Analyze routing distribution ──
    print(f"\n{'─' * 70}")
    print("ROUTING ANALYSIS")
    print(f"{'─' * 70}")

    with torch.no_grad():
        weights, indices, logits = moe.router(x)
        # indices: (B*S, top_k)
        N = batch_size * seq_len

        print(f"\n  Tokens: {N}, Top-k: {top_k}")
        print(f"\n  Expert usage (tokens assigned to each expert):")

        for expert_id in range(num_experts):
            count = (indices == expert_id).sum().item()
            bar = "█" * int(count / N * top_k * 40)
            pct = count / (N * top_k) * 100
            print(f"    Expert {expert_id}: {bar:40s} {count:3d} tokens ({pct:.0f}%)")

        # Show routing for first few tokens
        print(f"\n  Routing for first 8 tokens:")
        for i in range(min(8, N)):
            experts = indices[i].tolist()
            ws = weights[i].tolist()
            print(f"    Token {i}: Expert {experts[0]} ({ws[0]:.2f}) + "
                  f"Expert {experts[1]} ({ws[1]:.2f})")

    # ── Load balancing demonstration ──
    print(f"\n{'─' * 70}")
    print("LOAD BALANCING (training with aux loss)")
    print(f"{'─' * 70}")

    # Train for a few steps and watch balance improve
    optimizer = torch.optim.Adam(moe.parameters(), lr=1e-3)
    x_train = torch.randn(4, 32, d_model)

    for step in range(5):
        output, aux_loss = moe(x_train)
        main_loss = output.sum()  # dummy main loss
        total_loss = main_loss + 0.01 * aux_loss  # 0.01 is typical aux weight

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Check balance
        with torch.no_grad():
            _, indices, _ = moe.router(x_train)
            N = x_train.shape[0] * x_train.shape[1]
            counts = torch.zeros(num_experts)
            for e in range(num_experts):
                counts[e] = (indices == e).sum().item()
            cv = counts.std() / counts.mean()  # coefficient of variation
            print(f"  Step {step}: aux_loss={aux_loss.item():.4f}, "
                  f"balance CV={cv.item():.3f} "
                  f"(lower=more balanced)")

    # ── Scaling analysis ──
    print(f"\n{'─' * 70}")
    print("SCALING: MoE vs DENSE")
    print(f"{'─' * 70}")

    configs = [
        ("Dense 7B", 7e9, 7e9),
        ("Dense 13B", 13e9, 13e9),
        ("Dense 70B", 70e9, 70e9),
        ("Mixtral 8×7B", 47e9, 13e9),
        ("Mixtral 8×22B", 141e9, 39e9),
        ("Switch 1.6T (top-1)", 1600e9, 0.2e9),
    ]

    print(f"\n  {'Model':25s} {'Total Params':>14s} {'Active/Token':>14s} "
          f"{'Memory (FP16)':>14s} {'FLOPs ratio':>12s}")
    print(f"  {'─' * 25} {'─' * 14} {'─' * 14} {'─' * 14} {'─' * 12}")

    for name, total, active in configs:
        mem_gb = total * 2 / 1e9
        flops_ratio = active / active  # normalized to self
        print(f"  {name:25s} {total/1e9:11.1f}B {active/1e9:11.1f}B "
              f"{mem_gb:11.1f} GB {active/1e9:9.1f}B")

    print(f"\n{'=' * 70}")
    print("KEY TAKEAWAYS:")
    print("  1. MoE scales parameters WITHOUT scaling per-token FLOPs")
    print("  2. Router (gating) picks top-k experts per token")
    print("  3. Auxiliary loss prevents expert collapse (load balancing)")
    print("  4. Capacity factor limits max tokens per expert (prevents overload)")
    print("  5. Expert parallelism: distribute experts across GPUs")
    print("  6. Trend: fewer, larger experts (Mixtral) > many tiny experts (Switch)")
    print("=" * 70)


if __name__ == "__main__":
    demo()

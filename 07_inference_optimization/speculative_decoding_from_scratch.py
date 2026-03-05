"""
Speculative Decoding: Making LLM Inference Faster Without Losing Quality

This module explains speculative decoding from the ground up with working code.
Based on:
  - "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al., 2023)
  - "Accelerating Large Language Model Decoding with Speculative Sampling" (Chen et al., 2023)
  - Medusa (Cai et al., 2024), Eagle (Li et al., 2024)

================================================================================
PART 1: THE AUTOREGRESSIVE BOTTLENECK
================================================================================

LLM inference generates tokens one at a time:

    "The cat sat on the" → "mat" → "." → [EOS]

Each token requires a FULL forward pass through the model:

    Token 1: Load 70B weights from HBM → compute → 1 token
    Token 2: Load 70B weights from HBM → compute → 1 token
    Token 3: Load 70B weights from HBM → compute → 1 token
    ...

For a 70B model on H100:
    Weight size (FP16): 140 GB
    HBM bandwidth: 3.35 TB/s
    Time per token: 140 / 3350 ≈ 42 ms → ~24 tokens/sec

    The GPU is >95% idle! It loads 140 GB of weights to do
    a tiny amount of compute (one token's worth).

    Arithmetic intensity = FLOPs / bytes = pathetically low

The key insight: a FORWARD PASS with 1 token costs almost the same
as a forward pass with K tokens (for small K). Why?

    1 token:  Load 140 GB weights, multiply by 1 vector   = 140 GB loaded
    8 tokens: Load 140 GB weights, multiply by 8 vectors  = 140 GB loaded

    Same memory bandwidth cost, 8× more useful work!
    (The extra compute for 8 tokens is negligible compared to weight loading)

This is why batching helps throughput. But for LATENCY (single user),
you're stuck at 1 token at a time... unless you can SPECULATE.


================================================================================
PART 2: THE SPECULATIVE DECODING IDEA
================================================================================

What if a small, fast model could GUESS the next K tokens, and then
the large model could VERIFY all K tokens in one forward pass?

    ┌─────────────────────────────────────────────────────┐
    │  STANDARD AUTOREGRESSIVE DECODING                    │
    │                                                     │
    │  Step 1: Large model → token 1     (42 ms)          │
    │  Step 2: Large model → token 2     (42 ms)          │
    │  Step 3: Large model → token 3     (42 ms)          │
    │  Step 4: Large model → token 4     (42 ms)          │
    │  Step 5: Large model → token 5     (42 ms)          │
    │                                                     │
    │  Total: 5 × 42 ms = 210 ms for 5 tokens             │
    └─────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────┐
    │  SPECULATIVE DECODING                                │
    │                                                     │
    │  Step 1: Draft model → 5 guesses   (5 × 3 ms = 15ms)│
    │  Step 2: Large model verifies all 5 (42 ms)          │
    │  Result: 3 accepted + 1 corrected = 4 tokens         │
    │                                                     │
    │  Total: 15 + 42 = 57 ms for 4 tokens                 │
    │  Speedup: 210/57 ≈ 3.7× faster!                     │
    └─────────────────────────────────────────────────────┘

The magic: verification is CHEAP because the large model processes
all K draft tokens in PARALLEL (one forward pass, K tokens).

And the output distribution is IDENTICAL to the large model alone!
No quality loss — this is an EXACT speedup technique.


================================================================================
PART 3: THE MATH — ACCEPTANCE/REJECTION SAMPLING
================================================================================

Let:
    p(x) = large model's probability distribution
    q(x) = draft model's probability distribution

For each draft token x_i:

    Accept with probability:  min(1, p(x_i) / q(x_i))

    If rejected, sample from the ADJUSTED distribution:
        p'(x) = max(0, p(x) - q(x)) / Z
        where Z = Σ max(0, p(x) - q(x))   (normalization)

Why this works:
    - If draft model agrees with large model (q ≈ p), accept rate is high
    - If draft model is wrong (q(x) >> p(x)), token is rejected
    - The adjusted distribution ensures the FINAL output follows p(x) exactly
    - This is a form of rejection sampling from probability theory

    Proof sketch (that output follows p exactly):

    P(accept x and output x) = q(x) × min(1, p(x)/q(x))
                              = min(q(x), p(x))

    P(reject and resample x) = (1 - Σ min(q(x'), p(x'))) × p'(x)
                              = (Σ max(0, p(x')-q(x'))) × max(0, p(x)-q(x)) / Z
                              = max(0, p(x) - q(x))

    Total P(output x) = min(q(x), p(x)) + max(0, p(x) - q(x)) = p(x) ✓

Expected number of accepted tokens:
    E[accepted] = Σ_x min(p(x), q(x)) = 1 - TV(p, q)

    where TV(p, q) is the total variation distance.
    If draft model is good (TV ≈ 0), almost all tokens accepted.
    If draft model is bad (TV ≈ 1), almost no tokens accepted.


================================================================================
PART 4: IMPLEMENTATION
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Tuple, List, Optional


# ============================================================================
# A simple toy transformer for demonstration
# ============================================================================

class ToyTransformerBlock(nn.Module):
    """A minimal transformer block for demonstration."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with causal mask
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask, is_causal=(mask is None))
        x = x + h
        x = x + self.ff(self.ln2(x))
        return x


class ToyLM(nn.Module):
    """
    A toy language model. We'll use a LARGE version as the target model
    and a SMALL version as the draft model.

    Architecture:
        Embedding → N × TransformerBlock → LayerNorm → Linear → logits

    Parameters:
        vocab_size: vocabulary size
        d_model:    hidden dimension
        n_heads:    number of attention heads
        n_layers:   number of transformer blocks
        d_ff:       feedforward hidden dimension
    """
    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            ToyTransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) token IDs

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate_greedy(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Standard autoregressive greedy decoding (baseline)."""
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


# ============================================================================
# PART 4a: SPECULATIVE DECODING — CORE ALGORITHM
# ============================================================================

@torch.no_grad()
def speculative_decode(
    target_model: ToyLM,
    draft_model: ToyLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    K: int = 5,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Speculative decoding: use a small draft model to propose K tokens,
    then verify with the large target model in one forward pass.

    The algorithm:
    ┌─────────────────────────────────────────────────────────┐
    │  1. Draft model generates K tokens autoregressively     │
    │     (fast, because draft model is small)                │
    │                                                         │
    │  2. Target model processes ALL K tokens in one pass     │
    │     (gets probability p(x_i) for each draft token)      │
    │                                                         │
    │  3. For each draft token i = 1, ..., K:                 │
    │     - Accept with prob min(1, p(x_i) / q(x_i))         │
    │     - If rejected: resample from adjusted distribution  │
    │       and STOP (discard remaining draft tokens)         │
    │                                                         │
    │  4. If all K accepted: sample one BONUS token from      │
    │     target model's distribution at position K+1         │
    └─────────────────────────────────────────────────────────┘

    Args:
        target_model: Large model (the one we want to match)
        draft_model: Small model (fast, approximate)
        input_ids: (1, seq_len) initial token IDs
        max_new_tokens: Maximum tokens to generate
        K: Number of draft tokens per speculation round
        temperature: Sampling temperature (1.0 = standard)

    Returns:
        generated_ids: (1, seq_len + n_generated)
        stats: Dictionary with acceptance rates etc.
    """
    assert input_ids.shape[0] == 1, "Batch size must be 1 for this demo"

    generated = input_ids.clone()
    total_draft_tokens = 0
    total_accepted = 0
    n_rounds = 0
    total_generated = 0

    while total_generated < max_new_tokens:
        # ── Step 1: Draft model generates K tokens ──────────────────
        draft_ids = generated.clone()
        draft_probs_list = []  # Store q(x_i) for each draft token

        for _ in range(K):
            draft_logits = draft_model(draft_ids)
            # Get probability distribution at last position
            draft_logits_last = draft_logits[:, -1, :] / temperature
            draft_p = F.softmax(draft_logits_last, dim=-1)

            # Sample from draft distribution
            draft_token = torch.multinomial(draft_p, num_samples=1)
            draft_probs_list.append(draft_p)

            draft_ids = torch.cat([draft_ids, draft_token], dim=1)

        # draft_ids now has K extra tokens appended
        draft_tokens = draft_ids[:, generated.shape[1]:]  # (1, K)

        # ── Step 2: Target model verifies ALL K tokens in one pass ──
        # Feed the entire sequence (original + K draft tokens) to target model
        target_logits = target_model(draft_ids)

        # We need target model's probabilities at positions where
        # draft tokens were placed
        # target_logits[:, generated.shape[1]-1 : generated.shape[1]-1+K, :]
        # gives us p(x|context) for each draft token position

        n_accepted = 0

        for i in range(K):
            if total_generated >= max_new_tokens:
                break

            pos = generated.shape[1] - 1 + i  # position in the sequence

            # Target model's distribution at this position
            target_p = F.softmax(target_logits[:, pos, :] / temperature, dim=-1)

            # Draft model's distribution at this position
            draft_p = draft_probs_list[i]

            # The draft token that was sampled
            x_i = draft_tokens[:, i]  # (1,)

            # ── Step 3: Accept/reject ──
            # Accept probability = min(1, p(x_i) / q(x_i))
            p_x = target_p[0, x_i[0]]
            q_x = draft_p[0, x_i[0]]

            accept_prob = torch.min(
                torch.tensor(1.0, device=p_x.device),
                p_x / (q_x + 1e-10)
            )

            # Sample uniform random number
            r = torch.rand(1, device=accept_prob.device)

            if r < accept_prob:
                # ACCEPT this token
                generated = torch.cat([generated, x_i.unsqueeze(0)], dim=1)
                n_accepted += 1
                total_generated += 1
            else:
                # REJECT: sample from adjusted distribution
                # p'(x) = max(0, p(x) - q(x)) / Z
                adjusted = torch.clamp(target_p - draft_p, min=0)
                adjusted = adjusted / (adjusted.sum() + 1e-10)
                corrected_token = torch.multinomial(adjusted[0], num_samples=1)
                generated = torch.cat(
                    [generated, corrected_token.unsqueeze(0)], dim=1
                )
                total_generated += 1
                break  # Stop accepting further draft tokens

        else:
            # All K tokens accepted! Get a bonus token from target model
            if total_generated < max_new_tokens:
                bonus_p = F.softmax(
                    target_logits[:, generated.shape[1] - 1, :] / temperature,
                    dim=-1
                )
                bonus_token = torch.multinomial(bonus_p, num_samples=1)
                generated = torch.cat([generated, bonus_token], dim=1)
                total_generated += 1

        total_draft_tokens += K
        total_accepted += n_accepted
        n_rounds += 1

    stats = {
        "n_rounds": n_rounds,
        "total_draft_tokens": total_draft_tokens,
        "total_accepted": total_accepted,
        "acceptance_rate": total_accepted / max(total_draft_tokens, 1),
        "tokens_per_round": total_generated / max(n_rounds, 1),
    }

    return generated, stats


# ============================================================================
# PART 5: WHY VERIFICATION IS CHEAP
# ============================================================================

"""
Why can the target model verify K tokens as cheaply as generating 1?

    Standard generation (1 token):
        Input:  [t1, t2, ..., tn]           (n tokens)
        Output: logits at position n         (1 distribution)
        Cost:   Load all weights once        (140 GB for 70B model)
        Time:   ~42 ms

    Verification (K tokens):
        Input:  [t1, t2, ..., tn, d1, d2, ..., dK]   (n+K tokens)
        Output: logits at positions n through n+K      (K+1 distributions)
        Cost:   Load all weights once                  (same 140 GB!)
        Time:   ~42 ms + tiny overhead for K extra tokens

    The overhead for K extra tokens:
        Extra FLOPs: K × (2 × d_model × d_model) per layer × n_layers
        For K=5, d_model=8192, 80 layers:
            5 × 2 × 8192² × 80 ≈ 54 GFLOPs
        H100 peak: 990 TFLOPS
        Time: 54e9 / 990e12 = 0.05 ms   ← NEGLIGIBLE!

    So the verification costs ~42.05 ms vs ~42 ms for standard decoding.
    But we get K+1 tokens instead of 1!

    Expected speedup:

        Let α = acceptance rate (fraction of draft tokens accepted)
        Let c = cost ratio (draft model time / target model time)

        Standard: 1 token per target forward pass
        Speculative: E[accepted] + 1 tokens per (K × c + 1) target forward passes

        Speedup ≈ (1 - α^(K+1)) / ((1 - α) × (K × c + 1))

        For α=0.8, K=5, c=0.1:
            Speedup ≈ 2.8×

        For α=0.9, K=5, c=0.1:
            Speedup ≈ 3.5×
"""


# ============================================================================
# PART 6: VARIANTS OF SPECULATIVE DECODING
# ============================================================================

# ── Variant 1: MEDUSA — Multiple Decoding Heads ────────────────────────────

class MedusaHead(nn.Module):
    """
    Medusa: Instead of a separate draft model, add extra prediction heads
    to the target model itself.

    Standard LM:  hidden_state → lm_head → next token prediction
    Medusa:       hidden_state → lm_head → next token (position t+1)
                  hidden_state → medusa_head_1 → token at t+2
                  hidden_state → medusa_head_2 → token at t+3
                  hidden_state → medusa_head_3 → token at t+4

    ┌─────────────────────────────────┐
    │  Transformer hidden states      │
    │  (from ONE forward pass)        │
    └──────┬──────┬──────┬──────┬────┘
           │      │      │      │
           ▼      ▼      ▼      ▼
        ┌─────┐┌─────┐┌─────┐┌─────┐
        │Head ││Head ││Head ││Head │
        │ t+1 ││ t+2 ││ t+3 ││ t+4 │
        └─────┘└─────┘└─────┘└─────┘

    Advantages over separate draft model:
    - No separate model to load/store
    - Heads share the target model's representations
    - Only need to train small head networks
    - Can be added to any existing model

    Disadvantages:
    - Heads are less accurate (predicting further ahead is harder)
    - Need tree-based verification (multiple candidates per position)
    """
    def __init__(self, d_model: int, vocab_size: int, n_heads: int = 3):
        super().__init__()
        # Each Medusa head predicts a future token
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, vocab_size),
            )
            for _ in range(n_heads)
        ])

    def forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            hidden_states: (batch, seq_len, d_model) from the last transformer layer

        Returns:
            List of logits, one per head: [(batch, seq_len, vocab_size), ...]
            heads[0] predicts token at t+2, heads[1] at t+3, etc.
            (The main lm_head already predicts t+1)
        """
        return [head(hidden_states) for head in self.heads]


# ── Variant 2: EAGLE — Feature-Level Draft ─────────────────────────────────

"""
EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency):

Instead of drafting at the TOKEN level, EAGLE drafts at the FEATURE level.

    Standard speculative decoding:
        Draft model: tokens → embeddings → layers → logits → tokens
        (Full model inference, just smaller)

    EAGLE:
        Takes the target model's hidden states from the PREVIOUS step
        and uses a lightweight network to predict the NEXT hidden states.
        Then uses the target model's lm_head to convert to tokens.

    ┌──────────────────────────────────────────────────────┐
    │  Target model forward pass at step t:                │
    │  [tokens] → [hidden_states_t] → [logits_t]          │
    │                    │                                 │
    │                    ▼                                 │
    │  EAGLE predictor:                                    │
    │  [hidden_states_t] → [predicted_hidden_states_t+1]   │
    │                    → [predicted_hidden_states_t+2]   │
    │                    → ...                             │
    │                    │                                 │
    │                    ▼                                 │
    │  Target model's lm_head (shared!):                   │
    │  [predicted_hidden] → [draft_logits] → [draft_tokens]│
    └──────────────────────────────────────────────────────┘

    Advantages:
    - Uses target model's own representation space
    - Very high acceptance rate (>0.85 typical)
    - Lightweight predictor (just 1-2 transformer layers)
    - Shares the lm_head (no extra vocabulary projection)
"""


# ── Variant 3: Self-Speculative Decoding ───────────────────────────────────

"""
Self-Speculative Decoding: No separate draft model at all!

    Idea: Use EARLY EXIT from the target model as the draft.

    Full model: 80 layers
    Draft:      Use only layers 1-20 (early exit) → fast but less accurate
    Verify:     Use all 80 layers

    ┌─────────────────────────────┐
    │  Layer 1                    │
    │  Layer 2                    │  ← Draft uses
    │  ...                        │     only these
    │  Layer 20 → EARLY EXIT ─────┼──→ Draft token
    │  Layer 21                   │
    │  ...                        │  ← Verification
    │  Layer 80 → FULL EXIT ──────┼──→ Verified token
    └─────────────────────────────┘

    Advantages:
    - No separate model needed!
    - No extra memory for draft model
    - Draft naturally approximates the target
    - Can adaptively choose exit layer

    Disadvantages:
    - Need to train/add early exit heads
    - Can't parallelize draft and verify (same model)
    - Lower acceptance rate than well-matched draft model
"""


# ============================================================================
# PART 7: THROUGHPUT VS LATENCY TRADEOFFS
# ============================================================================

"""
When to use speculative decoding:

    ┌───────────────────────────────────────────────────────────┐
    │  SCENARIO                          USE SPECULATIVE?       │
    ├───────────────────────────────────────────────────────────┤
    │  Single user, latency matters      YES ✓                 │
    │  (chatbot, code completion)        Big speedup per user   │
    ├───────────────────────────────────────────────────────────┤
    │  High throughput, many users       MAYBE                  │
    │  (API serving, batch=large)        Batching already helps │
    │                                    Less benefit from spec │
    ├───────────────────────────────────────────────────────────┤
    │  Very small model                  NO ✗                  │
    │  (model already fast)              Overhead not worth it  │
    ├───────────────────────────────────────────────────────────┤
    │  Poor draft model match            NO ✗                  │
    │  (acceptance rate < 0.5)           Wasted draft compute   │
    ├───────────────────────────────────────────────────────────┤
    │  Long, creative generation         YES ✓                 │
    │  (stories, articles)               Many tokens to save    │
    ├───────────────────────────────────────────────────────────┤
    │  Short responses                   MAYBE                  │
    │  (classification, yes/no)          Not many tokens anyway │
    └───────────────────────────────────────────────────────────┘

    The fundamental tradeoff:

    Speculative decoding trades COMPUTE for LATENCY.
    - More total FLOPs (draft + verify > just verify)
    - But fewer SEQUENTIAL target model calls
    - So wall-clock time decreases

    In a high-throughput setting (large batch), the target model
    is already compute-bound (not memory-bound), so speculation
    doesn't help much — you're already utilizing the GPU well.

    In a low-batch setting, the target model is memory-bound,
    so speculation fills the compute gap with useful work.
"""


# ============================================================================
# PART 8: DEMO — COMPARING STANDARD VS SPECULATIVE DECODING
# ============================================================================

def demo():
    """
    Demonstrate speculative decoding with toy models.

    We create:
    - A "large" model (6 layers, d=256)  — the target
    - A "small" model (2 layers, d=128)  — the draft

    Then compare standard decoding vs speculative decoding.
    """
    print("=" * 70)
    print("SPECULATIVE DECODING DEMO")
    print("=" * 70)

    # ── Setup ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 1000
    torch.manual_seed(42)

    # Target model: larger
    target = ToyLM(
        vocab_size=vocab_size, d_model=256, n_heads=8,
        n_layers=6, d_ff=512
    ).to(device).eval()

    # Draft model: smaller
    draft = ToyLM(
        vocab_size=vocab_size, d_model=128, n_heads=4,
        n_layers=2, d_ff=256
    ).to(device).eval()

    target_params = sum(p.numel() for p in target.parameters())
    draft_params = sum(p.numel() for p in draft.parameters())
    print(f"\nTarget model: {target_params:,} parameters")
    print(f"Draft model:  {draft_params:,} parameters")
    print(f"Size ratio:   {target_params / draft_params:.1f}×")

    # Input prompt
    prompt = torch.randint(0, vocab_size, (1, 20), device=device)
    max_new = 50

    # ── Standard decoding ──
    print(f"\n{'─' * 70}")
    print("Standard autoregressive decoding:")
    start = time.time()
    standard_output = target.generate_greedy(prompt.clone(), max_new)
    standard_time = time.time() - start
    print(f"  Generated {max_new} tokens in {standard_time:.3f}s")
    print(f"  Throughput: {max_new / standard_time:.1f} tokens/sec")

    # ── Speculative decoding ──
    print(f"\n{'─' * 70}")
    print("Speculative decoding (K=5):")
    start = time.time()
    spec_output, stats = speculative_decode(
        target, draft, prompt.clone(), max_new, K=5, temperature=1.0
    )
    spec_time = time.time() - start
    print(f"  Generated {max_new} tokens in {spec_time:.3f}s")
    print(f"  Throughput: {max_new / spec_time:.1f} tokens/sec")
    print(f"  Acceptance rate: {stats['acceptance_rate']:.2%}")
    print(f"  Tokens per round: {stats['tokens_per_round']:.1f}")
    print(f"  Total rounds: {stats['n_rounds']}")
    print(f"  Speedup: {standard_time / spec_time:.2f}×")

    # ── Different K values ──
    print(f"\n{'─' * 70}")
    print("Effect of K (draft length):")
    for k in [1, 3, 5, 8, 12]:
        _, stats = speculative_decode(
            target, draft, prompt.clone(), max_new, K=k
        )
        print(f"  K={k:2d}: accept_rate={stats['acceptance_rate']:.2%}, "
              f"tokens/round={stats['tokens_per_round']:.1f}, "
              f"rounds={stats['n_rounds']}")

    # ── Medusa heads demo ──
    print(f"\n{'─' * 70}")
    print("Medusa heads (concept demo):")
    medusa = MedusaHead(d_model=256, vocab_size=vocab_size, n_heads=3)
    medusa = medusa.to(device)
    # Get hidden states from target model
    with torch.no_grad():
        x = target.embedding(prompt)
        for block in target.blocks:
            x = block(x)
        hidden = target.ln_f(x)
        # Medusa predictions
        medusa_logits = medusa(hidden)
        for i, logits in enumerate(medusa_logits):
            top_token = logits[:, -1, :].argmax(dim=-1)
            print(f"  Head {i+1} (predicts t+{i+2}): top token = {top_token.item()}")

    print(f"\n{'=' * 70}")
    print("KEY TAKEAWAYS:")
    print("  1. Speculative decoding gives 2-4× speedup for latency-bound inference")
    print("  2. Output distribution is IDENTICAL to standard decoding")
    print("  3. Works best when draft model closely matches target model")
    print("  4. Acceptance rate is the key metric — higher = more speedup")
    print("  5. K should be tuned: too high wastes draft compute, too low limits gains")
    print("=" * 70)


if __name__ == "__main__":
    demo()

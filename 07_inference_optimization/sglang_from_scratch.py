"""
SGLang: Efficient Execution of Complex LLM Programs

This module explains SGLang from the ground up with working implementations.
Based on:
  - "Efficiently Programming Large Language Models using SGLang" (Zheng et al., 2024)
  - RadixAttention for KV cache sharing
  - Constrained decoding with finite state machines

================================================================================
PART 1: THE PROBLEM — LLM PROGRAMS ARE INEFFICIENT
================================================================================

Modern LLM applications aren't just "prompt in, text out." They involve:

    Example: Multi-turn reasoning with tool use

        Turn 1: "What's the weather in Tokyo?"
        → LLM generates: "I'll check the weather. [TOOL_CALL: weather(Tokyo)]"
        → Tool returns: "72°F, sunny"
        Turn 2: [previous context + tool result]
        → LLM generates: "The weather in Tokyo is 72°F and sunny."
        Turn 3: "What about Osaka?"
        → LLM generates: "I'll check Osaka too. [TOOL_CALL: weather(Osaka)]"
        ...

    Example: Tree-of-thought / branching

        Prompt: "Solve this math problem"
        → Branch 1: "Let's try algebra..."
        → Branch 2: "Let's try geometry..."
        → Branch 3: "Let's try calculus..."
        → Pick the best branch, continue

    Example: Structured output (JSON)

        Prompt: "Extract entities from this text"
        → Must output: {"name": "...", "age": ..., "city": "..."}
        → Every token must be valid JSON!

The problem with naive execution:

    ┌─────────────────────────────────────────────────────────┐
    │  Naive approach: each call is independent               │
    │                                                         │
    │  Call 1: "What's the weather in Tokyo?"                 │
    │          → Compute KV cache from scratch                │
    │          → Generate response                            │
    │                                                         │
    │  Call 2: "What's the weather in Tokyo?" + response +    │
    │          tool_result + "What about Osaka?"              │
    │          → Recompute ENTIRE KV cache from scratch!      │
    │          → Even though 90% of the prefix is the same!   │
    │                                                         │
    │  Call 3: Same prefix again + more context               │
    │          → Recompute again!                             │
    │                                                         │
    │  Wasted compute: O(N² × num_turns) instead of O(N²)     │
    └─────────────────────────────────────────────────────────┘

And for branching:

    ┌─────────────────────────────────────────────────────────┐
    │  Three branches from the same prompt:                   │
    │                                                         │
    │  Branch 1: [PROMPT] + "algebra approach..."             │
    │  Branch 2: [PROMPT] + "geometry approach..."            │
    │  Branch 3: [PROMPT] + "calculus approach..."            │
    │                                                         │
    │  Naive: compute KV cache for [PROMPT] three times!      │
    │  Smart: compute once, share the prefix KV cache         │
    └─────────────────────────────────────────────────────────┘


================================================================================
PART 2: RADIXATTENTION — PREFIX TREE KV CACHE SHARING
================================================================================

The core innovation in SGLang's runtime: organize the KV cache as a
radix tree (prefix tree) so that shared prefixes are computed only ONCE.

What's a radix tree?

    A tree where each node represents a CHUNK of tokens, and
    paths from root to leaf represent complete sequences.

    Example: Three requests share a system prompt

    Request 1: [SYS_PROMPT] + "What is 2+2?"
    Request 2: [SYS_PROMPT] + "Tell me a joke"
    Request 3: [SYS_PROMPT] + "What is 2+3?"

    Radix tree:
                    [ROOT]
                      │
                [SYS_PROMPT]      ← KV cache computed ONCE
                 /    |    \\
        "What is"  "Tell me"  (shared prefix stops here)
           / \\       |
       "2+2?" "2+3?" "a joke"

    KV cache sharing:
    - [SYS_PROMPT] KV cache: stored once, shared by all 3 requests
    - "What is" KV cache: shared by requests 1 and 3
    - Only the unique suffixes need fresh computation

How RadixAttention differs from PagedAttention (vLLM):

    ┌─────────────────────────────────────────────────────────┐
    │  PagedAttention (vLLM — covered in module 03):          │
    │                                                         │
    │  - Manages KV cache as PAGES (like OS virtual memory)   │
    │  - Eliminates memory fragmentation                      │
    │  - Each request gets its own logical KV cache           │
    │  - Can share pages with copy-on-write (but doesn't      │
    │    actively build a prefix tree)                        │
    │  - Focus: memory EFFICIENCY (fit more requests)         │
    │                                                         │
    │  RadixAttention (SGLang):                               │
    │                                                         │
    │  - Manages KV cache as a RADIX TREE of token sequences  │
    │  - Actively matches request prefixes against tree       │
    │  - Automatically reuses KV cache for shared prefixes    │
    │  - Focus: COMPUTE efficiency (avoid redundant prefill)  │
    │  - Also uses paging internally for memory management    │
    │                                                         │
    │  Think of it as: PagedAttention = better memory layout  │
    │                  RadixAttention = better cache REUSE    │
    └─────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import re
import json
from collections import defaultdict


# ============================================================================
# PART 2a: RADIX TREE IMPLEMENTATION
# ============================================================================

class RadixTreeNode:
    """
    A node in the radix tree for KV cache management.

    Each node stores:
    - A chunk of token IDs (the edge label from parent to this node)
    - The corresponding KV cache tensors
    - References to children (branching points)
    - Reference count (how many active requests use this node)
    """
    def __init__(self):
        self.children: Dict[int, 'RadixTreeNode'] = {}  # first_token -> child
        self.token_ids: List[int] = []        # token IDs stored at this node
        self.kv_cache: Optional[torch.Tensor] = None  # KV cache for these tokens
        self.ref_count: int = 0               # number of active users
        self.parent: Optional['RadixTreeNode'] = None

    def __repr__(self):
        return f"RadixNode(tokens={len(self.token_ids)}, children={len(self.children)}, refs={self.ref_count})"


class RadixTree:
    """
    Radix tree for KV cache management.

    Operations:
    - insert(token_ids, kv_cache): Add a sequence and its KV cache
    - match_prefix(token_ids): Find the longest cached prefix
    - evict(): Remove least-recently-used entries when memory is full

    This is a simplified version of what SGLang actually implements.
    The real implementation handles concurrent access, batched operations,
    and integration with the paged memory manager.

    ┌─────────────────────────────────────────────────────┐
    │  Example tree after processing several requests:     │
    │                                                     │
    │  Root                                               │
    │   ├── "You are a helpful assistant" (system prompt) │
    │   │    ├── "What is the capital" (shared prefix)    │
    │   │    │    ├── " of France?" → KV cache for full   │
    │   │    │    └── " of Japan?" → KV cache for full    │
    │   │    └── "Tell me about" (different branch)       │
    │   │         └── " quantum physics" → KV cache       │
    │   └── "Translate to French:" (different sys prompt) │
    │        └── " Hello world" → KV cache                │
    └─────────────────────────────────────────────────────┘
    """
    def __init__(self, max_cache_tokens: int = 100000):
        self.root = RadixTreeNode()
        self.max_cache_tokens = max_cache_tokens
        self.total_cached_tokens = 0

    def match_prefix(self, token_ids: List[int]) -> Tuple[int, Optional[torch.Tensor]]:
        """
        Find the longest prefix of token_ids that exists in the tree.

        Returns:
            (matched_length, kv_cache_for_matched_prefix)

        This is the KEY operation: when a new request arrives, we find
        how much of its prefix is already cached, and only compute
        the remaining suffix.

        Example:
            Tree has: "You are a helpful assistant. What is"
            New request: "You are a helpful assistant. What is the capital?"

            match_prefix returns: (matched=8 tokens, kv_cache for those 8)
            We only need to compute KV for "the capital?" (3 tokens)
            Instead of all 11 tokens!
        """
        node = self.root
        matched = 0
        last_kv = None
        pos = 0

        while pos < len(token_ids):
            token = token_ids[pos]
            if token not in node.children:
                break

            child = node.children[token]
            # Check how much of this child's token chunk matches
            chunk_len = len(child.token_ids)
            remaining = token_ids[pos:pos + chunk_len]

            if remaining == child.token_ids[:len(remaining)]:
                if len(remaining) >= chunk_len:
                    # Full match of this node's chunk
                    matched += chunk_len
                    pos += chunk_len
                    if child.kv_cache is not None:
                        last_kv = child.kv_cache
                    node = child
                else:
                    # Partial match — matched up to len(remaining)
                    matched += len(remaining)
                    break
            else:
                break

        return matched, last_kv

    def insert(self, token_ids: List[int], kv_cache: Optional[torch.Tensor] = None):
        """
        Insert a token sequence into the radix tree.

        If the sequence shares a prefix with an existing sequence,
        the shared prefix is NOT duplicated — it branches at the
        point of divergence.
        """
        node = self.root
        pos = 0

        while pos < len(token_ids):
            token = token_ids[pos]

            if token not in node.children:
                # Create new child with remaining tokens
                child = RadixTreeNode()
                child.token_ids = token_ids[pos:]
                child.kv_cache = kv_cache
                child.parent = node
                child.ref_count = 1
                node.children[token] = child
                self.total_cached_tokens += len(child.token_ids)
                return

            child = node.children[token]
            chunk = child.token_ids
            remaining = token_ids[pos:]

            # Find where remaining and chunk diverge
            match_len = 0
            for i in range(min(len(chunk), len(remaining))):
                if chunk[i] == remaining[i]:
                    match_len += 1
                else:
                    break

            if match_len < len(chunk):
                # Need to SPLIT this node
                # Before: parent → child("abcde")
                # After:  parent → split("abc") → child("de")
                #                               → new("fg")
                split = RadixTreeNode()
                split.token_ids = chunk[:match_len]
                split.parent = node
                split.ref_count = child.ref_count

                child.token_ids = chunk[match_len:]
                child.parent = split
                split.children[child.token_ids[0]] = child

                node.children[token] = split

                if match_len < len(remaining):
                    new_child = RadixTreeNode()
                    new_child.token_ids = remaining[match_len:]
                    new_child.kv_cache = kv_cache
                    new_child.parent = split
                    new_child.ref_count = 1
                    split.children[new_child.token_ids[0]] = new_child
                    self.total_cached_tokens += len(new_child.token_ids)
                return

            # Full match of this chunk, continue to next node
            pos += len(chunk)
            node = child

    def cache_hit_stats(self, token_ids: List[int]) -> dict:
        """Get statistics about cache hit rate for a request."""
        matched, _ = self.match_prefix(token_ids)
        total = len(token_ids)
        return {
            "total_tokens": total,
            "cached_tokens": matched,
            "new_tokens": total - matched,
            "hit_rate": matched / max(total, 1),
        }


# ============================================================================
# PART 3: STRUCTURED GENERATION WITH FINITE STATE MACHINES
# ============================================================================

"""
The Problem: You want the LLM to output valid JSON, SQL, regex, etc.
But the LLM might generate invalid tokens at any point.

    Prompt: "Output a JSON object with name and age"
    Bad output: {"name": "Alice", "age": twenty}   ← not valid JSON!

Solution: CONSTRAINED DECODING

    At each step, determine which tokens are VALID given the current
    output so far, and MASK all invalid tokens before sampling.

    Step 1: Output so far: "{"
            Valid next: '"' (must be a key string)
            Invalid: everything else
            Mask: set logits of invalid tokens to -infinity

    Step 2: Output so far: '{"'
            Valid next: any character for the key name
            Invalid: numbers as first char, etc.

    This is formalized as a FINITE STATE MACHINE (FSM).

    ┌─────────────────────────────────────────────────────┐
    │  JSON FSM (simplified):                              │
    │                                                     │
    │  START ──{──→ OBJECT_START                          │
    │                    │                                │
    │                    "──→ KEY_STRING ──"──→ COLON_WAIT│
    │                                            │        │
    │                                            :──→ VALUE│
    │                                            │        │
    │                                    ┌───────┴───┐    │
    │                                    │  STRING   │    │
    │                                    │  NUMBER   │    │
    │                                    │  BOOL     │    │
    │                                    │  NULL     │    │
    │                                    │  OBJECT   │    │
    │                                    │  ARRAY    │    │
    │                                    └───────────┘    │
    │                                         │           │
    │                                    ,──→ NEXT_KEY    │
    │                                    }──→ END         │
    └─────────────────────────────────────────────────────┘
"""


class SimpleJSONFSM:
    """
    A simplified finite state machine for JSON generation.

    States track where we are in the JSON structure.
    At each state, we know which token categories are valid.

    In the real SGLang, this is done with the `outlines` library
    which compiles a regex/grammar into an FSM and precomputes
    the valid token mask for each state.
    """

    # States
    START = "start"
    OBJECT_OPEN = "object_open"
    KEY_START = "key_start"
    IN_KEY = "in_key"
    KEY_END = "key_end"
    COLON = "colon"
    VALUE_START = "value_start"
    IN_STRING_VALUE = "in_string_value"
    IN_NUMBER = "in_number"
    AFTER_VALUE = "after_value"
    DONE = "done"

    def __init__(self, schema: Optional[dict] = None):
        """
        Args:
            schema: Optional JSON schema to constrain the structure.
                    For now, we just do basic JSON validity.
        """
        self.state = self.START
        self.depth = 0
        self.schema = schema

    def get_valid_tokens(self, vocab: List[str]) -> List[int]:
        """
        Given the current state, return indices of valid next tokens.

        This is called at each decoding step to create a mask:
            logits[invalid_tokens] = -infinity
        """
        valid = []
        for i, token in enumerate(vocab):
            if self._is_valid_next(token):
                valid.append(i)
        return valid

    def _is_valid_next(self, token: str) -> bool:
        """Check if a token is valid in the current state."""
        if self.state == self.START:
            return token.strip().startswith("{")
        elif self.state == self.OBJECT_OPEN:
            return token.strip().startswith('"') or token.strip().startswith("}")
        elif self.state == self.KEY_START:
            return True  # any character in a string
        elif self.state == self.COLON:
            return ":" in token
        elif self.state == self.VALUE_START:
            t = token.strip()
            return (t.startswith('"') or t[0:1].isdigit() or
                    t.startswith("t") or t.startswith("f") or
                    t.startswith("n") or t.startswith("{") or
                    t.startswith("["))
        elif self.state == self.AFTER_VALUE:
            t = token.strip()
            return t.startswith(",") or t.startswith("}")
        return True

    def advance(self, token: str):
        """Advance the FSM state based on the generated token."""
        t = token.strip()
        if self.state == self.START and "{" in t:
            self.state = self.OBJECT_OPEN
            self.depth += 1
        elif self.state == self.OBJECT_OPEN and t.startswith('"'):
            self.state = self.IN_KEY
        elif self.state == self.OBJECT_OPEN and t.startswith("}"):
            self.depth -= 1
            self.state = self.DONE if self.depth == 0 else self.AFTER_VALUE
        elif self.state == self.IN_KEY and t.endswith('"'):
            self.state = self.COLON
        elif self.state == self.COLON:
            self.state = self.VALUE_START
        elif self.state == self.VALUE_START:
            self.state = self.AFTER_VALUE
        elif self.state == self.AFTER_VALUE and "," in t:
            self.state = self.OBJECT_OPEN
        elif self.state == self.AFTER_VALUE and "}" in t:
            self.depth -= 1
            self.state = self.DONE if self.depth == 0 else self.AFTER_VALUE


# ============================================================================
# PART 4: JUMP-FORWARD DECODING
# ============================================================================

"""
Jump-Forward Decoding: Skip tokens that are DETERMINISTIC.

When generating constrained output, sometimes the next several tokens
are completely determined by the constraint:

    Schema requires: {"name": "
    After generating: {"na
    The FSM knows: next MUST be 'me": "'
    → Jump forward 5 tokens without running the LLM!

    ┌──────────────────────────────────────────────────────┐
    │  Standard constrained decoding:                       │
    │  Step 1: LLM → "{" (constrained)                     │
    │  Step 2: LLM → '"' (constrained, only valid option)  │
    │  Step 3: LLM → "n" (constrained by schema: "name")   │
    │  Step 4: LLM → "a" (determined)                      │
    │  Step 5: LLM → "m" (determined)                      │
    │  Step 6: LLM → "e" (determined)                      │
    │  ...                                                  │
    │  10 LLM calls for deterministic tokens!               │
    │                                                      │
    │  Jump-forward decoding:                               │
    │  Step 1: LLM → "{" → FSM sees next 9 tokens forced   │
    │  JUMP:   Append '{"name": "' without LLM calls!      │
    │  Step 2: LLM → actual value (free choice)             │
    │                                                      │
    │  Only 2 LLM calls instead of 10!                      │
    └──────────────────────────────────────────────────────┘

This is especially powerful for structured schemas where field names,
delimiters, and formatting are all predetermined.
"""


def jump_forward_decode(
    fsm: SimpleJSONFSM,
    schema: dict,
    current_output: str,
) -> str:
    """
    Given the current FSM state and schema, determine if there's a
    deterministic sequence of tokens that can be jumped over.

    Returns the string to jump forward (empty if no jump possible).

    This is a simplified version — real implementations compile the
    regex/grammar into a DFA and find single-successor state chains.
    """
    # Example: if we just opened an object and schema says first key is "name"
    if fsm.state == SimpleJSONFSM.OBJECT_OPEN and "properties" in schema:
        keys = list(schema["properties"].keys())
        if keys:
            # The next key is determined
            return f'"{keys[0]}": '
    return ""


# ============================================================================
# PART 5: SGLANG FRONTEND — THE PYTHON DSL
# ============================================================================

"""
SGLang provides a Python DSL (Domain-Specific Language) for writing
LLM programs. Why a DSL instead of raw API calls?

    ┌──────────────────────────────────────────────────────┐
    │  WITHOUT SGLang (using OpenAI API):                   │
    │                                                      │
    │  # Multi-turn with branching                          │
    │  resp1 = openai.chat(messages=[...])                  │
    │  # Branch 1                                           │
    │  resp2a = openai.chat(messages=[... + resp1 + ...])   │
    │  # Branch 2                                           │
    │  resp2b = openai.chat(messages=[... + resp1 + ...])   │
    │  # Select best                                        │
    │  best = select(resp2a, resp2b)                        │
    │  resp3 = openai.chat(messages=[... + best + ...])     │
    │                                                      │
    │  Problems:                                            │
    │  1. Each call recomputes the shared prefix             │
    │  2. No KV cache sharing between branches               │
    │  3. No constrained decoding support                    │
    │  4. No automatic batching of independent branches      │
    └──────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────┐
    │  WITH SGLang:                                         │
    │                                                      │
    │  @sgl.function                                        │
    │  def my_program(s):                                   │
    │      s += sgl.system("You are helpful")               │
    │      s += sgl.user("Solve: " + problem)               │
    │      # Fork into branches (KV cache shared!)          │
    │      forks = s.fork(3)                                │
    │      for f in forks:                                  │
    │          f += sgl.assistant(sgl.gen("solution"))      │
    │      # Select best                                    │
    │      s += sgl.select(forks, key="solution")           │
    │      # Constrained JSON output                        │
    │      s += sgl.assistant(                              │
    │          sgl.gen("result", regex=r'\{.*\}')           │
    │      )                                                │
    │                                                      │
    │  Benefits:                                            │
    │  1. RadixAttention shares KV cache for forks           │
    │  2. Independent branches batched automatically         │
    │  3. Constrained decoding built in                      │
    │  4. Jump-forward optimization automatic                │
    └──────────────────────────────────────────────────────┘
"""

# Simplified SGLang-like DSL for demonstration

@dataclass
class SGLState:
    """Represents the state of an SGLang program execution."""
    messages: List[dict] = field(default_factory=list)
    generated: Dict[str, str] = field(default_factory=dict)
    token_ids: List[int] = field(default_factory=list)
    fork_parent: Optional['SGLState'] = None

    def system(self, content: str) -> 'SGLState':
        self.messages.append({"role": "system", "content": content})
        return self

    def user(self, content: str) -> 'SGLState':
        self.messages.append({"role": "user", "content": content})
        return self

    def assistant_gen(self, name: str, constraint: Optional[str] = None) -> 'SGLState':
        """Generate text with optional constraint (regex/json schema)."""
        # In real SGLang, this triggers the LLM with RadixAttention
        # and optional FSM-based constrained decoding
        self.generated[name] = f"<generated:{name}>"
        self.messages.append({
            "role": "assistant",
            "content": self.generated[name],
            "constraint": constraint,
        })
        return self

    def fork(self, n: int) -> List['SGLState']:
        """
        Fork into N branches, sharing the current KV cache.

        In the runtime:
        1. Current prefix KV cache is in the radix tree
        2. Each fork gets a reference to it (ref_count += 1)
        3. Each fork only computes KV for its NEW tokens
        4. Forks can be batched together for efficiency
        """
        forks = []
        for _ in range(n):
            child = SGLState(
                messages=list(self.messages),  # copy messages
                generated=dict(self.generated),
                token_ids=list(self.token_ids),
                fork_parent=self,
            )
            forks.append(child)
        return forks


# ============================================================================
# PART 6: COMPARISON WITH vLLM AND TensorRT-LLM
# ============================================================================

"""
    ┌──────────────────────────────────────────────────────────────────┐
    │  FEATURE COMPARISON                                              │
    ├────────────────────┬──────────┬──────────┬──────────────────────┤
    │                    │  vLLM    │  SGLang  │  TensorRT-LLM        │
    ├────────────────────┼──────────┼──────────┼──────────────────────┤
    │  KV Cache Mgmt     │ Paged    │ Radix    │ Paged + Fused        │
    │                    │ Attention│ Attention│                      │
    ├────────────────────┼──────────┼──────────┼──────────────────────┤
    │  Prefix Sharing    │ Basic    │ Advanced │ Limited              │
    │                    │ (CoW)    │ (tree)   │                      │
    ├────────────────────┼──────────┼──────────┼──────────────────────┤
    │  Constrained Gen   │ Via      │ Native   │ Via plugins          │
    │                    │ outlines │ FSM+JF   │                      │
    ├────────────────────┼──────────┼──────────┼──────────────────────┤
    │  Batching          │ Cont.    │ Cont.    │ In-flight            │
    │                    │ batching │ batching │ batching             │
    ├────────────────────┼──────────┼──────────┼──────────────────────┤
    │  Multi-turn Opt    │ Basic    │ Excellent│ Basic                │
    │                    │          │ (DSL)    │                      │
    ├────────────────────┼──────────┼──────────┼──────────────────────┤
    │  Kernel Optim.     │ Custom   │ Custom   │ TensorRT fused       │
    │                    │ CUDA     │ CUDA     │ kernels              │
    ├────────────────────┼──────────┼──────────┼──────────────────────┤
    │  Quantization      │ GPTQ,AWQ│ GPTQ,AWQ │ FP8,INT8,INT4        │
    │                    │          │          │ native               │
    ├────────────────────┼──────────┼──────────┼──────────────────────┤
    │  Ease of Use       │ High     │ High     │ Medium (C++ engine)  │
    ├────────────────────┼──────────┼──────────┼──────────────────────┤
    │  Best For          │ General  │ Complex  │ Max single-request   │
    │                    │ serving  │ LLM apps │ throughput            │
    └────────────────────┴──────────┴──────────┴──────────────────────┘

    Performance (approximate, varies by workload):

    Simple chatbot (single turn):
        vLLM ≈ SGLang ≈ TensorRT-LLM (all similar)

    Multi-turn conversation:
        SGLang > vLLM > TensorRT-LLM
        (SGLang's RadixAttention shines with shared prefixes)

    Structured JSON output:
        SGLang >> vLLM > TensorRT-LLM
        (Jump-forward decoding gives SGLang a big advantage)

    Branching/tree search:
        SGLang >> vLLM >> TensorRT-LLM
        (Fork + RadixAttention = massive KV cache savings)

    Raw throughput (simple prompts, large batch):
        TensorRT-LLM ≥ SGLang ≥ vLLM
        (TensorRT's fused kernels win on raw compute)
"""


# ============================================================================
# PART 7: PUTTING IT ALL TOGETHER — DEMO
# ============================================================================

def demo():
    """
    Demonstrate RadixAttention KV cache sharing and constrained decoding.
    """
    print("=" * 70)
    print("SGLANG CONCEPTS DEMO")
    print("=" * 70)

    # ── Demo 1: Radix Tree KV Cache Sharing ──
    print("\n" + "─" * 70)
    print("DEMO 1: Radix Tree for KV Cache Sharing")
    print("─" * 70)

    tree = RadixTree()

    # Simulate token IDs for different requests
    # System prompt (shared by all requests)
    sys_prompt = [10, 20, 30, 40, 50, 60, 70, 80]  # 8 tokens

    # Three requests with the same system prompt
    req1 = sys_prompt + [100, 101, 102, 103]  # "What is 2+2?"
    req2 = sys_prompt + [200, 201, 202]        # "Tell me a joke"
    req3 = sys_prompt + [100, 101, 300, 301]  # "What is 2+3?"

    # Insert first request
    tree.insert(req1, kv_cache=torch.randn(1, 12, 8))  # dummy KV
    print(f"\nAfter inserting request 1 ({len(req1)} tokens):")
    print(f"  Total cached tokens: {tree.total_cached_tokens}")

    # Check prefix match for request 2
    stats2 = tree.cache_hit_stats(req2)
    print(f"\nRequest 2 prefix match:")
    print(f"  Total tokens: {stats2['total_tokens']}")
    print(f"  Cached (shared prefix): {stats2['cached_tokens']}")
    print(f"  New (need computation): {stats2['new_tokens']}")
    print(f"  Hit rate: {stats2['hit_rate']:.1%}")
    print(f"  → Saved {stats2['cached_tokens']}/{stats2['total_tokens']} "
          f"prefill computations!")

    tree.insert(req2)

    # Check prefix match for request 3 (shares even more with req1)
    stats3 = tree.cache_hit_stats(req3)
    print(f"\nRequest 3 prefix match:")
    print(f"  Total tokens: {stats3['total_tokens']}")
    print(f"  Cached (shared prefix): {stats3['cached_tokens']}")
    print(f"  New (need computation): {stats3['new_tokens']}")
    print(f"  Hit rate: {stats3['hit_rate']:.1%}")
    print(f"  → Shares system prompt + 'What is' prefix with request 1!")

    # ── Demo 2: Constrained JSON Decoding ──
    print("\n" + "─" * 70)
    print("DEMO 2: Constrained JSON Decoding with FSM")
    print("─" * 70)

    fsm = SimpleJSONFSM()
    vocab = ["{", "}", '"', ":", ",", "name", "age", "Alice", "30",
             " ", "true", "false", "[", "]"]

    print("\nGenerating constrained JSON:")
    output = ""
    steps = ["{", '"', "name", '"', ":", '"', "Alice", '"', "}"]

    for token in steps:
        valid = fsm.get_valid_tokens(vocab)
        valid_tokens = [vocab[i] for i in valid]
        print(f"  State: {fsm.state:15s} | Valid: {valid_tokens[:5]}... | "
              f"Generated: '{token}'")
        fsm.advance(token)
        output += token

    print(f"\n  Final output: {output}")
    print(f"  FSM state: {fsm.state}")

    # ── Demo 3: SGLang-like Program ──
    print("\n" + "─" * 70)
    print("DEMO 3: SGLang-like Program with Forking")
    print("─" * 70)

    s = SGLState()
    s.system("You are a math tutor.")
    s.user("Solve: What is the integral of x²?")

    print("\n  Program state after setup:")
    print(f"  Messages: {len(s.messages)}")

    # Fork into 3 approaches
    forks = s.fork(3)
    approaches = ["substitution", "by parts", "direct formula"]
    for i, (f, approach) in enumerate(zip(forks, approaches)):
        f.assistant_gen(f"approach_{i}", constraint=None)
        print(f"  Fork {i}: trying {approach}")
        print(f"    → Shares KV cache for system + user messages with all forks")
        print(f"    → Only computes new KV for the approach-specific tokens")

    # ── Demo 4: Jump-Forward Optimization ──
    print("\n" + "─" * 70)
    print("DEMO 4: Jump-Forward Decoding")
    print("─" * 70)

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "city": {"type": "string"},
        }
    }

    print("\n  Schema: name, age, city")
    print("\n  Without jump-forward:")
    print("    Token 1: '{'  (LLM call)")
    print('    Token 2: \'"\'  (LLM call)')
    print("    Token 3: 'n'  (LLM call) — but this is DETERMINED by schema!")
    print("    Token 4: 'a'  (LLM call) — determined!")
    print("    Token 5: 'm'  (LLM call) — determined!")
    print("    Token 6: 'e'  (LLM call) — determined!")
    print('    Token 7: \'"\'  (LLM call) — determined!')
    print("    Token 8: ':'  (LLM call) — determined!")
    print("    → 8 LLM calls for mostly deterministic output")

    print("\n  With jump-forward:")
    print("    Token 1: '{'  (LLM call)")
    print('    JUMP:    \'{"name": \' (FREE — no LLM calls!)')
    print("    Token 2: actual name value (LLM call)")
    print("    → 2 LLM calls! 4× faster for this portion")

    fsm2 = SimpleJSONFSM()
    fsm2.state = SimpleJSONFSM.OBJECT_OPEN
    jump = jump_forward_decode(fsm2, schema, "{")
    print(f"\n  Jump-forward detected: '{jump}'")

    print(f"\n{'=' * 70}")
    print("KEY TAKEAWAYS:")
    print("  1. RadixAttention shares KV cache via prefix tree → less redundant compute")
    print("  2. FSM-based constrained decoding ensures valid structured output")
    print("  3. Jump-forward skips deterministic tokens → faster structured generation")
    print("  4. Fork/join enables efficient tree search with shared KV cache")
    print("  5. SGLang excels at COMPLEX LLM programs (multi-turn, branching, tools)")
    print("=" * 70)


if __name__ == "__main__":
    demo()

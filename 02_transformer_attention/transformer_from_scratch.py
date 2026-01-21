"""
Transformer Architecture - Learning from Scratch

This module explains the Transformer architecture from the ground up, following
the intuition from the original "Attention Is All You Need" paper (Vaswani et al., 2017).

================================================================================
PART 1: THE PROBLEM - WHY DO WE NEED SOMETHING NEW?
================================================================================

Let's start with what we already know about LSTMs:

    c_t = f_t * c_{t-1} + i_t * g_t    (cell state update)
    h_t = o_t * tanh(c_t)               (hidden state output)

LSTMs solved the vanishing gradient problem with the cell state highway.
But look carefully at these equations: c_t depends on c_{t-1}.

This creates the SEQUENTIAL BOTTLENECK:

    To compute c_5, you need c_4.
    To compute c_4, you need c_3.
    To compute c_3, you need c_2.
    ... and so on.

You MUST process the sequence one step at a time:

    Step 1: Process token 1 → get h_1, c_1
    Step 2: Process token 2 → get h_2, c_2  (needs h_1, c_1)
    Step 3: Process token 3 → get h_3, c_3  (needs h_2, c_2)
    ...

This is slow! Modern GPUs can do thousands of operations in parallel, but
LSTMs force us to do T sequential operations for a sequence of length T.

================================================================================
PART 2: THE KEY QUESTION - DO WE REALLY NEED RECURRENCE?
================================================================================

The authors of the Transformer asked: "What is recurrence actually doing?"

Recurrence serves ONE main purpose: letting information flow between positions.
When processing "it" in "The cat sat on the mat because it was tired",
the LSTM needs to remember "cat" from earlier in the sequence.

But there's another way to let information flow: DIRECT CONNECTIONS.

What if, instead of passing information step by step like a game of telephone:

    "cat" → h_1 → h_2 → h_3 → h_4 → h_5 → h_6 → h_7 → h_8 (reaches "it")

We just let "it" look directly at "cat"?

    "cat" ─────────────────────────────────────────────► "it"
                        (direct connection!)

This is the core insight of the Transformer: REMOVE RECURRENCE, use ATTENTION.

================================================================================
PART 3: WHAT IS ATTENTION? (THE INTUITION)
================================================================================

Attention is a mechanism that lets each position in a sequence "look at" and
"gather information from" all other positions.

Let's build the intuition step by step.

STEP 1: THE PROBLEM OF UNDERSTANDING CONTEXT
--------------------------------------------
Consider the sentence: "The animal didn't cross the street because it was too tired."

What does "it" refer to?
    - "it" = "the animal" (because animals get tired)
    - NOT "the street" (streets don't get tired)

To understand "it", we need to look at other words in the sentence and figure
out which ones are relevant.

STEP 2: THE NAIVE APPROACH - LOOK AT EVERYTHING EQUALLY
-------------------------------------------------------
We could just average all the word representations:

    representation_of_"it" = average(all other words)

But this is too crude. "animal" is very relevant to "it", but "the" is not.
We need WEIGHTED averaging, where relevant words get higher weights.

STEP 3: THE ATTENTION APPROACH - WEIGHTED COMBINATION
-----------------------------------------------------
Attention computes a WEIGHTED SUM of all positions, where the weights
indicate relevance:

    new_representation = w_1 * word_1 + w_2 * word_2 + ... + w_n * word_n

    where w_i = how relevant word_i is to the current word

For "it" looking at the sentence:
    - w_animal = 0.7  (high weight - very relevant!)
    - w_street = 0.1  (low weight - not as relevant)
    - w_the = 0.05    (very low weight - not informative)
    - ... etc

The weights must sum to 1 (they form a probability distribution).

STEP 4: HOW DO WE COMPUTE THE WEIGHTS?
--------------------------------------
This is where Query, Key, and Value come in.

Think of it like a search engine:

    QUERY:  What I'm looking for
            "it" asks: "What noun am I referring to?"

    KEY:    How each word describes itself (for matching)
            "animal" says: "I'm a noun, a living thing"
            "street" says: "I'm a noun, a place"
            "the" says: "I'm just an article"

    VALUE:  The actual information each word provides
            The representation/embedding of each word

The process:
    1. Compare the Query with each Key (compute similarity scores)
    2. Convert scores to weights using softmax (so they sum to 1)
    3. Use weights to combine the Values

================================================================================
PART 4: QUERY, KEY, VALUE - A DEEPER LOOK
================================================================================

Where do Q, K, V come from? They're all computed from the input!

Given an input token x (its embedding), we compute:

    Q = x @ W_Q    (Query: what am I looking for?)
    K = x @ W_K    (Key: how do I describe myself for matching?)
    V = x @ W_V    (Value: what information do I provide?)

W_Q, W_K, W_V are LEARNED weight matrices. The model learns:
    - What questions to ask (W_Q)
    - How to describe tokens for matching (W_K)
    - What information to pass along (W_V)

ANALOGY: DATABASE QUERY
-----------------------
Imagine a database of people:

    Name (Key)      | Info (Value)
    ----------------|------------------
    "John Smith"    | {age: 30, job: "engineer"}
    "Jane Doe"      | {age: 25, job: "doctor"}
    "John Doe"      | {age: 40, job: "teacher"}

Your query: "John"

Similarity scores:
    - "John Smith" vs "John" → high similarity (0.8)
    - "Jane Doe" vs "John"   → low similarity (0.1)
    - "John Doe" vs "John"   → high similarity (0.8)

After softmax: weights = [0.47, 0.06, 0.47]

Result: weighted combination of the values
    = 0.47 * {John Smith's info} + 0.06 * {Jane Doe's info} + 0.47 * {John Doe's info}

This result contains mostly information about the Johns!

ANALOGY: LIBRARY SEARCH
-----------------------
    - Query: Your search term ("machine learning books")
    - Key: Book titles/tags (how each book describes itself)
    - Value: Book content (the actual information)

    You compare your query to all keys, find the most relevant ones,
    and retrieve their values.

================================================================================
PART 5: SELF-ATTENTION - EVERY POSITION ATTENDS TO EVERY POSITION
================================================================================

In SELF-attention, every position in the sequence acts as:
    - A Query (asking questions)
    - A Key (being available to be matched)
    - A Value (providing information)

For a sequence of 5 tokens:

    Token 0: Q_0 compares with K_0, K_1, K_2, K_3, K_4 → gets weighted sum of V_0...V_4
    Token 1: Q_1 compares with K_0, K_1, K_2, K_3, K_4 → gets weighted sum of V_0...V_4
    Token 2: Q_2 compares with K_0, K_1, K_2, K_3, K_4 → gets weighted sum of V_0...V_4
    ... and so on

EVERY token can look at EVERY other token. No recurrence needed!

This produces an ATTENTION MATRIX of size (seq_len × seq_len):

                    Keys
                K_0  K_1  K_2  K_3  K_4
              ┌─────────────────────────┐
    Q_0       │ 0.5  0.2  0.1  0.1  0.1 │  (weights for token 0)
    Q_1       │ 0.3  0.4  0.2  0.05 0.05│  (weights for token 1)
    Queries   │ 0.1  0.1  0.6  0.1  0.1 │  (weights for token 2)
    Q_3       │ 0.2  0.2  0.2  0.3  0.1 │  (weights for token 3)
    Q_4       │ 0.1  0.3  0.1  0.1  0.4 │  (weights for token 4)
              └─────────────────────────┘

Each row sums to 1 (softmax). Each row tells us which positions that token
attends to.

================================================================================
PART 6: THE MATH - SCALED DOT-PRODUCT ATTENTION
================================================================================

Now let's see the actual formula. Don't worry - it directly implements
the intuition we just built!

Given:
    Q: Queries, shape (seq_len, d_k)    - d_k is the dimension of queries/keys
    K: Keys, shape (seq_len, d_k)
    V: Values, shape (seq_len, d_v)     - d_v is the dimension of values

The attention formula:

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

Let's break this down step by step:

STEP 1: Q @ K^T - Compute Similarity Scores
-------------------------------------------
This computes the dot product between each query and all keys.

    Q has shape (seq_len, d_k)
    K^T has shape (d_k, seq_len)
    Q @ K^T has shape (seq_len, seq_len)

The result is a matrix where entry [i, j] = "how much should position i
attend to position j?"

Why dot product for similarity? If two vectors point in similar directions,
their dot product is large. If they're orthogonal, dot product is zero.

    Q_i = [1, 0, 0]
    K_j = [1, 0, 0]  → Q_i · K_j = 1  (very similar!)

    Q_i = [1, 0, 0]
    K_j = [0, 1, 0]  → Q_i · K_j = 0  (orthogonal, no similarity)

STEP 2: / sqrt(d_k) - Scale Down
--------------------------------
We divide by sqrt(d_k) to prevent the dot products from getting too large.

Why does this matter? The paper explains:

    "For large values of d_k, the dot products grow large in magnitude,
     pushing the softmax function into regions where it has extremely
     small gradients."

Example with d_k = 512:
    - Each dot product is a sum of 512 terms
    - If Q and K have random values with variance 1
    - The sum has variance ≈ 512
    - So dot products might be around ±30 or larger

    softmax([30, -30, 0]) ≈ [1.0, 0.0, 0.0]  (almost one-hot!)

    The gradient of softmax near 1.0 or 0.0 is almost zero.
    This is the "softmax saturation" problem - similar to vanishing gradients!

Dividing by sqrt(512) ≈ 22.6 brings the values back to a reasonable range:
    [30, -30, 0] / 22.6 ≈ [1.3, -1.3, 0]
    softmax([1.3, -1.3, 0]) ≈ [0.54, 0.04, 0.15]  (much better!)

STEP 3: softmax(...) - Convert to Weights
-----------------------------------------
Softmax converts raw scores to probabilities (weights that sum to 1).

    softmax([2.0, 1.0, 0.5]) = [0.59, 0.24, 0.17]

Each row of the attention matrix becomes a probability distribution:
"This is how much attention to give to each position."

STEP 4: ... @ V - Weighted Sum of Values
----------------------------------------
Finally, we use the attention weights to combine the values.

    attention_weights has shape (seq_len, seq_len)
    V has shape (seq_len, d_v)
    output has shape (seq_len, d_v)

For position i:
    output[i] = sum_j (attention_weight[i,j] * V[j])

This is the weighted combination we talked about in the intuition!
Positions with high attention weights contribute more to the output.

================================================================================
PART 7: MULTI-HEAD ATTENTION - WHY ONE HEAD ISN'T ENOUGH
================================================================================

So far we've described single-head attention. But the Transformer uses
MULTI-HEAD attention. Why?

FIRST: WHAT IS A "HEAD"?
------------------------
A "head" is ONE complete attention mechanism with its own Q, K, V projections.

Think of it this way:
    - A head is like ONE person reading a sentence
    - That person can only focus on ONE type of relationship at a time
    - They produce ONE attention matrix (who attends to whom)

One head = One set of W_Q, W_K, W_V matrices = One attention pattern

    Input → [W_Q, W_K, W_V] → Q, K, V → Attention(Q,K,V) → Output
    └──────────────────── ONE HEAD ────────────────────────────┘

When we say "8-head attention", we mean 8 SEPARATE attention mechanisms
running in PARALLEL, each with its own learned weights, each looking
for different patterns.

THE PROBLEM WITH SINGLE-HEAD ATTENTION
--------------------------------------
With one attention head, each position computes ONE set of attention weights.
But language has many different types of relationships!

Consider: "The cat that I saw yesterday sat on the mat."

Position "sat" might need to attend to:
    - "cat" (subject-verb relationship: WHO sat?)
    - "mat" (verb-object relationship: sat WHERE?)
    - "yesterday" (temporal relationship: WHEN?)

With a single head, the model must average these different needs into
one set of weights. The paper says this "averaging inhibits" the ability
to capture diverse relationships.

THE SOLUTION: MULTIPLE HEADS IN PARALLEL
----------------------------------------
Multi-head attention runs MULTIPLE attention operations in parallel,
each with its own learned projections:

    Head 1: Maybe learns to focus on subject-verb relationships
    Head 2: Maybe learns to focus on object relationships
    Head 3: Maybe learns to focus on nearby words
    Head 4: Maybe learns to focus on punctuation/structure
    ... etc

Each head operates in a DIFFERENT "subspace" of the representation,
looking for different patterns.

THE FORMULA
-----------
    MultiHead(Q, K, V) = Concat(head_1, head_2, ..., head_h) @ W_O

    where head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)

Breaking it down:
    1. Each head has its OWN W_Q, W_K, W_V matrices
    2. These project the input into a smaller subspace (d_k = d_model / num_heads)
    3. Each head computes attention independently
    4. Results are concatenated and projected back to d_model

Example with d_model=512 and num_heads=8:
    - Each head works with d_k = 512/8 = 64 dimensions
    - 8 heads produce 8 outputs of size 64
    - Concatenate: 8 × 64 = 512
    - Final projection W_O: 512 → 512

The computational cost is similar to single-head attention with full
dimensionality, but we get richer representations!

VISUALIZING MULTI-HEAD ATTENTION
--------------------------------
    Input x (512 dimensions)
         │
         ├──────────────┬──────────────┬─────── ... ───┬──────────────┐
         ▼              ▼              ▼               ▼              │
    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐         │
    │ Head 1  │    │ Head 2  │    │ Head 3  │    │ Head 8  │         │
    │ 64 dims │    │ 64 dims │    │ 64 dims │    │ 64 dims │         │
    │         │    │         │    │         │    │         │         │
    │ W_Q_1   │    │ W_Q_2   │    │ W_Q_3   │    │ W_Q_8   │         │
    │ W_K_1   │    │ W_K_2   │    │ W_K_3   │    │ W_K_8   │         │
    │ W_V_1   │    │ W_V_2   │    │ W_V_3   │    │ W_V_8   │         │
    └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘         │
         │              │              │               │              │
         │   (each head computes its own attention)    │              │
         │              │              │               │              │
         ▼              ▼              ▼               ▼              │
      out_1          out_2          out_3          out_8             │
      (64d)          (64d)          (64d)          (64d)             │
         │              │              │               │              │
         └──────────────┴──────────────┴───────────────┘              │
                                │                                     │
                        Concatenate (512d)                            │
                                │                                     │
                                ▼                                     │
                        ┌──────────────┐                              │
                        │     W_O      │  Final projection            │
                        │  512 → 512   │                              │
                        └──────────────┘                              │
                                │                                     │
                                ▼                                     │
                         Output (512d)

Each head learns to look for DIFFERENT patterns in the data!

================================================================================
PART 8: THE TRANSFORMER BLOCK - PUTTING PIECES TOGETHER
================================================================================

Now we can build a complete Transformer block. Each block has:

    1. Multi-Head Self-Attention (positions communicate)
    2. Add & Normalize (residual connection + layer norm)
    3. Feed-Forward Network (process each position)
    4. Add & Normalize (another residual connection)

Diagram:

    Input (seq_len, d_model)
      │
      ▼
    ┌─────────────────────────┐
    │   Multi-Head Attention  │◄── Every position attends to every position
    └─────────────────────────┘    (this is where positions "talk" to each other)
      │
      ├──────────────────────────► + ◄── Add original input (residual connection)
      │                            │
      ▼                            ▼
    ┌─────────────────────────────────┐
    │          LayerNorm              │◄── Normalize (stabilizes training)
    └─────────────────────────────────┘
      │
      ▼
    ┌─────────────────────────┐
    │   Feed-Forward Network  │◄── Process each position independently
    └─────────────────────────┘    (expand, ReLU, contract)
      │
      ├──────────────────────────► + ◄── Add input to FFN (another residual)
      │                            │
      ▼                            ▼
    ┌─────────────────────────────────┐
    │          LayerNorm              │
    └─────────────────────────────────┘
      │
      ▼
    Output (seq_len, d_model)


WHY RESIDUAL CONNECTIONS? (THE "ADD" PART)
------------------------------------------
Remember how LSTM's cell state highway helped gradients flow?

    c_t = c_{t-1} + stuff    (LSTM)

Residual connections do the same thing:

    output = x + layer(x)    (Residual)

The gradient of x + layer(x) with respect to x is:

    d(x + layer(x))/dx = 1 + d(layer(x))/dx

Even if d(layer(x))/dx is tiny, the gradient is at least 1!
This creates a "gradient highway" just like in LSTMs.

In the original Transformer paper (2017), they stacked 6 blocks.
Modern models like GPT-3 stack 96 blocks! Without residual connections,
gradients would vanish through all those layers.


WHY LAYER NORMALIZATION? (THE "NORM" PART)
------------------------------------------
Layer normalization stabilizes training by normalizing activations:

    LayerNorm(x) = (x - mean(x)) / std(x) * gamma + beta

where gamma and beta are learned parameters.

Why not Batch Normalization (used in CNNs)?
    - BatchNorm normalizes across the batch dimension
    - Requires a reasonable batch size to compute statistics
    - Doesn't work well with varying sequence lengths

LayerNorm normalizes across the feature dimension (d_model):
    - Each token is normalized independently
    - Works with any batch size
    - Works with any sequence length


FEED-FORWARD NETWORK - THE "THINKING" LAYER
-------------------------------------------
The FFN is surprisingly simple:

    FFN(x) = ReLU(x @ W_1 + b_1) @ W_2 + b_2

The dimensions:
    - Input: d_model (e.g., 512)
    - Hidden: d_ff (e.g., 2048) - typically 4x larger!
    - Output: d_model (back to original size)

Why expand then contract?

    512 → 2048 → 512

    The expanded representation gives the model more "room to think."
    Research suggests a lot of the model's knowledge is stored in FFN weights.

Important: The FFN is applied to each position INDEPENDENTLY.
    - Position 0 gets transformed by FFN
    - Position 1 gets transformed by FFN (same weights, different input)
    - Position 2 gets transformed by FFN
    - ... etc

This is different from attention, which mixes information BETWEEN positions.

Think of it as:
    - Attention: "Let positions talk to each other"
    - FFN: "Let each position think on its own"

================================================================================
PART 9: POSITIONAL ENCODING - THE MISSING PIECE
================================================================================

There's a problem with attention that we haven't addressed yet.

THE PROBLEM: ATTENTION DOESN'T KNOW ABOUT ORDER
-----------------------------------------------
Look at the attention formula again:

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

If you shuffle the input tokens, the attention weights just get shuffled too!
The model has NO idea what order the tokens are in.

But word order matters a lot!
    "Dog bites man" ≠ "Man bites dog"

LSTMs naturally knew about order because they processed tokens one by one.
But we removed recurrence! How does the Transformer know position?

THE SOLUTION: ADD POSITION INFORMATION TO EMBEDDINGS
----------------------------------------------------
Before feeding tokens into the Transformer, we ADD positional information
to their embeddings:

    input_to_transformer = token_embedding + positional_encoding

If token_embedding for "cat" is [0.2, 0.5, 0.1, ...], and the positional
encoding for position 3 is [0.01, -0.02, 0.03, ...], then:

    input = [0.21, 0.48, 0.13, ...]  (element-wise addition)

Now the model can tell that this "cat" is at position 3!

HOW TO ENCODE POSITION? - SINUSOIDAL ENCODING
---------------------------------------------
The original paper uses a clever scheme with sine and cosine waves:

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Where:
    - pos = position in sequence (0, 1, 2, 3, ...)
    - i = dimension index (0, 1, 2, ... d_model/2)
    - 2i means even dimensions, 2i+1 means odd dimensions

This creates a unique "fingerprint" for each position:

    Position 0: [sin(0), cos(0), sin(0), cos(0), ...]
    Position 1: [sin(1/1), cos(1/1), sin(1/100), cos(1/100), ...]
    Position 2: [sin(2/1), cos(2/1), sin(2/100), cos(2/100), ...]
    ...

Why sinusoids?
    1. Each dimension oscillates at a different frequency
    2. The model can learn to compute relative positions
       (sin and cos allow linear transformations to shift positions)
    3. Can extrapolate to longer sequences than seen in training

WHY NOT JUST USE [0, 1, 2, 3, ...]?
-----------------------------------
You could encode position as just a number, but:
    - Large positions would dominate the embedding
    - Doesn't generalize well to longer sequences
    - Harder for the model to learn relative positions

The sinusoidal encoding keeps values bounded (-1 to 1) and has nice
mathematical properties for learning position relationships.

ALTERNATIVE: LEARNED POSITIONAL EMBEDDINGS
------------------------------------------
Modern models (like GPT) often just LEARN the positional embeddings:

    positional_embedding = nn.Embedding(max_seq_len, d_model)

This is simpler and often works just as well for fixed-length contexts.
The tradeoff is that it can't extrapolate to positions beyond max_seq_len.

================================================================================
PART 10: CAUSAL MASKING - PREVENTING CHEATING
================================================================================

For language models that generate text (like GPT), we have a problem:

THE PROBLEM: THE MODEL CAN SEE THE FUTURE
-----------------------------------------
In self-attention, every position attends to EVERY position.
But when generating text, we're predicting the NEXT word!

If we're predicting position 5, the model shouldn't be able to look at
positions 5, 6, 7, ... because those are the "future" we're trying to predict.

Example - predicting "sat" in "The cat sat on the mat":

    Position:  0    1    2    3   4   5
    Token:    The  cat  sat  on  the mat

    When predicting position 2 ("sat"), the model should only see:
    - Position 0: "The" ✓
    - Position 1: "cat" ✓
    - Position 2: "sat" ✗ (this is what we're predicting!)
    - Position 3: "on"  ✗ (future)
    - Position 4: "the" ✗ (future)
    - Position 5: "mat" ✗ (future)

If the model could see "sat" while predicting "sat", it would just copy!
This is called "cheating" or "data leakage."

THE SOLUTION: CAUSAL MASK (LOWER TRIANGULAR MATRIX)
---------------------------------------------------
We use a mask to block attention to future positions:

    mask = [[1, 0, 0, 0],      Position 0 can only see position 0
            [1, 1, 0, 0],      Position 1 can see positions 0, 1
            [1, 1, 1, 0],      Position 2 can see positions 0, 1, 2
            [1, 1, 1, 1]]      Position 3 can see all (0, 1, 2, 3)

    1 = can attend
    0 = cannot attend (will be blocked)

HOW MASKING WORKS MATHEMATICALLY
--------------------------------
Before applying softmax, we set the masked positions to negative infinity:

    attention_scores = Q @ K^T / sqrt(d_k)

    scores before masking:  [[0.5, 0.2, 0.3, 0.1],
                             [0.4, 0.6, 0.2, 0.3],
                             [0.1, 0.3, 0.8, 0.2],
                             [0.2, 0.1, 0.4, 0.5]]

    scores after masking:   [[0.5, -∞,  -∞,  -∞],
                             [0.4, 0.6, -∞,  -∞],
                             [0.1, 0.3, 0.8, -∞],
                             [0.2, 0.1, 0.4, 0.5]]

Now when we apply softmax:

    softmax([0.5, -∞, -∞, -∞]) = [1.0, 0.0, 0.0, 0.0]
    (exp(-∞) = 0, so masked positions get zero weight!)

The masked positions contribute NOTHING to the output.

WHY "CAUSAL"?
-------------
It's called "causal" because it enforces a causal relationship:
    - The output at position t only depends on inputs at positions ≤ t
    - Just like in real life: the present can only be caused by the past!

This is also called:
    - "Autoregressive" masking
    - "Decoder" masking
    - "Left-to-right" attention

================================================================================
PART 11: PUTTING IT ALL TOGETHER - TRANSFORMER VARIANTS
================================================================================

The Transformer architecture can be used in different configurations:

ENCODER-ONLY (e.g., BERT)
-------------------------
Used for: Understanding, classification, filling in blanks

    Input: "The [MASK] sat on the mat"
    Output: Representation of each token (or prediction for [MASK])

    - NO causal masking (can see the entire sequence)
    - Bidirectional attention (every position sees every other position)
    - Good for: sentiment analysis, named entity recognition, question answering

DECODER-ONLY (e.g., GPT)
------------------------
Used for: Text generation, language modeling

    Input: "The cat sat"
    Output: Predict next token ("on")

    - WITH causal masking (can only see past and present)
    - Unidirectional attention (left-to-right only)
    - Good for: text generation, chatbots, code completion

    This is what ChatGPT, GPT-4, Claude, etc. use!

ENCODER-DECODER (original Transformer)
--------------------------------------
Used for: Translation, summarization

    Encoder input: "The cat sat on the mat" (English)
    Decoder output: "Le chat était assis sur le tapis" (French)

    - Encoder: bidirectional (understand source fully)
    - Decoder: causal + cross-attention to encoder
    - Good for: translation, summarization, text-to-SQL

================================================================================
PART 12: WHY TRANSFORMERS BEAT LSTMs - THE SUMMARY
================================================================================

1. PARALLELIZATION
------------------
    LSTM:
        Process token 1 → wait → process token 2 → wait → process token 3...
        O(T) sequential steps

    Transformer:
        Process all tokens simultaneously!
        O(1) sequential steps (just the number of layers)

    On a GPU with thousands of cores, this is a MASSIVE speedup.

2. LONG-RANGE DEPENDENCIES
--------------------------
    LSTM:
        "cat" → h1 → h2 → h3 → h4 → h5 → h6 → h7 → "it"
        Information must travel through T steps
        Path length: O(T)

    Transformer:
        "cat" ────────────────────────────────────► "it"
        Direct attention connection!
        Path length: O(1)

    Even with the LSTM highway, distant information can degrade.
    Transformers have DIRECT connections between any two positions.

3. SCALABILITY
--------------
    Transformers scale incredibly well with:
    - More data
    - More compute
    - More parameters

    This led to:
    - GPT-3 (175B parameters)
    - GPT-4 (rumored ~1T parameters)
    - Modern large language models

THE TRADEOFF
------------
    Attention is O(T²) in computation and memory:
    - Every token attends to every other token
    - For sequence length T, that's T × T attention scores

    For T = 1000: 1,000,000 attention scores per head per layer!
    For T = 10000: 100,000,000 attention scores!

    This is why context windows were limited (2K, 4K, 8K tokens).
    Active research on "efficient attention" addresses this:
    - Sparse attention
    - Linear attention
    - Flash attention (memory-efficient)

================================================================================
SUMMARY: THE TRANSFORMER IN ONE PAGE
================================================================================

1. PROBLEM: LSTMs are slow (sequential) and struggle with long sequences

2. SOLUTION: Attention - let every position look at every other position directly

3. KEY COMPONENTS:
   - Embedding: Convert tokens to vectors
   - Positional Encoding: Add position information (attention doesn't know order)
   - Multi-Head Self-Attention: Positions communicate with each other
   - Feed-Forward Network: Each position processes independently
   - Residual Connections: Gradient highway (like LSTM cell state)
   - Layer Normalization: Stabilize training

4. CAUSAL MASKING: For generation, prevent looking at future tokens

5. FORMULA:
   Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

   - Q @ K^T: How much should each position attend to each other position?
   - / sqrt(d_k): Keep softmax in a good gradient range
   - softmax: Convert to probability weights
   - @ V: Weighted sum of values

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class ScaledDotProductAttention(nn.Module):
    """
    Implements the attention formula:

        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    This is the core building block of the Transformer.
    """

    def __init__(self, d_k: int):
        super().__init__()
        self.d_k = d_k
        self.scale = math.sqrt(d_k)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            Q: Queries, shape (batch, seq_len, d_k)
            K: Keys, shape (batch, seq_len, d_k)
            V: Values, shape (batch, seq_len, d_v)
            mask: Optional mask, shape (seq_len, seq_len) or (batch, seq_len, seq_len)
                  1 = attend, 0 = don't attend

        Returns:
            output: Attention output, shape (batch, seq_len, d_v)
        """
        # Step 1: Compute attention scores
        # Q @ K^T: (batch, seq_len, d_k) @ (batch, d_k, seq_len) -> (batch, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # Step 2: Scale by sqrt(d_k)
        scores = scores / self.scale

        # Step 3: Apply mask (if provided)
        if mask is not None:
            # Set masked positions to -inf so softmax gives 0
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Step 4: Softmax to get attention weights
        # Each row sums to 1 (probability distribution over positions)
        attention_weights = F.softmax(scores, dim=-1)

        # Step 5: Weighted sum of values
        # (batch, seq_len, seq_len) @ (batch, seq_len, d_v) -> (batch, seq_len, d_v)
        output = torch.matmul(attention_weights, V)

        return output


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention allows the model to attend to information
    from different representation subspaces at different positions.

        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
        where head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head

        # Linear projections for Q, K, V (all heads computed together)
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # Output projection
        self.W_O = nn.Linear(d_model, d_model)

        # Attention mechanism
        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: Shape (batch, seq_len, d_model)
            key: Shape (batch, seq_len, d_model)
            value: Shape (batch, seq_len, d_model)
            mask: Optional mask

        Returns:
            output: Shape (batch, seq_len, d_model)
        """
        batch_size = query.size(0)

        # Step 1: Linear projections
        Q = self.W_Q(query)  # (batch, seq_len, d_model)
        K = self.W_K(key)
        V = self.W_V(value)

        # Step 2: Split into multiple heads
        # Reshape: (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k)
        # Transpose: -> (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Step 3: Apply attention to all heads in parallel
        # The attention mechanism works on the last two dimensions
        # So (batch, num_heads, seq_len, d_k) is treated as (batch*num_heads, seq_len, d_k)
        attn_output = self.attention(Q, K, V, mask)

        # Step 4: Concatenate heads
        # Transpose: (batch, num_heads, seq_len, d_k) -> (batch, seq_len, num_heads, d_k)
        # Reshape: -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)

        # Step 5: Final linear projection
        output = self.W_O(attn_output)

        return output


class PositionwiseFeedForward(nn.Module):
    """
    The Feed-Forward Network in a Transformer block.

        FFN(x) = ReLU(x @ W_1 + b_1) @ W_2 + b_2

    Applied to each position independently (hence "position-wise").
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Shape (batch, seq_len, d_model)

        Returns:
            output: Shape (batch, seq_len, d_model)
        """
        # Expand to d_ff, apply ReLU, contract back to d_model
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from "Attention Is All You Need".

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    This adds information about position to the embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute the div term: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (max_len, d_model) -> (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter, but should be saved with model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Shape (batch, seq_len, d_model)

        Returns:
            x + positional_encoding
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """
    A single Transformer block (decoder-style with causal masking option).

    Structure:
        x -> MultiHeadAttention -> Add & LayerNorm -> FFN -> Add & LayerNorm -> output
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Shape (batch, seq_len, d_model)
            mask: Optional causal mask

        Returns:
            output: Shape (batch, seq_len, d_model)
        """
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)  # Q=K=V=x for self-attention
        x = self.norm1(x + self.dropout(attn_output))  # Add & Norm

        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))  # Add & Norm

        return x


class TransformerDecoder(nn.Module):
    """
    A decoder-only Transformer (like GPT).

    This is what's used for language modeling:
    - Takes token indices as input
    - Predicts next token probabilities
    - Uses causal masking so each position only sees previous positions
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Stack of Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

        # Output projection to vocabulary
        self.output_proj = nn.Linear(d_model, vocab_size)

    def generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate a causal mask for autoregressive decoding.

        Returns a lower-triangular matrix of 1s:
            [[1, 0, 0, 0],
             [1, 1, 0, 0],
             [1, 1, 1, 0],
             [1, 1, 1, 1]]
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token indices, shape (batch, seq_len)

        Returns:
            logits: Shape (batch, seq_len, vocab_size)
        """
        seq_len = x.size(1)
        device = x.device

        # Generate causal mask
        mask = self.generate_causal_mask(seq_len, device)

        # Token embedding + positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)  # Scale embedding
        x = self.pos_encoding(x)

        # Pass through Transformer blocks
        for layer in self.layers:
            x = layer(x, mask)

        # Final norm and projection
        x = self.norm(x)
        logits = self.output_proj(x)

        return logits


# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def visualize_attention():
    """
    Visualize how attention weights look for a simple example.
    """
    print("=" * 70)
    print("VISUALIZING ATTENTION WEIGHTS")
    print("=" * 70)

    # Create simple attention layer
    d_k = 4
    attention = ScaledDotProductAttention(d_k)

    # Create synthetic Q, K, V
    # Let's make tokens where some are more "similar" than others
    batch_size = 1
    seq_len = 5

    # Q, K that will create interesting attention patterns
    Q = torch.tensor([[[1, 0, 0, 0],     # Token 0 is "type A"
                       [1, 0, 0, 0],     # Token 1 is "type A"
                       [0, 1, 0, 0],     # Token 2 is "type B"
                       [0, 1, 0, 0],     # Token 3 is "type B"
                       [0, 0, 1, 0]]], dtype=torch.float)  # Token 4 is "type C"

    K = Q.clone()  # Same as Q for this example
    V = torch.randn(batch_size, seq_len, d_k)  # Random values

    # Compute attention (no mask)
    with torch.no_grad():
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        weights = F.softmax(scores, dim=-1)

    print("\nQuery/Key vectors (showing 'type' of each token):")
    print("  Token 0: type A [1,0,0,0]")
    print("  Token 1: type A [1,0,0,0]")
    print("  Token 2: type B [0,1,0,0]")
    print("  Token 3: type B [0,1,0,0]")
    print("  Token 4: type C [0,0,1,0]")

    print("\nAttention weights (which tokens each position attends to):")
    print("  (row = query position, col = key position)")
    print()

    weights_np = weights[0].numpy()
    print("        ", end="")
    for j in range(seq_len):
        print(f"  Key{j}", end="")
    print()

    for i in range(seq_len):
        print(f"  Query{i}", end="")
        for j in range(seq_len):
            print(f"  {weights_np[i, j]:.2f}", end="")
        print()

    print("\nNotice:")
    print("  - Tokens 0,1 (type A) attend mostly to each other")
    print("  - Tokens 2,3 (type B) attend mostly to each other")
    print("  - Token 4 (type C) has no similar keys, so attends uniformly")
    print("=" * 70)


def visualize_causal_mask():
    """
    Show how causal masking works for autoregressive generation.
    """
    print("=" * 70)
    print("VISUALIZING CAUSAL MASKING")
    print("=" * 70)

    seq_len = 5

    # Create causal mask
    mask = torch.tril(torch.ones(seq_len, seq_len))

    print("\nCausal mask (1 = can attend, 0 = cannot attend):")
    print()
    print("           Key positions")
    print("           0  1  2  3  4")
    print("        ┌─────────────────┐")
    for i in range(seq_len):
        print(f"Query {i} │", end="")
        for j in range(seq_len):
            print(f"  {int(mask[i, j])}", end="")
        print("  │")
    print("        └─────────────────┘")

    print("\nInterpretation:")
    print("  - Position 0 can only see position 0")
    print("  - Position 1 can see positions 0, 1")
    print("  - Position 2 can see positions 0, 1, 2")
    print("  - etc.")
    print("\nThis prevents the model from 'cheating' by looking at future tokens!")

    # Show effect on attention scores
    print("\nEffect on attention scores:")
    scores = torch.randn(seq_len, seq_len)
    masked_scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(masked_scores, dim=-1)

    print("\nOriginal scores (random):")
    for i in range(seq_len):
        print(f"  [{', '.join([f'{s:.1f}' for s in scores[i].tolist()])}]")

    print("\nAfter masking + softmax:")
    for i in range(seq_len):
        print(f"  [{', '.join([f'{w:.2f}' for w in weights[i].tolist()])}]")

    print("\nNotice: Each row only has non-zero weights for positions <= its index")
    print("=" * 70)


def compare_lstm_vs_transformer():
    """
    Compare the computational patterns of LSTM vs Transformer.
    """
    print("=" * 70)
    print("LSTM vs TRANSFORMER: COMPUTATIONAL COMPARISON")
    print("=" * 70)

    print("""
    LSTM Processing Pattern:
    ========================

    Time →

    x_1 ──► [LSTM] ──► h_1 ──► [LSTM] ──► h_2 ──► [LSTM] ──► h_3 ──► ...
                        │                │                │
                        ▼                ▼                ▼
                      output           output           output

    - Each step MUST wait for the previous step
    - Cannot parallelize across time
    - O(T) sequential steps


    Transformer Processing Pattern:
    ===============================

    x_1     x_2     x_3     x_4     x_5
     │       │       │       │       │
     ▼       ▼       ▼       ▼       ▼
    ┌───────────────────────────────────┐
    │         Self-Attention            │  ← All positions processed in parallel!
    │  (every token attends to every    │
    │   other token simultaneously)     │
    └───────────────────────────────────┘
     │       │       │       │       │
     ▼       ▼       ▼       ▼       ▼
    out_1   out_2   out_3   out_4   out_5

    - All positions computed in parallel
    - O(1) sequential steps (just the number of layers)
    - Much better GPU utilization


    Trade-off:
    ==========
    - LSTM: O(T) time, O(1) memory per step
    - Transformer: O(1) depth, but O(T^2) attention computation

    For modern hardware (GPUs), the Transformer's parallelism wins
    despite the O(T^2) attention cost (up to reasonable sequence lengths).
    """)
    print("=" * 70)


def demonstrate_transformer_forward():
    """
    Run a forward pass through a small Transformer to see shapes.
    """
    print("=" * 70)
    print("TRANSFORMER FORWARD PASS DEMONSTRATION")
    print("=" * 70)

    # Small model for demonstration
    vocab_size = 100
    d_model = 64
    num_heads = 4
    num_layers = 2
    d_ff = 256

    model = TransformerDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff
    )

    # Sample input
    batch_size = 2
    seq_len = 10
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    print(f"\nModel configuration:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads} (d_k = {d_model // num_heads} per head)")
    print(f"  num_layers: {num_layers}")
    print(f"  d_ff: {d_ff}")

    print(f"\nInput shape: {tuple(x.shape)} (batch_size, seq_len)")
    print(f"Input (token indices):\n{x}")

    # Forward pass
    with torch.no_grad():
        logits = model(x)

    print(f"\nOutput shape: {tuple(logits.shape)} (batch_size, seq_len, vocab_size)")
    print(f"\nInterpretation:")
    print(f"  - For each position, we get {vocab_size} logits (one per vocab token)")
    print(f"  - Apply softmax to get probability of next token")
    print(f"  - Position i predicts the token at position i+1")

    # Show prediction for last position
    last_logits = logits[0, -1, :]  # First batch, last position
    probs = F.softmax(last_logits, dim=-1)
    top_5 = torch.topk(probs, 5)

    print(f"\nTop 5 predicted next tokens (first sequence):")
    for i, (prob, idx) in enumerate(zip(top_5.values, top_5.indices)):
        print(f"  {i+1}. Token {idx.item()}: {prob.item():.4f}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print("=" * 70)


if __name__ == "__main__":
    visualize_attention()
    print("\n")
    visualize_causal_mask()
    print("\n")
    compare_lstm_vs_transformer()
    print("\n")
    demonstrate_transformer_forward()

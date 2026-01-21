"""
Vanilla RNN Implementation from Scratch

This module implements an RNN cell and layer WITHOUT using PyTorch's built-in
nn.RNN. The goal is to make the sequential bottleneck explicit.

THE SEQUENTIAL BOTTLENECK:
==========================
The fundamental equation of an RNN is:
    h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b_h)
    y_t = W_hy @ h_t + b_y

Notice that h_t DEPENDS on h_{t-1}. This means:
1. You CANNOT compute h_t until h_{t-1} is ready
2. You CANNOT parallelize across time steps
3. For a sequence of length T, you need T sequential operations

This is in stark contrast to Transformers, where all positions can be
computed in parallel using self-attention.

Author: Learning project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import time


class VanillaRNNCell(nn.Module):
    """
    A single RNN cell that processes ONE timestep.

    This is the atomic unit of recurrence. It takes:
    - x_t: input at current timestep (batch_size, input_size)
    - h_prev: hidden state from previous timestep (batch_size, hidden_size)

    And produces:
    - h_t: new hidden state (batch_size, hidden_size)

    The computation is:
        h_t = tanh(W_xh @ x_t + W_hh @ h_prev + b_h)
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Weight matrix for input: maps input_size -> hidden_size
        self.W_xh = nn.Linear(input_size, hidden_size, bias=False)

        # Weight matrix for hidden state: maps hidden_size -> hidden_size
        # This is the RECURRENT connection - it's what creates the sequential dependency
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)

        # Bias term
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

        # Initialize weights (important for training stability)
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for better gradient flow."""
        nn.init.xavier_uniform_(self.W_xh.weight)
        nn.init.xavier_uniform_(self.W_hh.weight)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        Process a single timestep.

        Args:
            x_t: Input at time t, shape (batch_size, input_size)
            h_prev: Hidden state from time t-1, shape (batch_size, hidden_size)

        Returns:
            h_t: New hidden state, shape (batch_size, hidden_size)
        """
        # THE CORE RNN COMPUTATION
        # Note: h_t depends on h_prev - this is the sequential bottleneck!
        h_t = torch.tanh(self.W_xh(x_t) + self.W_hh(h_prev) + self.b_h)
        return h_t


class VanillaRNN(nn.Module):
    """
    Full RNN layer that processes an entire sequence.

    This demonstrates the SEQUENTIAL BOTTLENECK explicitly by using a
    Python for-loop over time steps. There's no way to avoid this loop
    in a true RNN - each step depends on the previous one.

    Compare this to a Transformer, where you could replace the loop with
    a single matrix multiplication (self-attention) that processes all
    positions in parallel.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Stack of RNN cells (for multi-layer RNN)
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            cell_input_size = input_size if layer == 0 else hidden_size
            self.cells.append(VanillaRNNCell(cell_input_size, hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        h_0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process an entire sequence.

        Args:
            x: Input sequence, shape (batch_size, seq_len, input_size)
            h_0: Initial hidden state, shape (num_layers, batch_size, hidden_size)
                 If None, initialized to zeros.

        Returns:
            output: All hidden states, shape (batch_size, seq_len, hidden_size)
            h_n: Final hidden state, shape (num_layers, batch_size, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Initialize hidden state if not provided
        if h_0 is None:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

        # Current hidden states for each layer
        h = [h_0[layer] for layer in range(self.num_layers)]

        # Collect outputs
        outputs = []

        # =================================================================
        # THE SEQUENTIAL BOTTLENECK IS RIGHT HERE
        # =================================================================
        # We MUST process timesteps one by one because h_t depends on h_{t-1}
        # There is no way to parallelize this loop!
        #
        # In contrast, a Transformer's self-attention can compute all
        # positions in a single matrix multiplication.
        # =================================================================
        for t in range(seq_len):
            # Extract input at timestep t
            x_t = x[:, t, :]  # Shape: (batch_size, input_size)

            # Process through each layer
            layer_input = x_t
            for layer, cell in enumerate(self.cells):
                # h[layer] is h_{t-1} for this layer
                # After this call, h[layer] becomes h_t
                h[layer] = cell(layer_input, h[layer])
                layer_input = h[layer]  # Output of this layer is input to next

            # Store the output of the final layer
            outputs.append(h[-1])

        # Stack outputs: (seq_len, batch_size, hidden_size) -> (batch_size, seq_len, hidden_size)
        output = torch.stack(outputs, dim=1)

        # Stack final hidden states
        h_n = torch.stack(h, dim=0)

        return output, h_n


def demonstrate_training():
    """
    Demonstrate how to train an RNN.

    This shows the complete training loop:
    1. Forward pass - compute predictions
    2. Compute loss - how wrong are we?
    3. Backward pass - compute gradients (autograd does this!)
    4. Update weights - optimizer adjusts parameters

    We'll train on a simple task: predict the next character in a sequence.
    """
    print("=" * 60)
    print("DEMONSTRATING RNN TRAINING")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # =================================================================
    # SETUP: Define a simple character-level prediction task
    # =================================================================

    # Simple vocabulary: just a few characters for demonstration
    chars = ['a', 'b', 'c', 'd', 'e']
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}
    vocab_size = len(chars)

    # Hyperparameters
    hidden_size = 32
    num_epochs = 100
    learning_rate = 0.01
    seq_length = 4

    # =================================================================
    # MODEL: RNN + output layer
    # =================================================================

    class CharPredictor(nn.Module):
        """
        A character-level language model.

        Given a sequence of characters, predict the next character.
        """
        def __init__(self, vocab_size, hidden_size):
            super().__init__()
            # Embedding: convert character indices to vectors
            self.embedding = nn.Embedding(vocab_size, hidden_size)

            # RNN: process the sequence
            self.rnn = VanillaRNN(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=1
            )

            # Output layer: convert hidden state to character probabilities
            self.output_layer = nn.Linear(hidden_size, vocab_size)

        def forward(self, x):
            # x shape: (batch_size, seq_length) - character indices

            # Embed characters: (batch_size, seq_length, hidden_size)
            embedded = self.embedding(x)

            # Run through RNN: (batch_size, seq_length, hidden_size)
            rnn_out, _ = self.rnn(embedded)

            # Get output for each position: (batch_size, seq_length, vocab_size)
            logits = self.output_layer(rnn_out)

            return logits

    model = CharPredictor(vocab_size, hidden_size).to(device)

    # =================================================================
    # TRAINING DATA: Simple pattern "abcde" repeating
    # =================================================================

    # Create training sequence: "abcdeabcdeabcde..."
    # The model should learn: a->b, b->c, c->d, d->e, e->a
    pattern = "abcde" * 20  # Repeat pattern
    data = torch.tensor([char_to_idx[c] for c in pattern], device=device)

    # =================================================================
    # TRAINING LOOP
    # =================================================================

    # Loss function: cross-entropy for classification
    loss_fn = nn.CrossEntropyLoss()

    # Optimizer: Adam adjusts weights based on gradients
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\nTraining to predict next character in pattern: 'abcde'")
    print(f"Model should learn: a→b, b→c, c→d, d→e, e→a")
    print("-" * 60)

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        # Slide a window over the data
        for i in range(0, len(data) - seq_length - 1, seq_length):
            # Input: characters at positions [i, i+seq_length)
            x = data[i:i+seq_length].unsqueeze(0)  # Add batch dimension

            # Target: characters at positions [i+1, i+seq_length+1)
            # (shifted by 1 - we predict the NEXT character)
            y = data[i+1:i+seq_length+1].unsqueeze(0)

            # ---------------------------------------------------------
            # FORWARD PASS
            # ---------------------------------------------------------
            # PyTorch builds the computation graph here
            logits = model(x)  # (batch_size, seq_length, vocab_size)

            # Reshape for loss: (batch_size * seq_length, vocab_size)
            logits_flat = logits.view(-1, vocab_size)
            y_flat = y.view(-1)

            # ---------------------------------------------------------
            # COMPUTE LOSS
            # ---------------------------------------------------------
            loss = loss_fn(logits_flat, y_flat)

            # ---------------------------------------------------------
            # BACKWARD PASS (autograd does this!)
            # ---------------------------------------------------------
            optimizer.zero_grad()  # Clear old gradients
            loss.backward()        # Compute gradients via chain rule
                                   # After this: model.rnn.cells[0].W_hh.weight.grad exists!

            # ---------------------------------------------------------
            # UPDATE WEIGHTS
            # ---------------------------------------------------------
            optimizer.step()       # W = W - lr * grad

            total_loss += loss.item()
            num_batches += 1

        # Print progress
        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {avg_loss:.4f}")

    # =================================================================
    # TEST THE TRAINED MODEL
    # =================================================================

    print("-" * 60)
    print("Testing trained model:")
    print()

    model.eval()  # Set to evaluation mode
    with torch.no_grad():  # No need for gradients during testing
        for start_char in ['a', 'b', 'c', 'd', 'e']:
            # Start with one character
            current_idx = char_to_idx[start_char]
            generated = [start_char]

            # Hidden state starts at zero
            h = None

            # Generate 5 characters
            for _ in range(5):
                # Prepare input
                x = torch.tensor([[current_idx]], device=device)
                embedded = model.embedding(x)

                # Run one step of RNN
                rnn_out, h = model.rnn(embedded, h)

                # Get prediction
                logits = model.output_layer(rnn_out[:, -1, :])
                predicted_idx = logits.argmax(dim=-1).item()

                # Add to generated sequence
                generated.append(idx_to_char[predicted_idx])
                current_idx = predicted_idx

            print(f"  Start: '{start_char}' → Generated: {''.join(generated)}")

    print()
    print("Expected: a→b→c→d→e→a, b→c→d→e→a→b, etc.")
    print("=" * 60)


def demonstrate_sequential_bottleneck():
    """
    Demonstrate that RNN computation time scales linearly with sequence length.

    This is because we MUST process each timestep sequentially.
    A Transformer, in contrast, has O(1) depth regardless of sequence length
    (though O(n^2) memory for attention).
    """
    print("=" * 60)
    print("DEMONSTRATING THE SEQUENTIAL BOTTLENECK")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create RNN
    input_size = 128
    hidden_size = 256
    rnn = VanillaRNN(input_size, hidden_size).to(device)

    # Test with different sequence lengths
    seq_lengths = [32, 64, 128, 256, 512]
    batch_size = 32

    print(f"\nBatch size: {batch_size}, Input size: {input_size}, Hidden size: {hidden_size}")
    print("-" * 60)

    for seq_len in seq_lengths:
        x = torch.randn(batch_size, seq_len, input_size, device=device)

        # Warm up
        with torch.no_grad():
            _ = rnn(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Time the forward pass
        start = time.perf_counter()
        n_iterations = 10
        with torch.no_grad():
            for _ in range(n_iterations):
                _ = rnn(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = (time.perf_counter() - start) / n_iterations * 1000

        print(f"Seq length: {seq_len:4d} | Time: {elapsed:8.3f} ms | "
              f"Time per step: {elapsed/seq_len:.3f} ms")

    print("-" * 60)
    print("Notice: Time increases ~linearly with sequence length!")
    print("This is the sequential bottleneck - we can't parallelize across time.")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_training()
    print("\n\n")
    demonstrate_sequential_bottleneck()

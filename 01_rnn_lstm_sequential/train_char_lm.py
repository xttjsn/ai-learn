"""
Character-Level Language Model Training

This script trains an RNN/LSTM to predict the next character in a sequence.
It's a classic task that demonstrates:
1. How recurrent models learn sequential patterns
2. The training dynamics of RNNs vs LSTMs
3. Sampling from a trained language model

THE TASK:
=========
Given a sequence of characters "hello worl", predict the next character "d".

This requires the model to:
- Remember context from earlier in the sequence
- Learn patterns like "wor" -> "l" -> "d"
- Handle long-range dependencies

TRAINING MECHANICS:
==================
1. Embed each character into a vector
2. Process the sequence through RNN/LSTM
3. At each position, predict the next character
4. Compute cross-entropy loss
5. Backpropagate through time (BPTT)

The sequential bottleneck affects both:
- Forward pass: O(T) sequential operations
- Backward pass: O(T) sequential operations for BPTT

Author: Learning project
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import argparse
from typing import Optional, Tuple
import math

from rnn_from_scratch import VanillaRNN
from lstm_from_scratch import LSTM


# Sample text for training (you can replace with a larger corpus)
SAMPLE_TEXT = """
The quick brown fox jumps over the lazy dog.
Machine learning is a subset of artificial intelligence.
Neural networks learn patterns from data through training.
Recurrent neural networks process sequences one step at a time.
The hidden state carries information from previous timesteps.
Long short-term memory networks use gates to control information flow.
The forget gate decides what information to discard from the cell state.
The input gate decides what new information to store in the cell state.
The output gate decides what information to output from the cell state.
Transformers replaced recurrent models with self-attention mechanisms.
Self-attention allows processing all positions in parallel.
This eliminates the sequential bottleneck of recurrent models.
Attention is all you need was a groundbreaking paper in 2017.
The key insight was that attention alone is sufficient for sequence modeling.
Modern large language models are based on the Transformer architecture.
They can process thousands of tokens in parallel on GPUs.
This parallelism enables training on massive datasets efficiently.
Understanding the sequential bottleneck helps appreciate this advance.
"""


class CharDataset(Dataset):
    """
    Dataset for character-level language modeling.

    Each sample is:
    - Input: sequence of characters [c_0, c_1, ..., c_{T-1}]
    - Target: next characters [c_1, c_2, ..., c_T]

    This means for each position, the model predicts the next character.
    """

    def __init__(self, text: str, seq_length: int):
        self.seq_length = seq_length

        # Create character vocabulary
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

        # Encode entire text
        self.data = torch.tensor([self.char_to_idx[ch] for ch in text], dtype=torch.long)

        # Number of samples
        self.n_samples = len(self.data) - seq_length

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Input: characters at positions [idx, idx+seq_length)
        # Target: characters at positions [idx+1, idx+seq_length+1)
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]
        return x, y


class CharLanguageModel(nn.Module):
    """
    Character-level language model using RNN or LSTM.

    Architecture:
    1. Embedding: char index -> dense vector
    2. RNN/LSTM: process sequence, produce hidden states
    3. Output layer: hidden state -> vocabulary logits
    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        model_type: str = "lstm"
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.model_type = model_type

        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # Recurrent layer
        if model_type == "rnn":
            self.rnn = VanillaRNN(embed_size, hidden_size, num_layers)
        elif model_type == "lstm":
            self.rnn = LSTM(embed_size, hidden_size, num_layers)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Output projection
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input characters, shape (batch_size, seq_len)
            hidden: Optional initial hidden state

        Returns:
            logits: Output logits, shape (batch_size, seq_len, vocab_size)
            hidden: Final hidden state
        """
        # Embed characters
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_size)

        # Process through RNN/LSTM
        # THIS IS WHERE THE SEQUENTIAL BOTTLENECK HAPPENS
        output, hidden = self.rnn(embedded, hidden)

        # Project to vocabulary
        logits = self.output(output)  # (batch_size, seq_len, vocab_size)

        return logits, hidden

    def generate(
        self,
        start_text: str,
        char_to_idx: dict,
        idx_to_char: dict,
        max_length: int = 100,
        temperature: float = 1.0,
        device: str = "cpu"
    ) -> str:
        """
        Generate text character by character.

        This demonstrates the sequential nature even at inference time:
        we must generate one character at a time, feeding each back in.
        """
        self.eval()

        # Encode start text
        chars = [char_to_idx.get(ch, 0) for ch in start_text]
        x = torch.tensor([chars], device=device)

        # Process start text
        with torch.no_grad():
            logits, hidden = self(x)

        # Generate character by character
        generated = list(start_text)

        for _ in range(max_length):
            # Get prediction for next character
            last_logits = logits[0, -1, :] / temperature

            # Sample from distribution
            probs = torch.softmax(last_logits, dim=0)
            next_idx = torch.multinomial(probs, 1).item()
            next_char = idx_to_char[next_idx]
            generated.append(next_char)

            # Feed back in for next prediction
            # Note: This is sequential - we can't parallelize generation!
            x = torch.tensor([[next_idx]], device=device)
            logits, hidden = self(x, hidden)

        return "".join(generated)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_tokens = 0

    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Forward pass
        logits, _ = model(x)

        # Compute loss (flatten for cross-entropy)
        loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (important for RNNs!)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def main():
    parser = argparse.ArgumentParser(description="Train character-level language model")
    parser.add_argument("--model", type=str, default="lstm", choices=["rnn", "lstm"])
    parser.add_argument("--embed-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seq-length", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--generate", action="store_true", help="Generate text after training")
    args = parser.parse_args()

    print("=" * 60)
    print(f"CHARACTER-LEVEL LANGUAGE MODEL ({args.model.upper()})")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataset
    dataset = CharDataset(SAMPLE_TEXT, args.seq_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print(f"\nDataset:")
    print(f"  Vocabulary size: {dataset.vocab_size}")
    print(f"  Sequence length: {args.seq_length}")
    print(f"  Number of samples: {len(dataset)}")

    # Create model
    model = CharLanguageModel(
        vocab_size=dataset.vocab_size,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        model_type=args.model
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel:")
    print(f"  Type: {args.model.upper()}")
    print(f"  Embedding size: {args.embed_size}")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Number of layers: {args.num_layers}")
    print(f"  Total parameters: {n_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"\nTraining:")
    print("-" * 60)

    # Training loop
    for epoch in range(args.epochs):
        start_time = time.time()

        loss, perplexity = train_epoch(model, dataloader, optimizer, criterion, device)

        elapsed = time.time() - start_time

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d} | Loss: {loss:.4f} | "
                  f"Perplexity: {perplexity:.2f} | Time: {elapsed:.2f}s")

    print("-" * 60)

    # Generate some text
    if args.generate:
        print("\nGenerating text:")
        print("-" * 60)

        prompts = ["The ", "Neural ", "Machine "]
        for prompt in prompts:
            generated = model.generate(
                prompt,
                dataset.char_to_idx,
                dataset.idx_to_char,
                max_length=100,
                temperature=0.8,
                device=device
            )
            print(f"'{prompt}' -> {generated[:100]}...")
            print()

    print("=" * 60)


if __name__ == "__main__":
    main()

"""
LSTM (Long Short-Term Memory) Implementation from Scratch

This module implements an LSTM cell and layer WITHOUT using PyTorch's built-in
nn.LSTM. The goal is to understand:
1. Why LSTMs were invented (vanishing gradient problem)
2. How gating mechanisms work
3. Why LSTMs STILL have the sequential bottleneck

================================================================================
PREREQUISITE: THE CHAIN RULE (CALCULUS)
================================================================================
The chain rule tells us how to find derivatives of nested functions:

    If y = f(g(x)), then:

        dy/dx = dy/dg * dg/dx

When a variable affects the output through MULTIPLE paths, we SUM the contributions:

    If z = f(x, y) where both x and y depend on t:

        dz/dt = (dz/dx * dx/dt) + (dz/dy * dy/dt)

================================================================================
PREREQUISITE: BACKPROPAGATION (CHAIN RULE APPLIED TO NEURAL NETWORKS)
================================================================================
Backpropagation uses the chain rule to compute gradients through a network.

For a simple network: input -> layer1 -> layer2 -> loss

    dLoss/d(layer1_weights) = dLoss/d(layer2_output)
                             * d(layer2_output)/d(layer1_output)
                             * d(layer1_output)/d(layer1_weights)

For an RNN over T timesteps, gradients flow backward through time:

    dLoss/dh_1 = dLoss/dh_T * dh_T/dh_{T-1} * dh_{T-1}/dh_{T-2} * ... * dh_2/dh_1

               = dLoss/dh_T * PRODUCT_{t=2}^{T} (dh_t/dh_{t-1})

The PRODUCT is the problem - if each term is < 1, the product vanishes.
If each term is > 1, the product explodes.

================================================================================
VANILLA RNN FORMULA (for reference)
================================================================================
    h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b_h)

The derivative for backpropagation:

    dh_t/dh_{t-1} = W_hh^T * diag(1 - tanh^2(z))

    where z = W_hh @ h_{t-1} + W_xh @ x_t + b_h

================================================================================
THE VANISHING GRADIENT PROBLEM (in Vanilla RNNs)
================================================================================
Over many timesteps, the gradient becomes:

    dh_T/dh_1 = PRODUCT_{t=2}^{T} [W_hh^T * diag(1 - tanh^2(z_t))]

Problems:
    - tanh'(z) = 1 - tanh^2(z) is always in range (0, 1]
    - If ||W_hh|| < 1: gradients vanish exponentially (0.9^50 ≈ 0.005)
    - If ||W_hh|| > 1: gradients explode exponentially (1.1^50 ≈ 117)

Example: After 50 timesteps with ||W_hh|| = 0.9:

    gradient ≈ 0.9^50 ≈ 0.005  (nearly zero - can't learn!)

================================================================================
LSTM FORMULA (the solution)
================================================================================
LSTM introduces a CELL STATE c_t that uses ADDITION instead of just multiplication:

Gates (all use sigmoid, output values between 0 and 1):
    f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)    # Forget gate: what to erase
    i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i)    # Input gate: how much to write
    o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o)    # Output gate: what to expose

Cell candidate (uses tanh, output values between -1 and 1):
    g_t = tanh(W_g @ [h_{t-1}, x_t] + b_g)       # New info to potentially add

Cell state update (THE KEY EQUATION - uses ADDITION):
    c_t = f_t * c_{t-1} + i_t * g_t
          └────┬────┘    └────┬────┘
          keep some of   add some
          old memory     new info

Hidden state output:
    h_t = o_t * tanh(c_t)

================================================================================
WHY LSTM FIXES VANISHING GRADIENTS
================================================================================
The gradient through the cell state is:

    dc_t/dc_{t-1} = f_t    (just the forget gate value!)

Over T timesteps:

    dc_T/dc_1 = PRODUCT_{t=2}^{T} f_t = f_T * f_{T-1} * ... * f_2

Why this is better:
    1. f_t is bounded in (0, 1) - no explosion possible
    2. f_t is typically close to 1 (network learns to remember)
    3. NO MATRIX MULTIPLICATION - just scalar multiplication

Example: After 50 timesteps with f ≈ 0.9:

    gradient ≈ 0.9^50 ≈ 0.005  (small but usable!)

Compare to vanilla RNN:

    gradient ≈ (W * tanh')^50 ≈ 0.0000...  (effectively zero)

The ADDITION in "c_t = f_t * c_{t-1} + i_t * g_t" creates a gradient highway!
Gradients flow through the + without repeated matrix multiplication.

================================================================================
HOW GRADIENTS FLOW TO ALL GATE WEIGHTS (W_f, W_i, W_g, W_o)
================================================================================
You might wonder: we showed dc_t/dc_{t-1} = f_t, but what about the weights
for the input gate (W_i), cell candidate (W_g), and output gate (W_o)?
Don't they have vanishing gradient problems?

The key insight: ALL gate weights receive gradients through the CELL STATE HIGHWAY.

Let's trace the gradient for W_i (input gate weights) at timestep t=5:

    dLoss/dW_i = dLoss/dc_T * dc_T/dc_5 * dc_5/di_5 * di_5/dW_i
                 └────┬────┘   └───┬───┘   └───┬───┘   └───┬───┘
                 from loss    cell state    local       local
                              highway       derivative  derivative

Breaking it down:

1. dLoss/dc_T: Gradient at the final cell state (from the loss)

2. dc_T/dc_5: Gradient flowing BACK through the cell state highway

   dc_T/dc_5 = f_T * f_{T-1} * ... * f_6  (product of forget gates)

   This is the "highway" - it doesn't vanish because f ≈ 0.9

3. dc_5/di_5: Local derivative at timestep 5

   c_5 = f_5 * c_4 + i_5 * g_5
   dc_5/di_5 = g_5  (just the cell candidate value!)

4. di_5/dW_i: Derivative of sigmoid through the weights

   i_5 = sigmoid(W_i @ [h_4, x_5])
   di_5/dW_i = sigmoid'(...) * [h_4, x_5]

The same pattern applies to ALL gate weights:

    ┌─────────────────────────────────────────────────────────────────┐
    │                     CELL STATE HIGHWAY                          │
    │  c_1 ───[×f]───> c_2 ───[×f]───> c_3 ───[×f]───> ... ───> c_T  │
    │         ↑               ↑               ↑                  ↓    │
    │         │               │               │              Loss     │
    └─────────┼───────────────┼───────────────┼──────────────────────┘
              │               │               │
              │               │               │
         ┌────┴────┐     ┌────┴────┐     ┌────┴────┐
         │ Gates   │     │ Gates   │     │ Gates   │
         │ f,i,g,o │     │ f,i,g,o │     │ f,i,g,o │
         │    ↑    │     │    ↑    │     │    ↑    │
         │ W_f,W_i │     │ W_f,W_i │     │ W_f,W_i │
         │ W_g,W_o │     │ W_g,W_o │     │ W_g,W_o │
         └─────────┘     └─────────┘     └─────────┘

    Gradients flow DOWN from Loss through the highway,
    then INTO each gate's weights at each timestep.

WHY GATE WEIGHTS DON'T HAVE VANISHING GRADIENTS:
------------------------------------------------
1. The cell state highway DELIVERS healthy gradients to each timestep
2. Gate weights only need LOCAL derivatives (one step, not T steps)
3. The long-range dependency is handled by the highway, not by the gates

Example gradient calculation for W_i at timestep 5 (with T=50):

    dLoss/dW_i = dLoss/dc_50 * (f_50 * f_49 * ... * f_6) * g_5 * sigmoid'(...) * [h_4, x_5]
                              └──────────────┬──────────┘
                                 ≈ 0.9^45 ≈ 0.01 (small but NOT zero!)

Compare to vanilla RNN:

    dLoss/dW at timestep 5 = (W * tanh')^45 * local_terms
                            └──────┬──────┘
                              ≈ 0 (vanished!)

The highway keeps gradients alive so they can reach all gate weights.

GRADIENT FORMULAS FOR EACH GATE:
--------------------------------
Given: c_t = f_t * c_{t-1} + i_t * g_t
       h_t = o_t * tanh(c_t)

For forget gate (W_f):
    dc_t/df_t = c_{t-1}
    dLoss/dW_f = dLoss/dc_t * c_{t-1} * sigmoid'(W_f @ [...]) * [h_{t-1}, x_t]

For input gate (W_i):
    dc_t/di_t = g_t
    dLoss/dW_i = dLoss/dc_t * g_t * sigmoid'(W_i @ [...]) * [h_{t-1}, x_t]

For cell candidate (W_g):
    dc_t/dg_t = i_t
    dLoss/dW_g = dLoss/dc_t * i_t * tanh'(W_g @ [...]) * [h_{t-1}, x_t]

For output gate (W_o):
    dh_t/do_t = tanh(c_t)
    dLoss/dW_o = dLoss/dh_t * tanh(c_t) * sigmoid'(W_o @ [...]) * [h_{t-1}, x_t]

In ALL cases, dLoss/dc_t (or dLoss/dh_t) arrives via the highway with manageable magnitude!

================================================================================
THE GATES EXPLAINED
================================================================================
1. Forget gate (f_t):
   - Controls what to ERASE from memory
   - f_t ≈ 1: keep everything, f_t ≈ 0: forget everything

2. Input gate (i_t):
   - Controls HOW MUCH new information to write
   - i_t ≈ 1: write fully, i_t ≈ 0: ignore new input

3. Cell candidate (g_t):
   - WHAT new information is available (same formula as vanilla RNN!)
   - Range: -1 to 1

4. Output gate (o_t):
   - Controls what part of cell state to expose to next layer
   - o_t ≈ 1: output everything, o_t ≈ 0: output nothing

================================================================================
THE SEQUENTIAL BOTTLENECK REMAINS
================================================================================
Despite fixing vanishing gradients, LSTMs still compute:

    c_t = f_t * c_{t-1} + i_t * g_t
    h_t = o_t * tanh(c_t)

c_t depends on c_{t-1}, h_t depends on c_t and h_{t-1} (through gates).
We STILL cannot parallelize across time steps!

This is why Transformers (which use attention instead of recurrence)
were a major breakthrough - they CAN parallelize across positions.

Author: Learning project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import time


class LSTMCell(nn.Module):
    """
    A single LSTM cell that processes ONE timestep.

    The LSTM equations are:
        f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)    # Forget gate
        i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i)    # Input gate
        g_t = tanh(W_g @ [h_{t-1}, x_t] + b_g)       # Cell candidate
        o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o)    # Output gate

        c_t = f_t * c_{t-1} + i_t * g_t              # New cell state
        h_t = o_t * tanh(c_t)                        # New hidden state

    Note the dependencies:
    - All gates depend on h_{t-1}
    - c_t depends on c_{t-1}
    - h_t depends on c_t

    This means we STILL have a sequential bottleneck!
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # For efficiency, we compute all 4 gates in one matrix multiplication
        # W_combined has shape (4 * hidden_size, input_size + hidden_size)
        # The 4 corresponds to: forget, input, cell candidate, output
        self.W_combined = nn.Linear(
            input_size + hidden_size,
            4 * hidden_size,
            bias=True
        )

        self._init_weights()

    def _init_weights(self):
        """
        LSTM-specific initialization.

        Key insight: Initialize forget gate bias to 1.0!
        This means the LSTM starts by remembering everything,
        which helps gradient flow at the beginning of training.
        """
        nn.init.xavier_uniform_(self.W_combined.weight)
        nn.init.zeros_(self.W_combined.bias)

        # Set forget gate bias to 1.0 (important!)
        # The forget gate is the first of the 4 gates
        hidden_size = self.hidden_size
        self.W_combined.bias.data[0:hidden_size] = 1.0

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single timestep.

        Args:
            x_t: Input at time t, shape (batch_size, input_size)
            h_prev: Hidden state from time t-1, shape (batch_size, hidden_size)
            c_prev: Cell state from time t-1, shape (batch_size, hidden_size)

        Returns:
            h_t: New hidden state, shape (batch_size, hidden_size)
            c_t: New cell state, shape (batch_size, hidden_size)
        """
        # Concatenate input and previous hidden state
        combined = torch.cat([x_t, h_prev], dim=1)

        # Compute all gates in one go
        gates = self.W_combined(combined)

        # Split into individual gates
        # Each gate has shape (batch_size, hidden_size)
        f_gate, i_gate, g_gate, o_gate = gates.chunk(4, dim=1)

        # Apply activations
        f_t = torch.sigmoid(f_gate)  # Forget gate: what to forget
        i_t = torch.sigmoid(i_gate)  # Input gate: what to add
        g_t = torch.tanh(g_gate)     # Cell candidate: new values
        o_t = torch.sigmoid(o_gate)  # Output gate: what to output

        # ===================================================================
        # THE CELL STATE UPDATE - This is the "constant error carousel"
        # ===================================================================
        # c_t = f_t * c_prev + i_t * g_t
        #
        # When f_t ≈ 1 and i_t ≈ 0:
        #   c_t ≈ c_prev
        #   dc_t/dc_prev ≈ 1
        #
        # This allows gradients to flow unchanged through many timesteps!
        # But note: c_t STILL depends on c_prev. Sequential bottleneck remains.
        # ===================================================================
        c_t = f_t * c_prev + i_t * g_t

        # Hidden state (what we output to next layer / final output)
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


class LSTM(nn.Module):
    """
    Full LSTM layer that processes an entire sequence.

    Just like VanillaRNN, we MUST use a for-loop over time steps.
    The LSTM solved the vanishing gradient problem but NOT the sequential
    bottleneck. This is why Transformers were a breakthrough.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Stack of LSTM cells
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            cell_input_size = input_size if layer == 0 else hidden_size
            self.cells.append(LSTMCell(cell_input_size, hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process an entire sequence.

        Args:
            x: Input sequence, shape (batch_size, seq_len, input_size)
            initial_state: Tuple of (h_0, c_0), each with shape
                          (num_layers, batch_size, hidden_size)

        Returns:
            output: All hidden states, shape (batch_size, seq_len, hidden_size)
            (h_n, c_n): Final states, each with shape (num_layers, batch_size, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Initialize states if not provided
        if initial_state is None:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        else:
            h_0, c_0 = initial_state

        # Current states for each layer
        h = [h_0[layer] for layer in range(self.num_layers)]
        c = [c_0[layer] for layer in range(self.num_layers)]

        # Collect outputs
        outputs = []

        # =================================================================
        # THE SEQUENTIAL BOTTLENECK - SAME AS VANILLA RNN
        # =================================================================
        # Despite all the fancy gating, we STILL must process timesteps
        # one by one. There is no escaping this loop in an LSTM.
        # =================================================================
        for t in range(seq_len):
            x_t = x[:, t, :]

            layer_input = x_t
            for layer, cell in enumerate(self.cells):
                h[layer], c[layer] = cell(layer_input, h[layer], c[layer])
                layer_input = h[layer]

            outputs.append(h[-1])

        output = torch.stack(outputs, dim=1)
        h_n = torch.stack(h, dim=0)
        c_n = torch.stack(c, dim=0)

        return output, (h_n, c_n)


def visualize_gates():
    """
    Visualize what the LSTM gates are doing during a forward pass.
    """
    print("=" * 60)
    print("VISUALIZING LSTM GATE ACTIVATIONS")
    print("=" * 60)

    # Create a simple LSTM cell
    cell = LSTMCell(input_size=4, hidden_size=8)

    # Create some input
    batch_size = 1
    x_t = torch.randn(batch_size, 4)
    h_prev = torch.zeros(batch_size, 8)
    c_prev = torch.zeros(batch_size, 8)

    # Manual forward pass to see gate values
    combined = torch.cat([x_t, h_prev], dim=1)
    gates = cell.W_combined(combined)
    f_gate, i_gate, g_gate, o_gate = gates.chunk(4, dim=1)

    f_t = torch.sigmoid(f_gate)
    i_t = torch.sigmoid(i_gate)
    g_t = torch.tanh(g_gate)
    o_t = torch.sigmoid(o_gate)

    print("\nGate activations (showing first 4 dimensions):")
    print(f"  Forget gate (f_t): {f_t[0, :4].detach().numpy().round(3)}")
    print(f"    -> Values close to 1 mean 'remember', close to 0 mean 'forget'")
    print(f"  Input gate (i_t):  {i_t[0, :4].detach().numpy().round(3)}")
    print(f"    -> Values close to 1 mean 'add new info', close to 0 mean 'ignore'")
    print(f"  Cell candidate:    {g_t[0, :4].detach().numpy().round(3)}")
    print(f"    -> The new values to potentially add (range -1 to 1)")
    print(f"  Output gate (o_t): {o_t[0, :4].detach().numpy().round(3)}")
    print(f"    -> What fraction of cell state to output")

    # Show the cell state update
    c_t = f_t * c_prev + i_t * g_t
    h_t = o_t * torch.tanh(c_t)

    print(f"\n  c_prev:            {c_prev[0, :4].detach().numpy().round(3)}")
    print(f"  c_t (new cell):    {c_t[0, :4].detach().numpy().round(3)}")
    print(f"  h_t (output):      {h_t[0, :4].detach().numpy().round(3)}")
    print("=" * 60)


def compare_rnn_vs_lstm_gradient_flow():
    """
    Demonstrate that LSTM gradients flow better than vanilla RNN gradients.

    We'll look at how gradients at early timesteps compare to late timesteps.
    """
    print("=" * 60)
    print("COMPARING GRADIENT FLOW: RNN vs LSTM")
    print("=" * 60)

    from rnn_from_scratch import VanillaRNN

    # Parameters
    input_size = 16
    hidden_size = 32
    seq_len = 50
    batch_size = 8

    # Create models
    rnn = VanillaRNN(input_size, hidden_size)
    lstm = LSTM(input_size, hidden_size)

    # Random input
    x = torch.randn(batch_size, seq_len, input_size, requires_grad=False)

    # Make input require grad so we can measure gradients
    x_rnn = x.clone().detach().requires_grad_(True)
    x_lstm = x.clone().detach().requires_grad_(True)

    # Forward pass
    output_rnn, _ = rnn(x_rnn)
    output_lstm, _ = lstm(x_lstm)

    # Create a loss that only depends on the final timestep
    # This forces gradients to flow all the way back through time
    loss_rnn = output_rnn[:, -1, :].sum()
    loss_lstm = output_lstm[:, -1, :].sum()

    # Backward pass
    loss_rnn.backward()
    loss_lstm.backward()

    # Look at gradient magnitudes at different timesteps
    print("\nGradient magnitude at input x at different timesteps:")
    print("(Loss computed only at final timestep, so gradients must flow back)")
    print("-" * 60)
    print(f"{'Timestep':<12} {'RNN Grad Norm':<18} {'LSTM Grad Norm':<18}")
    print("-" * 60)

    for t in [0, 10, 20, 30, 40, 49]:
        rnn_grad_norm = x_rnn.grad[:, t, :].norm().item()
        lstm_grad_norm = x_lstm.grad[:, t, :].norm().item()
        print(f"{t:<12} {rnn_grad_norm:<18.6f} {lstm_grad_norm:<18.6f}")

    print("-" * 60)
    print("Notice: RNN gradients at early timesteps are much smaller!")
    print("This is the vanishing gradient problem.")
    print("LSTM gradients are more uniform across timesteps.")
    print("=" * 60)


if __name__ == "__main__":
    visualize_gates()
    print("\n")
    compare_rnn_vs_lstm_gradient_flow()

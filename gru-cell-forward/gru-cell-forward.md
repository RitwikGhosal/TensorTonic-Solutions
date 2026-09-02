## The Problem with Vanilla RNNs

A vanilla RNN processes sequences by maintaining a hidden state that gets updated at each time step:

$$
h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b)
$$

This simple update has a critical flaw: **vanishing gradients**. When backpropagating through many time steps, gradients shrink exponentially because they pass through the tanh derivative (max value 1) repeatedly. After 50-100 steps, the gradient is essentially zero, and the network cannot learn long-range dependencies.

The GRU (Gated Recurrent Unit) solves this by introducing **gates** that control information flow.

---

## The Core Idea: Gating

A gate is a vector of values between 0 and 1 (produced by a sigmoid). When you multiply a signal by a gate:

- Gate value near 0: block the signal (multiply by ~0)
- Gate value near 1: pass the signal through (multiply by ~1)
- Gate value of 0.5: pass half the signal

Gates let the network learn **when** to update its memory and **when** to preserve it unchanged. This creates shortcuts for gradients to flow backward without shrinking.

---

## GRU Architecture Overview

The GRU has two gates:

1. **Update gate ($z_t$)**: decides how much of the old hidden state to keep vs. replace with new information
2. **Reset gate ($r_t$)**: decides how much of the old hidden state to use when computing the new candidate

And one intermediate value:

3. **Candidate hidden state ($\tilde{h}_t$)**: the proposed new hidden state, computed using the reset gate

The final hidden state is a blend of the old state and the candidate, controlled by the update gate.

---

## Step-by-Step Computation

**Input at time step $t$:**
- $x_t$: input vector (shape: input_size)
- $h_{t-1}$: previous hidden state (shape: hidden_size)

**Step 1: Compute the update gate**

$$
z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)
$$

- $W_z$: weight matrix for input (shape: hidden_size x input_size)
- $U_z$: weight matrix for hidden state (shape: hidden_size x hidden_size)
- $b_z$: bias vector (shape: hidden_size)
- $\sigma$: sigmoid function, outputs values in $(0, 1)$

The update gate decides: "How much should I update my hidden state with new information?"

**Step 2: Compute the reset gate**

$$
r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)
$$

Same structure as the update gate, but with different learned parameters.

The reset gate decides: "How much of the previous hidden state should I consider when computing the new candidate?"

**Step 3: Compute the candidate hidden state**

$$
\tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h)
$$

- $\odot$ denotes element-wise multiplication
- $r_t \odot h_{t-1}$: the reset gate filters the previous hidden state

When $r_t$ is close to 0, the previous hidden state is ignored, and the candidate is computed mostly from the current input. When $r_t$ is close to 1, the full previous hidden state is used.

**Step 4: Compute the final hidden state**

$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

This is a **linear interpolation** between the old hidden state and the candidate:
- When $z_t$ is close to 0: $h_t \approx h_{t-1}$ (keep the old state)
- When $z_t$ is close to 1: $h_t \approx \tilde{h}_t$ (use the new candidate)
- Values in between: blend of old and new

---

## Why This Solves Vanishing Gradients

The key is the update equation:

$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

When $z_t$ is close to 0, we have $h_t \approx h_{t-1}$. This means the gradient flows directly from $h_t$ to $h_{t-1}$ with a multiplier close to 1. No tanh, no shrinking.

The network can learn to keep $z_t$ small for time steps where nothing important happens, creating a "gradient highway" that lets information flow backward through many steps without vanishing.

---

## A Concrete Example

Suppose we have:
- Input size: 3
- Hidden size: 2
- $x_t = [1.0, 0.5, -0.5]$
- $h_{t-1} = [0.8, -0.3]$

After computing with learned weights:

**Update gate:** $z_t = [0.7, 0.2]$
- First hidden unit will mostly update (0.7)
- Second hidden unit will mostly keep old value (0.2)

**Reset gate:** $r_t = [0.9, 0.1]$
- First unit uses most of previous hidden state
- Second unit ignores most of previous hidden state

**Candidate:** $\tilde{h}_t = [0.5, 0.9]$

**Final hidden state:**
- $h_t[0] = (1 - 0.7) \times 0.8 + 0.7 \times 0.5 = 0.24 + 0.35 = 0.59$
- $h_t[1] = (1 - 0.2) \times (-0.3) + 0.2 \times 0.9 = -0.24 + 0.18 = -0.06$

Result: $h_t = [0.59, -0.06]$

---

## GRU vs. LSTM

Both GRU and LSTM solve the vanishing gradient problem with gating. The differences:

**LSTM:**
- Three gates: input, forget, output
- Separate cell state and hidden state
- More parameters
- Slightly more expressive

**GRU:**
- Two gates: update, reset
- Single hidden state (no separate cell state)
- Fewer parameters (about 2/3 of LSTM)
- Often performs comparably to LSTM
- Faster to train due to fewer computations

GRU can be seen as a simplified LSTM that merges the cell state and hidden state, and combines the input and forget gates into a single update gate.

---

## Parameter Shapes

For a GRU with input size $d$ and hidden size $h$:

**Weight matrices (6 total):**
- $W_z, W_r, W_h$: each has shape $(h, d)$
- $U_z, U_r, U_h$: each has shape $(h, h)$

**Bias vectors (3 total):**
- $b_z, b_r, b_h$: each has shape $(h,)$

**Total parameters:** $3 \times (h \times d) + 3 \times (h \times h) + 3 \times h = 3h(d + h + 1)$

---

## Common Implementation Notes

**Concatenated weights:** Many implementations concatenate $W$ and $U$ into a single matrix and concatenate $x_t$ and $h_{t-1}$ into a single vector. This allows one matrix multiplication instead of two:

$$
[z_t; r_t] = \sigma([W_z; W_r] \cdot [x_t; h_{t-1}] + [b_z; b_r])
$$

**Batch processing:** In practice, inputs are batched, so $x_t$ has shape (batch_size, input_size) and all operations are vectorized across the batch dimension.

**Bidirectional GRU:** Process the sequence in both directions and concatenate the hidden states, capturing both past and future context.
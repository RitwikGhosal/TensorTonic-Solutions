## RNN Forward Pass Recap

Before understanding the backward pass, recall the forward pass of a vanilla RNN cell:

$$
h_t = \tanh(W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h)
$$

where:
- $x_t$ is the input at time step $t$ (shape: input_size)
- $h_{t-1}$ is the hidden state from the previous time step (shape: hidden_size)
- $h_t$ is the new hidden state (shape: hidden_size)
- $W_{xh}$ is the input-to-hidden weight matrix (shape: hidden_size x input_size)
- $W_{hh}$ is the hidden-to-hidden weight matrix (shape: hidden_size x hidden_size)
- $b_h$ is the bias vector (shape: hidden_size)

The computation can be broken into steps:
1. Compute the linear combination: $z_t = W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h$
2. Apply the activation: $h_t = \tanh(z_t)$

---

## What the Backward Pass Computes

Given the gradient of the loss with respect to $h_t$ (denoted $\frac{\partial L}{\partial h_t}$ or $dh_t$), the backward pass computes:

1. **Gradient w.r.t. the input:** $\frac{\partial L}{\partial x_t}$
2. **Gradient w.r.t. the previous hidden state:** $\frac{\partial L}{\partial h_{t-1}}$
3. **Gradient w.r.t. the weights:** $\frac{\partial L}{\partial W_{xh}}$, $\frac{\partial L}{\partial W_{hh}}$
4. **Gradient w.r.t. the bias:** $\frac{\partial L}{\partial b_h}$

These gradients are used to:
- Update the weights during training
- Propagate gradients back to earlier time steps (backpropagation through time)

---

## The Chain Rule Through Tanh

The gradient flows backward through the tanh activation first.

**Tanh derivative:**

$$
\frac{d}{dz} \tanh(z) = 1 - \tanh^2(z)
$$

Since we already computed $h_t = \tanh(z_t)$ in the forward pass, the derivative is:

$$
\frac{d}{dz_t} \tanh(z_t) = 1 - h_t^2
$$

**Gradient w.r.t. $z_t$:**

$$
\frac{\partial L}{\partial z_t} = \frac{\partial L}{\partial h_t} \odot (1 - h_t^2)
$$

where $\odot$ is element-wise multiplication.

This is often denoted as $dz_t$ or $d_{raw}$.

---

## Gradients for Weights and Bias

Now that we have $\frac{\partial L}{\partial z_t}$, we can compute gradients for the parameters.

**Gradient w.r.t. $W_{xh}$:**

Since $z_t = W_{xh} \cdot x_t + ...$, the gradient is:

$$
\frac{\partial L}{\partial W_{xh}} = \frac{\partial L}{\partial z_t} \cdot x_t^T
$$

This is an outer product: (hidden_size, 1) times (1, input_size) gives (hidden_size, input_size).

**Gradient w.r.t. $W_{hh}$:**

Since $z_t = ... + W_{hh} \cdot h_{t-1} + ...$, the gradient is:

$$
\frac{\partial L}{\partial W_{hh}} = \frac{\partial L}{\partial z_t} \cdot h_{t-1}^T
$$

**Gradient w.r.t. $b_h$:**

Since $z_t = ... + b_h$, the gradient is simply:

$$
\frac{\partial L}{\partial b_h} = \frac{\partial L}{\partial z_t}
$$

The gradient equals $dz_t$ directly (summed over the batch if batched).

---

## Gradients for Inputs

**Gradient w.r.t. $x_t$:**

$$
\frac{\partial L}{\partial x_t} = W_{xh}^T \cdot \frac{\partial L}{\partial z_t}
$$

This propagates the gradient back to the input, which is needed if there are layers before the RNN.

**Gradient w.r.t. $h_{t-1}$:**

$$
\frac{\partial L}{\partial h_{t-1}} = W_{hh}^T \cdot \frac{\partial L}{\partial z_t}
$$

This is **critical** for backpropagation through time. It sends the gradient to the previous time step.

---

## Step-by-Step Backward Pass

Given: $dh_t = \frac{\partial L}{\partial h_t}$ (gradient from the loss or from the next time step)

**Step 1: Backprop through tanh**

$$
dz_t = dh_t \odot (1 - h_t^2)
$$

**Step 2: Compute weight gradients**

$$
dW_{xh} = dz_t \cdot x_t^T
$$

$$
dW_{hh} = dz_t \cdot h_{t-1}^T
$$

$$
db_h = dz_t
$$

**Step 3: Compute input gradients**

$$
dx_t = W_{xh}^T \cdot dz_t
$$

$$
dh_{t-1} = W_{hh}^T \cdot dz_t
$$

---

## A Numerical Example

Suppose:
- hidden_size = 2, input_size = 3
- $h_t = [0.8, -0.5]$ (from forward pass)
- $dh_t = [1.0, 0.5]$ (gradient from loss)
- $x_t = [1, 2, 3]$
- $h_{t-1} = [0.3, 0.7]$

**Step 1: Backprop through tanh**

$1 - h_t^2 = [1 - 0.64, 1 - 0.25] = [0.36, 0.75]$

$dz_t = [1.0 \times 0.36, 0.5 \times 0.75] = [0.36, 0.375]$

**Step 2: Weight gradients**

$dW_{xh} = dz_t \cdot x_t^T = \begin{bmatrix} 0.36 \\ 0.375 \end{bmatrix} \cdot \begin{bmatrix} 1 & 2 & 3 \end{bmatrix}$

$= \begin{bmatrix} 0.36 & 0.72 & 1.08 \\ 0.375 & 0.75 & 1.125 \end{bmatrix}$

$dW_{hh} = dz_t \cdot h_{t-1}^T = \begin{bmatrix} 0.36 \\ 0.375 \end{bmatrix} \cdot \begin{bmatrix} 0.3 & 0.7 \end{bmatrix}$

$= \begin{bmatrix} 0.108 & 0.252 \\ 0.1125 & 0.2625 \end{bmatrix}$

$db_h = [0.36, 0.375]$

**Step 3: Input gradients**

These depend on the actual values of $W_{xh}$ and $W_{hh}$.

---

## Backpropagation Through Time (BPTT)

For a sequence of length $T$, the backward pass proceeds from $t = T$ back to $t = 1$:

1. At $t = T$: receive $dh_T$ from the loss
2. Compute gradients for step $T$, get $dh_{T-1}$
3. At $t = T-1$: $dh_{T-1}$ might also receive gradient from any loss at that step
4. Continue back to $t = 1$

Weight gradients are **accumulated** across all time steps:

$$
dW_{xh} = \sum_{t=1}^{T} dW_{xh}^{(t)}
$$

The same weights are used at every time step, so their gradients add up.

---

## The Vanishing Gradient Problem

At each backward step, $dh_{t-1}$ is computed as:

$$
dh_{t-1} = W_{hh}^T \cdot (dh_t \odot (1 - h_t^2))
$$

The tanh derivative $(1 - h_t^2)$ is at most 1 (when $h_t = 0$) and often much smaller. Multiplying by this many times causes the gradient to shrink exponentially.

After 50-100 time steps, $dh_1$ may be essentially zero, preventing the network from learning long-range dependencies.

This is why architectures like LSTM and GRU were invented: they provide paths for gradients to flow without repeated multiplication by small numbers.
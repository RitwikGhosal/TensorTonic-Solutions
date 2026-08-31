## The Initialization Problem

Neural network training starts with random weights. Poor initialization causes:

**Vanishing gradients:** Weights too small, activations shrink through layers, gradients become tiny.

**Exploding gradients:** Weights too large, activations grow through layers, gradients become huge.

Both prevent effective learning. The network either learns nothing (vanishing) or becomes unstable (exploding).

---

## The Goal of Good Initialization

We want activations and gradients to maintain **stable variance** as they flow through the network:

- Forward pass: activation variance stays roughly constant across layers
- Backward pass: gradient variance stays roughly constant across layers

This requires carefully choosing the initial weight distribution based on layer sizes.

---

## Xavier Initialization (Glorot Initialization)

Xavier initialization, proposed by Glorot and Bengio (2010), sets weights to maintain variance for **linear activations** and **tanh/sigmoid**.

**Key insight:** For a layer with $n_{in}$ inputs and $n_{out}$ outputs, the variance of weights should be:

$$
\text{Var}(W) = \frac{2}{n_{in} + n_{out}}
$$

This balances the needs of both forward and backward passes.

---

## Xavier Uniform Distribution

Draw weights uniformly from:

$$
W \sim U\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right)
$$

The bounds $\pm\sqrt{\frac{6}{n_{in} + n_{out}}}$ give the correct variance for a uniform distribution.

**Example:** Layer with 100 inputs, 50 outputs

$\text{bound} = \sqrt{\frac{6}{100 + 50}} = \sqrt{0.04} = 0.2$

Weights drawn from $U(-0.2, 0.2)$

---

## Xavier Normal Distribution

Draw weights from a normal distribution:

$$
W \sim N\left(0, \sqrt{\frac{2}{n_{in} + n_{out}}}\right)
$$

The standard deviation is $\sqrt{\frac{2}{n_{in} + n_{out}}}$.

**Example:** Layer with 100 inputs, 50 outputs

$\sigma = \sqrt{\frac{2}{150}} = \sqrt{0.0133} \approx 0.115$

Weights drawn from $N(0, 0.115)$

---

## Derivation Intuition

Consider a linear layer: $y = Wx$ where $W$ is $(n_{out}, n_{in})$.

Each output $y_j = \sum_{i=1}^{n_{in}} W_{ji} x_i$

Assuming $x_i$ and $W_{ji}$ are independent with zero mean:

$$
\text{Var}(y_j) = n_{in} \cdot \text{Var}(W) \cdot \text{Var}(x)
$$

To keep $\text{Var}(y) = \text{Var}(x)$:

$$
\text{Var}(W) = \frac{1}{n_{in}}
$$

For the backward pass (gradients), a similar analysis gives $\text{Var}(W) = \frac{1}{n_{out}}$.

Xavier compromises: $\text{Var}(W) = \frac{2}{n_{in} + n_{out}}$

---

## When Xavier Works Well

Xavier initialization is designed for:

**Linear activations:** Identity function, no nonlinearity

**Saturating activations:** Tanh and sigmoid, which compress to a bounded range

These activations have gradients close to 1 near zero, matching Xavier's assumptions.

---

## When Xavier Does NOT Work Well

**ReLU activation:**

ReLU zeroes out negative values, effectively halving the variance of activations. Xavier initialization leads to variance shrinking through layers with ReLU.

For ReLU and its variants (Leaky ReLU, ELU), use **He initialization** instead.

---

## Fan-In vs. Fan-Out

**Fan-in ($n_{in}$):** Number of inputs to a neuron

**Fan-out ($n_{out}$):** Number of outputs from a neuron

Different variants emphasize one or the other:

**Xavier (balanced):** Uses both: $\frac{2}{n_{in} + n_{out}}$

**LeCun initialization:** Uses fan-in only: $\frac{1}{n_{in}}$

The balanced approach works for both forward and backward passes.

---

## For Convolutional Layers

For a conv layer with kernel size $(k_h, k_w)$, $c_{in}$ input channels, $c_{out}$ output channels:

**Fan-in:** $n_{in} = c_{in} \times k_h \times k_w$

**Fan-out:** $n_{out} = c_{out} \times k_h \times k_w$

Apply Xavier using these values.

**Example:** 3x3 conv, 64 input channels, 128 output channels

$n_{in} = 64 \times 3 \times 3 = 576$

$n_{out} = 128 \times 3 \times 3 = 1152$

$\text{Var}(W) = \frac{2}{576 + 1152} = \frac{2}{1728} \approx 0.00116$

---

## Comparing Initialization Methods

**Zero initialization:**
All weights = 0. Terrible: all neurons compute the same thing, no learning.

**Small random:**
$W \sim N(0, 0.01)$. Can work for shallow networks, fails for deep ones.

**Xavier:**
Variance scaled by layer size. Works for tanh/sigmoid.

**He:**
Variance scaled for ReLU. Standard for modern CNNs.

---

## Practical Usage

**For tanh/sigmoid activations:** Use Xavier

**For ReLU/Leaky ReLU:** Use He initialization

**For SELU:** Use LeCun initialization

**For Transformers:** Often use Xavier or custom scaled initialization

Most deep learning frameworks provide both:
- Xavier/Glorot uniform and normal
- He/Kaiming uniform and normal

---

## Biases

Biases are typically initialized to **zero**:

$$
b = 0
$$

Biases do not suffer from the same scaling issues as weights. Zero initialization is simple and effective.

Some exceptions:
- LSTM forget gate bias often initialized to 1
- Certain architectures may use small positive biases

---

## Historical Context

Before Xavier (pre-2010):
- Small random initialization was common
- Deep networks were hard to train
- People thought deep learning did not work

Xavier initialization was a key breakthrough that enabled training deeper networks, along with:
- Better activation functions
- Batch normalization
- Skip connections

---

## Empirical Validation

You can verify initialization quality by:

1. Initialize the network
2. Pass random data through (forward pass)
3. Check activation statistics at each layer
4. Variance should be approximately constant (not growing or shrinking)

If variance collapses to 0 or explodes to infinity, initialization is wrong.
## The Need for Bounded, Zero-Centered Activation

Before ReLU took over, neural networks needed activation functions that had two properties:

1. **Bounded outputs**: prevent activations from growing unboundedly as they pass through many layers
2. **Zero-centered**: output both positive and negative values, so the mean activation is close to zero

Sigmoid ($\sigma(x) = \frac{1}{1 + e^{-x}}$) provides bounded outputs in $(0, 1)$, but it is **not zero-centered**. All outputs are positive, which creates a systematic bias that can slow down gradient descent.

Tanh solves this.

---

## What Tanh Does

The hyperbolic tangent function is:

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

It squashes any input into the range $(-1, 1)$:

- $\tanh(-5) \approx -0.9999$ (saturated negative)
- $\tanh(-1) \approx -0.762$
- $\tanh(0) = 0$ (exactly zero at the origin)
- $\tanh(1) \approx 0.762$
- $\tanh(5) \approx 0.9999$ (saturated positive)

The function is an S-shaped curve, symmetric around the origin. It looks similar to sigmoid but shifted and scaled.

---

## Tanh and Sigmoid: The Exact Relationship

Tanh is actually a rescaled version of sigmoid:

$$
\tanh(x) = 2\sigma(2x) - 1
$$

Both functions have the same S-shape. The difference:

- **Sigmoid**: output range $(0, 1)$, centered at $0.5$
- **Tanh**: output range $(-1, 1)$, centered at $0$

This shift matters for training. When all activations are positive (sigmoid), the gradients for the weights in the next layer all have the same sign. This constrains the gradient to only move in certain directions, causing a zigzag path during optimization. Zero-centered activations (tanh) allow gradients to have mixed signs, enabling more direct paths to the minimum.

---

## The Gradient

The derivative of tanh has a clean formula:

$$
\frac{d}{dx} \tanh(x) = 1 - \tanh^2(x)
$$

This means once you have computed $\tanh(x)$, the gradient is essentially free to calculate.

Some derivative values:

- At $x = 0$: derivative $= 1 - 0^2 = 1$ (maximum)
- At $x = 1$: derivative $= 1 - 0.762^2 \approx 0.42$
- At $x = 2$: derivative $= 1 - 0.964^2 \approx 0.07$
- At $x = 5$: derivative $\approx 0.00018$ (nearly zero)

The gradient is largest at the origin and drops off quickly as the input moves away from zero. For $|x| > 3$, the gradient is practically zero.

---

## The Vanishing Gradient Problem

This rapid gradient decay is tanh's biggest weakness. In deep networks or recurrent networks:

- The gradient at each layer is multiplied by the tanh derivative (at most 1, often much less)
- After many layers, the product of these small numbers shrinks exponentially
- Early layers receive almost no gradient signal and learn very slowly

For example, if the derivative is about 0.4 at each of 10 layers:

$$
0.4^{10} \approx 0.0001
$$

The gradient reaching the first layer is 10,000 times smaller than at the last layer.

This is why tanh was largely replaced by ReLU for deep feedforward networks. ReLU's gradient is 1 for positive inputs, so it does not suffer from this multiplicative shrinking.

---

## Where Tanh Is Still Used

Despite the vanishing gradient issue, tanh remains important in several contexts:

- **RNN hidden states**: the tanh RNN cell uses tanh to keep the hidden state bounded in $[-1, 1]$. Without bounding, the hidden state could explode over many time steps.
- **LSTM and GRU internals**: both architectures use tanh to generate candidate values for the cell/hidden state. The gates use sigmoid (for values in $[0,1]$), but the state updates use tanh (for values in $[-1,1]$).
- **Output layers for bounded targets**: when the target values are in $[-1, 1]$, tanh is a natural choice for the output activation.
- **Normalizing representations**: tanh can be used to normalize embeddings or feature vectors to a bounded range.
- **Historical importance**: tanh was the dominant activation function from the 1990s through the late 2000s, used in almost every neural network before the ReLU revolution.

---

## Numerical Stability

Computing tanh from the raw formula $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ can cause overflow for large $|x|$ because $e^x$ grows very fast. In practice:

- For large positive $x$: $\tanh(x) \approx 1$
- For large negative $x$: $\tanh(x) \approx -1$
- Most numerical libraries handle this internally, so you can call the built-in tanh function without worrying about overflow
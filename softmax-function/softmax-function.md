## From Scores to Probabilities

The final layer of a classification network outputs a vector of raw numbers, one per class. These are called **logits**. For a 3-class problem, you might get:

$$
z = [2.0, 1.0, 0.5]
$$

These logits tell you that class 0 is the most confident prediction and class 2 is the least. But they are not probabilities:

- They do not sum to 1
- They can be negative
- Their magnitude is hard to interpret

Softmax converts these raw scores into a proper **probability distribution**: all values between 0 and 1, summing to exactly 1.

---

## The Softmax Formula

For a vector $z = [z_1, z_2, \ldots, z_n]$:

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}
$$

Each element is exponentiated, then divided by the sum of all exponentiated elements.

**Example**: $z = [2.0, 1.0, 0.5]$

1. Exponentiate each element:
   - $e^{2.0} \approx 7.389$
   - $e^{1.0} \approx 2.718$
   - $e^{0.5} \approx 1.649$

2. Sum: $7.389 + 2.718 + 1.649 = 11.756$

3. Divide each by the sum:
   - $\frac{7.389}{11.756} \approx 0.629$
   - $\frac{2.718}{11.756} \approx 0.231$
   - $\frac{1.649}{11.756} \approx 0.140$

Result: $[0.629, 0.231, 0.140]$. These sum to 1.0 and can be interpreted as: "63% chance of class 0, 23% chance of class 1, 14% chance of class 2."

---

## Why Exponentiation?

The exponential function $e^x$ does several things at once:

- **Makes everything positive**: $e^x > 0$ for all $x$, even negative inputs. This ensures all probabilities are positive.
- **Preserves ordering**: if $z_i > z_j$, then $e^{z_i} > e^{z_j}$. The highest logit still gets the highest probability.
- **Amplifies differences**: the exponential is a convex, rapidly growing function. A small difference in logits becomes a larger difference in probabilities. If one class has logit 5 and another has logit 3, the ratio of their probabilities is $e^{5}/e^{3} = e^{2} \approx 7.4$, not just $5/3$.

This amplification effect means softmax is "opinionated." It concentrates probability mass on the largest logits. The more confident the network is (larger gap between top logit and the rest), the closer the output is to a one-hot vector.

---

## The Numerical Overflow Problem

There is a practical issue. If $z_i$ is large (say, 1000), then $e^{1000}$ overflows to infinity in floating-point arithmetic. Even moderately large values like $e^{100} \approx 2.7 \times 10^{43}$ can cause problems.

The fix is to subtract the maximum value before exponentiating:

$$
\text{softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum_{j} e^{z_j - \max(z)}}
$$

This works because:

$$
\frac{e^{z_i - c}}{\sum_j e^{z_j - c}} = \frac{e^{z_i} / e^c}{\sum_j e^{z_j} / e^c} = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Subtracting any constant $c$ from all elements does not change the result. Choosing $c = \max(z)$ ensures the largest exponent is $e^0 = 1$, and all others are $e^{\text{negative}} < 1$. No overflow possible.

**Example**: $z = [1000, 999, 998]$

Without the trick: $e^{1000} = \text{Inf}$. Computation fails.

With the trick: subtract 1000 to get $[0, -1, -2]$, then:
- $e^0 = 1.0$, $e^{-1} \approx 0.368$, $e^{-2} \approx 0.135$
- Sum $= 1.503$
- Result: $[0.665, 0.245, 0.090]$

---

## Softmax Temperature

A common modification is to divide the logits by a **temperature** parameter $T$ before applying softmax:

$$
\text{softmax}(z_i; T) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
$$

Temperature controls how "sharp" or "soft" the distribution is:

- $T = 1$: standard softmax
- $T < 1$ (e.g., 0.1): sharper distribution. The highest logit dominates even more. At $T \to 0$, softmax approaches argmax (one-hot).
- $T > 1$ (e.g., 5.0): softer distribution. Probabilities become more uniform. At $T \to \infty$, all classes get equal probability.

Temperature is used in:
- **Language model sampling**: lower temperature gives more deterministic text, higher temperature gives more creative/random text
- **Knowledge distillation**: high temperature softens teacher outputs to reveal the "dark knowledge" about class similarities

---

## Softmax on 2D Arrays

When the input is a matrix (batch of logit vectors), softmax is applied **row-wise**. Each row is a separate data point with its own probability distribution:

$$
\text{For each row } i: \quad \text{softmax}(z_{i,j}) = \frac{e^{z_{i,j}}}{\sum_k e^{z_{i,k}}}
$$

The max subtraction and normalization happen independently per row.

---

## Where Softmax Shows Up

- **Classification output layer**: almost every classification network ends with a softmax. A 1000-class ImageNet model outputs 1000 logits, and softmax converts them to 1000 probabilities.
- **Cross-entropy loss**: the standard classification loss is $-\sum_i y_i \log(\text{softmax}(z_i))$. Softmax and cross-entropy are so tightly coupled that frameworks provide a fused "softmax + cross-entropy" function for numerical stability.
- **Attention mechanisms**: in Transformers, attention weights are computed by applying softmax to the dot products of queries and keys. This ensures the attention weights sum to 1 (a weighted average).
- **Reinforcement learning**: softmax is used to convert action-value estimates into action selection probabilities (Boltzmann exploration).
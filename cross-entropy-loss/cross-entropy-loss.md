## What Is Cross-Entropy Loss?

Cross-entropy loss (also called log loss or softmax loss) is the standard loss function for classification problems. It measures how different the predicted probability distribution is from the true distribution.

For multi-class classification with $C$ classes:

$$
\text{CE} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)
$$

Where:
- $y_c$ is 1 if class $c$ is the true class, 0 otherwise (one-hot encoding)
- $\hat{y}_c$ is the predicted probability for class $c$ (output of softmax)

---

## Simplification for One-Hot Labels

When the true label is one-hot encoded, only one term in the sum is non-zero:

$$
\text{CE} = -\log(\hat{y}_{\text{true class}})
$$

This is equivalent to: "negative log of the predicted probability for the correct class."

Examples:
- True class: 2, predicted probability for class 2: 0.9
- Loss: $-\log(0.9) \approx 0.105$

- True class: 2, predicted probability for class 2: 0.1
- Loss: $-\log(0.1) \approx 2.303$

The more confident the model is in the correct class, the lower the loss.

---

## Understanding the Negative Log

Why negative logarithm? Let us trace through the reasoning:

**Step 1: We want high probability for the true class**
- If predicted probability = 1.0 (perfect prediction), loss should be 0
- If predicted probability = 0.0 (complete failure), loss should be infinite

**Step 2: Log gives us this behavior**
- log(1) = 0
- log(x) approaches negative infinity as x approaches 0

**Step 3: Negate to make it a loss**
- -log(1) = 0 (perfect, no loss)
- -log(0.5) = 0.693 (uncertain)
- -log(0.1) = 2.303 (confident but wrong)
- -log(0.01) = 4.605 (very confident and wrong)

The loss grows without bound as the predicted probability for the true class approaches zero.

---

## The Loss Curve

For a single sample where the true class is $c$, the loss as a function of predicted probability:

**Predicted probability 0.99:** loss = 0.010
**Predicted probability 0.90:** loss = 0.105
**Predicted probability 0.70:** loss = 0.357
**Predicted probability 0.50:** loss = 0.693
**Predicted probability 0.30:** loss = 1.204
**Predicted probability 0.10:** loss = 2.303
**Predicted probability 0.01:** loss = 4.605

Key observations:
- Loss is nearly zero when confidence is high
- Loss grows slowly at first, then rapidly as probability approaches zero
- The curve is convex (bowl-shaped), which helps optimization

---

## The Gradient

The gradient of cross-entropy with respect to the logits (before softmax) has a remarkably simple form:

$$
\frac{\partial \text{CE}}{\partial z_c} = \hat{y}_c - y_c
$$

Where $z_c$ is the logit for class $c$.

What this means:
- For the true class ($y_c = 1$): gradient = predicted - 1 (negative, pushes logit up)
- For other classes ($y_c = 0$): gradient = predicted (positive, pushes logit down)

The gradient is exactly the difference between predicted and true probabilities. This simplicity is one reason cross-entropy works so well with softmax.

---

## Cross-Entropy and Information Theory

Cross-entropy comes from information theory. Given two probability distributions $p$ (true) and $q$ (predicted):

$$
H(p, q) = -\sum_x p(x) \log q(x)
$$

This measures the average number of bits needed to encode samples from $p$ using a code optimized for $q$.

- If $q = p$ (perfect prediction): cross-entropy equals entropy of $p$ (minimum possible)
- If $q \neq p$: cross-entropy is larger than entropy of $p$
- The difference is the KL divergence: $D_{KL}(p || q) = H(p, q) - H(p)$

Minimizing cross-entropy is equivalent to minimizing KL divergence from the true distribution.

---

## Binary Cross-Entropy

For binary classification (2 classes), the formula simplifies:

$$
\text{BCE} = -[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]
$$

Where:
- $y \in \{0, 1\}$ is the true label
- $\hat{y} \in (0, 1)$ is the predicted probability for class 1

This is equivalent to:
- If $y = 1$: loss $= -\log(\hat{y})$
- If $y = 0$: loss $= -\log(1 - \hat{y})$

Binary cross-entropy is used with sigmoid output, while multi-class cross-entropy is used with softmax output.

---

## Numerical Stability

Computing log(predicted) directly is dangerous:
- If predicted = 0: log(0) = negative infinity
- If predicted is a very small float: log(predicted) can underflow

Solutions:
- Clip predictions: predicted = clip(predicted, epsilon, 1 - epsilon) where epsilon is around 1e-7
- Use fused operations: compute log(softmax(z)) directly from logits z (more stable)
- Frameworks provide stable implementations: torch.nn.CrossEntropyLoss, tf.nn.softmax_cross_entropy_with_logits

---

## Where Cross-Entropy Is Used

- **Image classification**: predicting which of $C$ classes an image belongs to
- **Language modeling**: predicting the next word from a vocabulary
- **Sentiment analysis**: classifying text as positive/negative/neutral
- **Object detection**: classifying detected objects
- **Any multi-class or binary classification task**

Cross-entropy is the default loss for classification because:
- It has a clean probabilistic interpretation
- The gradient is simple and well-behaved
- It strongly penalizes confident wrong predictions
- It works seamlessly with softmax/sigmoid outputs
## The Origins of Hinge Loss

Hinge loss is the loss function behind Support Vector Machines (SVMs). It was designed with a specific goal: maximize the margin between classes.

For binary classification with labels $y \in \{-1, +1\}$ and raw model output $f(x)$ (not a probability):

$$
L = \max(0, 1 - y \cdot f(x))
$$

This deceptively simple formula encodes a powerful idea: the model should not just classify correctly, but classify with confidence.

---

## Breaking Down the Formula

The term $y \cdot f(x)$ is called the **functional margin**:
- If y = +1 and f(x) > 0: correct classification, positive margin
- If y = -1 and f(x) < 0: correct classification, positive margin
- If y = +1 and f(x) < 0: wrong classification, negative margin
- If y = -1 and f(x) > 0: wrong classification, negative margin

The loss max(0, 1 - y * f(x)) then says:
- If margin >= 1: loss is 0 (correct and confident)
- If margin < 1: loss is (1 - margin) (penalize low confidence or wrong predictions)

---

## Numerical Examples

**Example 1: Confident correct prediction**
- True label: y = +1
- Model output: f(x) = 2.5
- Margin: 1 * 2.5 = 2.5
- Loss: max(0, 1 - 2.5) = max(0, -1.5) = 0 (correct)

**Example 2: Weak correct prediction**
- True label: y = +1
- Model output: f(x) = 0.3
- Margin: 1 * 0.3 = 0.3
- Loss: max(0, 1 - 0.3) = 0.7 (penalized for low confidence)

**Example 3: Wrong prediction**
- True label: y = +1
- Model output: f(x) = -0.5
- Margin: 1 * (-0.5) = -0.5
- Loss: max(0, 1 - (-0.5)) = 1.5 (high penalty)

**Example 4: Confident wrong prediction**
- True label: y = -1
- Model output: f(x) = 2.0
- Margin: (-1) * 2.0 = -2.0
- Loss: max(0, 1 - (-2.0)) = 3.0 (severe penalty)

---

## The Margin Concept

The "1" in hinge loss represents the **target margin**. The model is asked to:
- Not just get the sign right
- But get the sign right with magnitude at least 1

Why margin matters:
- A model that outputs f(x) = 0.001 for a positive sample is technically correct
- But it is not confident; small perturbations could flip the prediction
- Hinge loss says: "I want you to output at least f(x) = 1"

This is the core idea behind SVMs: find the hyperplane that separates classes with the largest margin.

---

## The Loss Curve

For a positive sample (y = +1), plotting loss vs. f(x):

**f(x) = -2.0:** margin = -2.0, loss = 3.0
**f(x) = -1.0:** margin = -1.0, loss = 2.0
**f(x) = 0.0:** margin = 0.0, loss = 1.0
**f(x) = 0.5:** margin = 0.5, loss = 0.5
**f(x) = 1.0:** margin = 1.0, loss = 0.0
**f(x) = 2.0:** margin = 2.0, loss = 0.0

The curve is:
- Linear with slope -1 for f(x) < 1
- Flat at 0 for f(x) >= 1
- Has a "hinge" (sharp corner) at f(x) = 1

This is where the name "hinge loss" comes from.

---

## The Gradient

$$
\frac{\partial L}{\partial f(x)} = \begin{cases} 0 & \text{if } y \cdot f(x) \geq 1 \\ -y & \text{if } y \cdot f(x) < 1 \end{cases}
$$

Key properties:
- **Zero gradient for confident correct predictions**: once the margin exceeds 1, no more gradient. The model does not try to push the margin even higher.
- **Constant gradient otherwise**: the gradient is always +/- 1 (just the sign), regardless of how wrong the prediction is.

This sparsity of gradients is why SVMs have "support vectors": only samples near the decision boundary contribute to the gradient.

---

## Hinge Loss vs. Cross-Entropy

**Hinge loss:**
- Outputs are raw scores (unbounded)
- Zero loss for confident correct predictions
- No gradient once margin > 1
- Piecewise linear (non-smooth at the hinge)

**Cross-entropy:**
- Outputs are probabilities (via softmax/sigmoid)
- Never zero loss (always some gradient)
- Continuously pushes toward higher confidence
- Smooth everywhere

Hinge loss is "satisfied" once the margin is large enough. Cross-entropy always wants higher confidence, even for samples already correctly classified with high confidence.

---

## Multi-Class Hinge Loss

For $C$ classes, the multi-class hinge loss is:

$$
L = \sum_{j \neq y} \max(0, 1 + f_j(x) - f_y(x))
$$

Where:
- $y$ is the true class
- $f_y(x)$ is the score for the true class
- $f_j(x)$ is the score for each incorrect class

Interpretation: for each wrong class, penalize if its score is within 1 of the true class score.

---

## Squared Hinge Loss

A variant that penalizes misclassifications more heavily:

$$
L = \max(0, 1 - y \cdot f(x))^2
$$

Differences from standard hinge:
- Squared penalty for violations
- Smooth gradient at the hinge point
- Larger penalty for confident wrong predictions
- Sometimes used when you want to more heavily penalize errors

---

## Where Hinge Loss Is Used

- **Support Vector Machines**: the original and primary use case
- **Maximum-margin classifiers**: any model aiming for large margins
- **Structured prediction**: variants like structured hinge loss for sequence labeling
- **Neural networks**: sometimes used as an alternative to cross-entropy, especially for binary classification
- **Ranking problems**: pairwise hinge loss for learning to rank
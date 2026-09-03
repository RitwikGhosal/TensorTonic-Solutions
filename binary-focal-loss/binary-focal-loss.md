## The Class Imbalance Problem

In many binary classification tasks, one class is much more common than the other:
- Fraud detection: 99.9% legitimate, 0.1% fraud
- Medical diagnosis: 95% healthy, 5% diseased
- Object detection: most regions are background, few contain objects

Standard binary cross-entropy treats all samples equally. This causes problems:

**During training:**
- The model sees vastly more negative examples
- Gradients are dominated by the majority class
- The model learns to always predict "negative" (achieves high accuracy by ignoring the minority class)

**The loss breakdown:**
- 1000 easy negatives with loss 0.01 each contribute 10.0 total
- 10 hard positives with loss 2.0 each contribute 20.0 total
- The easy negatives still contribute significantly to the gradient

---

## What Focal Loss Does

Focal loss adds a modulating factor that down-weights easy examples:

$$
\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

Where:
- $p_t$ is the predicted probability for the true class
- $\gamma$ is the focusing parameter (typically 2)
- $\alpha_t$ is the class balancing weight
- $(1 - p_t)^\gamma$ is the modulating factor

The modulating factor $(1 - p_t)^\gamma$ is small when $p_t$ is high (easy examples) and large when $p_t$ is low (hard examples).

---

## Breaking Down the Formula

For binary classification:

$$
p_t = \begin{cases} p & \text{if } y = 1 \\ 1 - p & \text{if } y = 0 \end{cases}
$$

Where $p$ is the model's predicted probability for class 1.

Expanded formula:

$$
\text{FL} = \begin{cases} -\alpha (1 - p)^\gamma \log(p) & \text{if } y = 1 \\ -(1 - \alpha) \cdot p^\gamma \log(1 - p) & \text{if } y = 0 \end{cases}
$$

Two components work together:
- **Class weight alpha**: balances importance of positive vs. negative class
- **Focusing factor**: down-weights well-classified examples

---

## How the Focusing Factor Works

Let us compute the modulating factor $(1 - p_t)^\gamma$ for $\gamma = 2$:

**Confidence 0.9 (very confident):** factor = 0.01, giving 100x reduction
**Confidence 0.8:** factor = 0.04, giving 25x reduction
**Confidence 0.6:** factor = 0.16, giving 6x reduction
**Confidence 0.5 (uncertain):** factor = 0.25, giving 4x reduction
**Confidence 0.3:** factor = 0.49, giving 2x reduction
**Confidence 0.1 (wrong):** factor = 0.81, minimal reduction

Key insight:
- Easy examples (confidence > 0.5) get heavily down-weighted
- Hard examples (confidence < 0.5) retain most of their loss
- The model focuses learning on the examples it struggles with

---

## The Effect of Gamma

$\gamma$ controls how aggressively easy examples are down-weighted:

**gamma = 0:**
- Factor = 1 for all predictions
- No focusing effect
- Reduces to standard cross-entropy (with class weights)

**gamma = 1:**
- Mild focusing
- Easy examples at confidence 0.9 get 10x reduction

**gamma = 2 (most common):**
- Strong focusing
- Easy examples at confidence 0.9 get 100x reduction
- This is the default in most implementations

**gamma = 5:**
- Extreme focusing
- Easy examples at confidence 0.9 get 100,000x reduction
- Can cause instability; rarely used

---

## The Alpha Parameter

Alpha is a class balancing weight:
- Alpha for the positive class (when y = 1)
- (1 - alpha) for the negative class (when y = 0)

Common settings:
- alpha = 0.25 for the minority class in highly imbalanced data
- alpha = 0.5 for balanced data (equal weight)
- Often set to the inverse class frequency

Example with 99% negatives, 1% positives:
- Set alpha = 0.25 for positives
- This means positives get 3x more weight than negatives before focusing

---

## Comparing Losses on Easy vs. Hard Examples

Example: positive sample (y = 1)

**Prediction 0.9:** BCE = 0.105, Focal = 0.001, ratio = 105x smaller
**Prediction 0.7:** BCE = 0.357, Focal = 0.032, ratio = 11x smaller
**Prediction 0.5:** BCE = 0.693, Focal = 0.173, ratio = 4x smaller
**Prediction 0.3:** BCE = 1.204, Focal = 0.590, ratio = 2x smaller
**Prediction 0.1:** BCE = 2.303, Focal = 1.866, ratio = 1.2x smaller

Observation:
- For easy examples (high prediction): focal loss is orders of magnitude smaller
- For hard examples (low prediction): focal loss is similar to BCE
- The relative loss contribution shifts toward hard examples

---

## The Gradient

The gradient of focal loss is more complex than cross-entropy:

$$
\frac{\partial \text{FL}}{\partial p} = -\alpha_t \left[ \gamma (1 - p_t)^{\gamma - 1} \log(p_t) + (1 - p_t)^\gamma \cdot \frac{1}{p_t} \right] \cdot \frac{\partial p_t}{\partial p}
$$

Key properties:
- Gradient magnitude is reduced for easy examples (same as loss)
- The model receives stronger learning signal from hard examples
- Convergence can be slower initially but reaches better solutions on imbalanced data

---

## Where Focal Loss Is Used

**Object detection:**
- The original use case (RetinaNet paper by Lin et al., 2017)
- Most anchor boxes are background (easy negatives)
- Focal loss allows training with all anchors instead of hard negative mining

**Medical imaging:**
- Rare disease detection
- Lesion segmentation where most pixels are healthy

**Fraud detection:**
- Rare positive class (fraudulent transactions)
- Most samples are easy negatives (legitimate transactions)

**Any highly imbalanced binary classification task**

---

## Practical Tips

- **Start with gamma = 2**: this works well for most cases
- **Tune alpha based on class frequency**: try inverse frequency first
- **Watch for instability**: very high gamma can cause training issues
- **Compare against class weighting**: sometimes simple class weights work just as well
- **Monitor per-class metrics**: accuracy can be misleading; use precision, recall, F1
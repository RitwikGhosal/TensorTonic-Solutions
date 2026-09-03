## The Problem with Cross-Entropy on Imbalanced Data

Multi-class classification often has imbalanced class distributions:
- Object detection: 10,000 background regions for every object
- Text classification: some categories have 1000x more examples than others
- Fine-grained classification: rare subcategories are underrepresented

Standard cross-entropy computes the average loss across all samples. When classes are imbalanced:
- Easy, well-represented classes dominate the gradient
- Rare classes contribute little to learning
- The model becomes biased toward majority classes

---

## Focal Loss for Multi-Class Classification

Focal loss extends cross-entropy by adding a modulating factor:

$$
\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

For multi-class with $C$ classes and true class $c^*$:

$$
\text{FL} = -\alpha_{c^*} (1 - \hat{y}_{c^*})^\gamma \log(\hat{y}_{c^*})
$$

Where:
- $\hat{y}_{c^*}$ is the softmax probability for the true class
- $\alpha_{c^*}$ is the weight for the true class
- $\gamma$ is the focusing parameter

---

## The Two Key Components

**1. Class weights (alpha values):**
- A weight for each class
- Typically set inversely proportional to class frequency
- Addresses class imbalance directly

**2. Focusing term:**
- Reduces loss for well-classified samples
- Regardless of which class they belong to
- Addresses the "easy example" problem

These work together:
- Rare classes get higher alpha weights
- Easy samples from any class get down-weighted by the focusing term

---

## Numerical Example

Consider 3-class classification:

**Sample 1: Easy, majority class**
- True class: 0 (common class, alpha = 0.1)
- Predicted probabilities: [0.95, 0.03, 0.02]
- p_t = 0.95
- Focusing factor: (1 - 0.95)^2 = 0.0025
- Loss: -0.1 x 0.0025 x log(0.95) = 0.0000128

**Sample 2: Hard, minority class**
- True class: 2 (rare class, alpha = 0.8)
- Predicted probabilities: [0.4, 0.3, 0.3]
- p_t = 0.3
- Focusing factor: (1 - 0.3)^2 = 0.49
- Loss: -0.8 x 0.49 x log(0.3) = 0.472

The hard minority sample contributes 37,000x more to the loss than the easy majority sample.

---

## Effect of Gamma on Different Confidence Levels

How the focusing factor scales with gamma:

**At confidence 0.9 (confident correct):**
- gamma = 0: factor = 1.0 (no reduction)
- gamma = 1: factor = 0.1 (10x reduction)
- gamma = 2: factor = 0.01 (100x reduction)
- gamma = 5: factor = 0.00001 (100,000x reduction)

**At confidence 0.5 (uncertain):**
- gamma = 0: factor = 1.0
- gamma = 1: factor = 0.5 (2x reduction)
- gamma = 2: factor = 0.25 (4x reduction)
- gamma = 5: factor = 0.03125 (32x reduction)

**At confidence 0.1 (confident wrong):**
- gamma = 0: factor = 1.0
- gamma = 1: factor = 0.9 (minimal reduction)
- gamma = 2: factor = 0.81 (minimal reduction)
- gamma = 5: factor = 0.59 (modest reduction)

---

## Setting Alpha Weights

Common strategies for setting alpha values:

**Inverse class frequency:**

$$
\alpha_c = \frac{N}{C \cdot N_c}
$$
Where $N$ is total samples, $N_c$ is samples in class $c$, $C$ is number of classes.

**Effective number of samples:**

$$
\alpha_c = \frac{1 - \beta}{1 - \beta^{N_c}}
$$
Where $\beta \in [0, 1)$ is a hyperparameter. This accounts for diminishing returns of more samples.

**Equal weights:**
- Set all alpha values to 1
- Let the focusing term handle imbalance
- Sometimes works when imbalance is moderate

**Normalized:**
- After computing weights, normalize so they sum to C (number of classes)
- Keeps the loss scale similar to standard cross-entropy

---

## Comparison: Cross-Entropy vs. Focal Loss

Training behavior differences:

**Cross-entropy:**
- All samples contribute equally (weighted by class weights if used)
- Easy samples generate small but non-zero gradients
- Gradient is dominated by the sheer number of easy samples
- Model quickly learns easy patterns, then slowly improves on hard cases

**Focal loss:**
- Easy samples contribute almost nothing
- Hard samples dominate the gradient
- Model is forced to focus on difficult cases from the start
- Can achieve better performance on hard/rare cases

---

## The Gradient

The gradient of focal loss with respect to the logit for the true class:

$$
\frac{\partial \text{FL}}{\partial z_{c^*}} = \alpha_{c^*} \left[ \gamma (1 - p_t)^{\gamma - 1} p_t \log(p_t) + (1 - p_t)^\gamma (p_t - 1) \right]
$$

Properties:
- More complex than cross-entropy gradient (predicted - true)
- Gradient magnitude is scaled by the focusing factor
- Easy samples have tiny gradients
- Hard samples have gradients similar to cross-entropy

---

## Implementation Considerations

**Numerical stability:**
- Compute log(p_t) carefully when p_t is small
- Use log-softmax for better numerical stability
- Clip p_t to avoid log(0)

**Initialization:**
- With focal loss, the model starts seeing primarily hard examples
- May need to adjust initial bias: set output layer bias to log((1 - pi) / pi) where pi is prior probability of rare class
- This prevents early training instability

**Batch size:**
- Focal loss reduces effective batch size (fewer samples contribute meaningfully)
- May need larger batches for stable gradient estimates

---

## When to Use Focal Loss

**Good use cases:**
- Highly imbalanced classification (1:100 or more)
- Dense prediction tasks (object detection, segmentation)
- When hard negative mining is too slow or complex
- When class weights alone do not solve the problem

**May not help:**
- Balanced datasets
- When "hard" examples are actually mislabeled
- Very small datasets (may need all gradient signal)

---

## Origin and Applications

Focal loss was introduced by Lin et al. in "Focal Loss for Dense Object Detection" (2017):
- Enabled single-stage detectors (RetinaNet) to match two-stage detectors
- Eliminated need for complicated sampling strategies
- Now widely adopted beyond object detection:
  - Medical image analysis
  - Fraud detection
  - Document classification
  - Any imbalanced classification task
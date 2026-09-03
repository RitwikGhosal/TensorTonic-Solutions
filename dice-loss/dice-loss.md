## The Segmentation Problem

In semantic segmentation, the goal is to classify each pixel in an image:
- Input: an image of size $H \times W$
- Output: a mask of size $H \times W$ where each pixel has a class label

The challenge: class imbalance is extreme.
- Example: segmenting tumors in brain scans
- 99% of pixels are healthy tissue (background)
- 1% of pixels are tumor (foreground)

If you use pixel-wise cross-entropy:
- The model can achieve 99% accuracy by predicting "background" everywhere
- It completely ignores the region of interest

---

## What Dice Loss Measures

Dice loss is based on the **Dice coefficient** (also called the Sorensen-Dice coefficient or F1 score):

$$
\text{Dice} = \frac{2 |A \cap B|}{|A| + |B|}
$$

Where:
- $A$ is the set of predicted positive pixels
- $B$ is the set of ground truth positive pixels
- $|A \cap B|$ is the number of pixels in both sets (true positives)
- $|A| + |B|$ is the total count of positive pixels in both masks

The Dice coefficient ranges from 0 (no overlap) to 1 (perfect overlap).

---

## Converting to a Differentiable Loss

The set-based formula is not differentiable. For training, we use soft predictions:

$$
\text{Dice Loss} = 1 - \frac{2 \sum_i p_i g_i + \epsilon}{\sum_i p_i + \sum_i g_i + \epsilon}
$$

Where:
- $p_i$ is the predicted probability for pixel $i$ (from sigmoid/softmax)
- $g_i$ is the ground truth label for pixel $i$ (0 or 1)
- $\epsilon$ is a small constant for numerical stability (e.g., $10^{-6}$)
- The sum is over all pixels

This is "soft" because:
- Instead of hard 0/1 predictions, we use probabilities
- The intersection $\sum_i p_i g_i$ is a soft count of true positives
- As predictions become more confident (0 or 1), this approaches the hard Dice score

---

## Numerical Example

Consider a tiny 4-pixel image:

| Pixel | Ground Truth $g$ | Prediction $p$ | $p \cdot g$ |
|-------|------------------|----------------|-------------|
| 1 | 1 | 0.9 | 0.9 |
| 2 | 1 | 0.7 | 0.7 |
| 3 | 0 | 0.2 | 0.0 |
| 4 | 0 | 0.1 | 0.0 |

Calculations:
- $\sum p_i g_i = 0.9 + 0.7 = 1.6$ (soft intersection)
- $\sum p_i = 0.9 + 0.7 + 0.2 + 0.1 = 1.9$
- $\sum g_i = 1 + 1 + 0 + 0 = 2$

Dice coefficient:

$$
\text{Dice} = \frac{2 \times 1.6}{1.9 + 2} = \frac{3.2}{3.9} \approx 0.82
$$

Dice loss:

$$
\text{Loss} = 1 - 0.82 = 0.18
$$

---

## Why Dice Loss Handles Imbalance

Consider extreme imbalance: 1000 pixels, only 10 are foreground.

**With cross-entropy:**
- Predicting all background: 990 correct, 10 wrong
- Loss is dominated by the 990 easy background pixels
- Small gradient signal from the 10 foreground pixels

**With Dice loss:**
- Only foreground pixels contribute to the intersection term
- Background pixels only affect the denominator slightly
- The loss directly measures overlap with the foreground region
- If you miss the foreground, Dice = 0, loss = 1 (maximum)

Dice loss asks: "What fraction of the foreground did you capture?" rather than "What fraction of pixels did you classify correctly?"

---

## The Gradient

For a predicted probability $p_i$ on a foreground pixel ($g_i = 1$):

$$
\frac{\partial \text{Dice}}{\partial p_i} = \frac{2(\sum g)(\sum p + \sum g) - 2(\sum pg)(1)}{(\sum p + \sum g)^2}
$$

Key insight:
- The gradient depends on the global statistics ($\sum p$, $\sum g$, $\sum pg$)
- This creates a "soft" constraint that considers all pixels together
- Unlike cross-entropy, where each pixel's gradient is independent

---

## Dice vs. IoU (Jaccard)

IoU (Intersection over Union) is a related metric:

$$
\text{IoU} = \frac{|A \cap B|}{|A \cup B|}
$$

Relationship to Dice:

$$
\text{Dice} = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}}
$$

They are monotonically related: maximizing one maximizes the other.

Differences:
- Dice weights the intersection twice
- IoU is the standard evaluation metric in segmentation
- Dice loss is often preferred for training (better gradient properties)

---

## Generalized Dice Loss

For multi-class segmentation, Generalized Dice Loss (GDL) handles multiple classes and per-class weighting:

$$
\text{GDL} = 1 - 2 \frac{\sum_{c} w_c \sum_i p_{ic} g_{ic}}{\sum_{c} w_c (\sum_i p_{ic} + \sum_i g_{ic})}
$$

Where:
- $c$ indexes classes
- $w_c$ is the weight for class $c$ (often inversely proportional to class frequency)

This balances the contribution of rare classes against common ones.

---

## Combining Dice with Cross-Entropy

A common practice is to use both losses together:

$$
L = \alpha \cdot \text{Dice Loss} + (1 - \alpha) \cdot \text{Cross-Entropy}
$$

Why combine them?
- Cross-entropy provides per-pixel gradients (local signal)
- Dice loss provides region-level gradients (global signal)
- Together, they give more stable training
- $\alpha = 0.5$ is a common starting point

---

## Where Dice Loss Is Used

- **Medical image segmentation**: organ segmentation, tumor detection, lesion identification
- **Satellite image analysis**: land cover classification, building footprint extraction
- **Any binary/multi-class segmentation with class imbalance**
- **Instance segmentation**: as part of mask head losses
- **3D segmentation**: volumetric medical imaging (CT, MRI)
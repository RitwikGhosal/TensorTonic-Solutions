## The Problem with Hard Labels

Standard classification uses one-hot labels:
- True class: 1
- All other classes: 0

Example for 5-class classification with true class 2:
- Hard label: [0, 0, 1, 0, 0]

Cross-entropy with hard labels pushes the model to:
- Predict probability 1.0 for the true class
- Predict probability 0.0 for all other classes

This causes problems:

**Overconfidence:** The model becomes extremely confident, outputting predictions like [0.001, 0.001, 0.997, 0.001, 0.000]. This overconfidence is rarely warranted.

**Poor calibration:** The model's confidence does not match its accuracy. A 95% confident prediction might only be correct 80% of the time.

**Sensitivity to noise:** If labels have errors, the model tries hard to fit the wrong labels.

---

## What Label Smoothing Does

Label smoothing replaces hard labels with soft labels:

$$
y_{\text{smooth}} = (1 - \epsilon) \cdot y_{\text{hard}} + \frac{\epsilon}{K}
$$

Where:
- $\epsilon$ is the smoothing parameter (typically 0.1)
- $K$ is the number of classes
- $y_{\text{hard}}$ is the original one-hot vector

For true class $c$:
- $y_c = 1 - \epsilon + \epsilon/K$

For other classes $j \neq c$:
- $y_j = \epsilon/K$

---

## Numerical Example

5-class classification, true class = 2, smoothing = 0.1:

**Hard label:** [0, 0, 1, 0, 0]

**Smoothed label:**
- True class: 1 - 0.1 + 0.1/5 = 0.92
- Other classes: 0.1/5 = 0.02
- Result: [0.02, 0.02, 0.92, 0.02, 0.02]

The model is now encouraged to:
- Predict around 92% for the true class (not 100%)
- Predict around 2% for other classes (not 0%)

---

## The Loss Function

Label smoothing modifies cross-entropy by using soft labels:

$$
L = -\sum_{k=1}^{K} y_k^{\text{smooth}} \log(\hat{y}_k)
$$

Expanding:

$$
L = (1 - \epsilon) \cdot [-\log(\hat{y}_c)] + \epsilon \cdot \left[-\frac{1}{K}\sum_{k=1}^{K} \log(\hat{y}_k)\right]
$$

This is:
- $(1 - \epsilon)$ times the original cross-entropy loss
- Plus $\epsilon$ times the entropy of the predicted distribution (with uniform target)

The second term pushes predictions toward uniform, counteracting overconfidence.

---

## Why It Works

**Regularization effect:**
- The model cannot achieve zero loss (target is not one-hot)
- This prevents the logits from growing unboundedly large
- Acts as implicit weight decay on the final layer

**Better calibration:**
- Predictions are less extreme
- Confidence better reflects actual accuracy
- Important for applications where confidence matters (medical, autonomous driving)

**Robustness to label noise:**
- The model does not try to fit any single label with 100% confidence
- Noisy labels have less impact

**Improved generalization:**
- Often gives small but consistent accuracy improvements
- Especially helpful in large models prone to overfitting

---

## Choosing the Smoothing Parameter

The smoothing parameter $\epsilon$ controls how soft the labels are:

**epsilon = 0:** No smoothing, standard hard labels
**epsilon = 0.1:** Typical value, mild smoothing
**epsilon = 0.2:** Stronger smoothing, more regularization
**epsilon = 1/K:** Uniform labels, the model learns nothing about class identity

Common practice:
- Start with epsilon = 0.1
- Tune based on validation performance
- Higher epsilon for noisier labels or when overfitting is a problem

---

## Effect on Predictions

Without label smoothing:
- Logits can grow very large
- Final predictions cluster near 0 and 1
- Model is overconfident

With label smoothing:
- Logits stay moderate
- Final predictions are less extreme
- Confidence is better calibrated

Example predictions for a sample:

**Without smoothing:** [0.001, 0.002, 0.995, 0.001, 0.001]
**With smoothing:** [0.03, 0.04, 0.88, 0.03, 0.02]

Both predict class 2, but the smoothed version is less extreme.

---

## The Gradient

The gradient with respect to logits changes slightly:

**Without smoothing (for true class):**

$$
\frac{\partial L}{\partial z_c} = \hat{y}_c - 1
$$

**With smoothing (for true class):**

$$
\frac{\partial L}{\partial z_c} = \hat{y}_c - (1 - \epsilon + \epsilon/K)
$$

**For other classes:**

$$
\frac{\partial L}{\partial z_j} = \hat{y}_j - \epsilon/K
$$

The key difference: with smoothing, the gradient for non-true classes is nonzero. The model receives a small signal pushing predictions for other classes away from zero.

---

## Label Smoothing vs. Temperature Scaling

Both affect prediction confidence, but differently:

**Label smoothing:**
- Applied during training
- Changes what the model learns
- Affects both accuracy and calibration
- Permanent effect

**Temperature scaling:**
- Applied after training (during inference)
- Divides logits by temperature before softmax
- Only affects calibration, not accuracy
- Adjustable post-hoc

They can be combined: train with label smoothing, then fine-tune calibration with temperature scaling.

---

## Where Label Smoothing Is Used

- **Image classification**: standard in modern architectures (EfficientNet, ViT)
- **Machine translation**: helps with the large vocabulary (many classes)
- **Speech recognition**: reduces overconfidence in transcriptions
- **Knowledge distillation**: soft labels from teacher are similar to label smoothing
- **Any classification task where calibration matters**

Not recommended when:
- Labels are 100% reliable (e.g., synthetic data)
- You need to distinguish high-confidence correct predictions
- Very small datasets (regularization might hurt)
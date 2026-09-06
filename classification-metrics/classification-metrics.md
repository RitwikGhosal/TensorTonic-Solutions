## The Confusion Matrix Foundation

All classification metrics derive from the **confusion matrix**, which counts predictions vs. actual labels.

For binary classification:

**True Positive (TP):** Predicted positive, actually positive

**True Negative (TN):** Predicted negative, actually negative

**False Positive (FP):** Predicted positive, actually negative (Type I error)

**False Negative (FN):** Predicted negative, actually positive (Type II error)

$$
\begin{array}{c|cc}
 & \text{Predicted +} & \text{Predicted -} \\
\hline
\text{Actual +} & TP & FN \\
\text{Actual -} & FP & TN
\end{array}
$$

---

## Accuracy

The most intuitive metric: what fraction of predictions is correct?

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

**Strengths:**
- Simple and intuitive
- Works well for balanced classes

**Weaknesses:**
- Misleading with imbalanced data
- A model predicting all negatives on 95% negative data gets 95% accuracy

---

## Precision

Of all positive predictions, what fraction is actually positive?

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

**Intuition:** "When the model says positive, how often is it right?"

**High precision matters when:**
- False positives are costly
- Spam detection (do not want to lose important emails)
- Fraud alerts (do not want to bother legitimate customers)

---

## Recall (Sensitivity, True Positive Rate)

Of all actual positives, what fraction did we catch?

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

**Intuition:** "Of all the real positives, how many did we find?"

**High recall matters when:**
- False negatives are costly
- Disease screening (do not want to miss sick patients)
- Security threats (do not want to miss attacks)

---

## The Precision-Recall Trade-off

Increasing the classification threshold:
- Fewer positive predictions
- Precision typically increases (more selective)
- Recall typically decreases (miss more positives)

Decreasing the threshold:
- More positive predictions
- Recall increases (catch more)
- Precision typically decreases (more false positives)

You cannot maximize both simultaneously. The right balance depends on the application.

---

## F1 Score

The harmonic mean of precision and recall:

$$
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \times TP}{2 \times TP + FP + FN}
$$

**Why harmonic mean?**
- Penalizes extreme imbalances
- If precision = 1.0 and recall = 0.01, F1 = 0.02 (not 0.5)
- Both precision and recall must be high for high F1

**F1 range:** 0 to 1, higher is better

---

## F-beta Score

Generalization of F1 that lets you weight precision vs. recall:

$$
F_\beta = (1 + \beta^2) \times \frac{\text{Precision} \times \text{Recall}}{\beta^2 \times \text{Precision} + \text{Recall}}
$$

**F1:** $\beta = 1$, equal weight

**F2:** $\beta = 2$, recall weighted higher (2x importance)

**F0.5:** $\beta = 0.5$, precision weighted higher (2x importance)

---

## Specificity (True Negative Rate)

Of all actual negatives, what fraction did we correctly identify?

$$
\text{Specificity} = \frac{TN}{TN + FP}
$$

**Intuition:** "How well do we identify negatives?"

Important in medical testing where you want to avoid false alarms.

---

## False Positive Rate

$$
\text{FPR} = \frac{FP}{FP + TN} = 1 - \text{Specificity}
$$

Used in ROC curves. The fraction of negatives incorrectly classified as positive.

---

## Worked Example

**100 patients, 20 have disease, 80 healthy**

Model predictions: 25 positive, 75 negative

Confusion matrix:
- TP = 18 (correctly identified sick)
- FN = 2 (missed sick patients)
- FP = 7 (healthy marked as sick)
- TN = 73 (correctly identified healthy)

**Calculations:**

Accuracy = (18 + 73) / 100 = 0.91

Precision = 18 / (18 + 7) = 18/25 = 0.72

Recall = 18 / (18 + 2) = 18/20 = 0.90

F1 = 2 * (0.72 * 0.90) / (0.72 + 0.90) = 1.296 / 1.62 = 0.80

Specificity = 73 / (73 + 7) = 73/80 = 0.91

---

## Matthews Correlation Coefficient (MCC)

A balanced metric that uses all four confusion matrix values:

$$
\text{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
$$

**Range:** -1 to +1
- +1: Perfect prediction
- 0: Random prediction
- -1: Perfect inverse prediction

**Advantages:**
- Works well with imbalanced classes
- Symmetric (treats both classes equally)
- Only high when all four quadrants are good

---

## Balanced Accuracy

Average of recall for each class:

$$
\text{Balanced Accuracy} = \frac{\text{Recall}_+ + \text{Recall}_-}{2} = \frac{TPR + TNR}{2}
$$

Useful for imbalanced datasets where regular accuracy is misleading.

---

## Choosing the Right Metric

**Balanced classes, general purpose:** Accuracy or F1

**Imbalanced classes:** F1, MCC, or Balanced Accuracy

**False positives costly:** Precision

**False negatives costly:** Recall

**Need single number for model comparison:** F1 or MCC

**Threshold will be tuned later:** AUC-ROC

**Multi-class:** Micro/Macro/Weighted F1

---

## Multi-Class Extension

For $n$ classes, the confusion matrix is $n \times n$.

**Per-class metrics:** Treat each class as "positive vs. rest" to compute precision, recall, F1.

**Aggregation:**
- Micro: Sum TP, FP, FN across classes, then compute
- Macro: Compute per-class, then average
- Weighted: Compute per-class, weighted average by class size

---

## Metric Pitfalls

**Accuracy paradox:** High accuracy on imbalanced data is meaningless.

**Optimizing wrong metric:** Optimizing precision alone can destroy recall (and vice versa).

**Ignoring costs:** All metrics treat errors equally. Real-world costs vary.

**Single metric obsession:** Report multiple metrics for a complete picture.

**Threshold dependence:** Precision, recall, F1 depend on threshold. AUC does not.
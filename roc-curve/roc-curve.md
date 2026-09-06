## What the ROC Curve Shows

The ROC (Receiver Operating Characteristic) curve visualizes classifier performance across all possible thresholds. Instead of evaluating at one threshold, it shows the full trade-off between true positive rate and false positive rate.

**X-axis:** False Positive Rate (FPR) = FP / (FP + TN)

**Y-axis:** True Positive Rate (TPR) = TP / (TP + FN)

Each point on the curve corresponds to a specific classification threshold.

---

## The Two Key Rates

**True Positive Rate (TPR), also called Recall or Sensitivity:**

$$
\text{TPR} = \frac{TP}{TP + FN} = \frac{\text{correctly identified positives}}{\text{all actual positives}}
$$

"Of all the real positives, what fraction did we catch?"

**False Positive Rate (FPR), also called Fall-out:**

$$
\text{FPR} = \frac{FP}{FP + TN} = \frac{\text{incorrectly flagged negatives}}{\text{all actual negatives}}
$$

"Of all the real negatives, what fraction did we incorrectly flag?"

---

## How Thresholds Create the Curve

A classifier outputs scores (e.g., probabilities). We choose a threshold $t$: predict positive if score $\geq t$.

**High threshold (e.g., t = 0.9):**
- Very selective, few positive predictions
- Low TPR (miss many positives)
- Low FPR (few false alarms)
- Point near (0, 0)

**Low threshold (e.g., t = 0.1):**
- Permissive, many positive predictions
- High TPR (catch most positives)
- High FPR (many false alarms)
- Point near (1, 1)

Sweeping from high to low threshold traces the ROC curve from (0, 0) to (1, 1).

---

## Computing ROC Curve Points

**Step 1:** Sort all samples by predicted score (descending)

**Step 2:** Start with threshold above max score: TPR = 0, FPR = 0

**Step 3:** Lower threshold to each unique score value:
- Samples at or above threshold become positive predictions
- Recompute TPR and FPR
- Record the point

**Step 4:** End with threshold below min score: TPR = 1, FPR = 1

---

## Worked Example

**Data (score, label):**

(0.9, 1), (0.8, 1), (0.7, 0), (0.6, 1), (0.4, 0), (0.3, 0)

**Totals:** 3 positives, 3 negatives

**Threshold = 1.0 (nothing predicted positive):**
- TP = 0, FP = 0, FN = 3, TN = 3
- TPR = 0/3 = 0, FPR = 0/3 = 0
- Point: (0, 0)

**Threshold = 0.9:**
- Predict positive: (0.9, 1)
- TP = 1, FP = 0
- TPR = 1/3 = 0.33, FPR = 0/3 = 0
- Point: (0, 0.33)

**Threshold = 0.8:**
- Predict positive: (0.9, 1), (0.8, 1)
- TP = 2, FP = 0
- TPR = 2/3 = 0.67, FPR = 0
- Point: (0, 0.67)

**Threshold = 0.7:**
- Predict positive: top 3 including (0.7, 0)
- TP = 2, FP = 1
- TPR = 0.67, FPR = 1/3 = 0.33
- Point: (0.33, 0.67)

**Threshold = 0.6:**
- TP = 3, FP = 1
- TPR = 1.0, FPR = 0.33
- Point: (0.33, 1.0)

**Threshold = 0.4:**
- TP = 3, FP = 2
- TPR = 1.0, FPR = 0.67
- Point: (0.67, 1.0)

**Threshold = 0.3:**
- TP = 3, FP = 3
- TPR = 1.0, FPR = 1.0
- Point: (1, 1)

---

## Interpreting the Curve Shape

**Perfect classifier:**
Curve goes from (0, 0) straight up to (0, 1), then across to (1, 1). All positives are ranked above all negatives.

**Random classifier:**
Diagonal line from (0, 0) to (1, 1). Positives and negatives are randomly mixed.

**Good classifier:**
Curve bows toward the upper-left corner. High TPR achieved with low FPR.

**Worse than random:**
Curve below the diagonal. The model is anti-predictive (flip predictions to improve).

---

## The Upper-Left Corner

The ideal operating point is (0, 1): TPR = 1, FPR = 0. This means:
- All positives correctly identified
- No false alarms

Real classifiers cannot reach this point. The goal is to get as close as possible.

The point on the curve closest to (0, 1) is often a good threshold choice if you want to balance TPR and FPR.

---

## Choosing an Operating Point

Different applications need different trade-offs:

**Medical screening (high TPR critical):**
Choose a point with high TPR even if FPR is moderate. Missing a disease is worse than extra tests.

**Spam filtering (low FPR critical):**
Choose a point with low FPR even if TPR is lower. Losing important emails is worse than seeing some spam.

**Balanced:**
Choose the point closest to (0, 1), or where TPR - FPR is maximized (Youden's J statistic).

---

## ROC Curve vs. Precision-Recall Curve

**ROC Curve:**
- X: FPR, Y: TPR
- Uses TN in FPR calculation
- Less affected by class imbalance
- Good for balanced datasets

**Precision-Recall Curve:**
- X: Recall (TPR), Y: Precision
- Does not use TN
- More informative for imbalanced data
- Shows performance on positive class

For highly imbalanced data (e.g., 1% positives), ROC can look good even when precision is poor. Use PR curve instead.

---

## Area Under ROC Curve (AUC)

The area under the ROC curve summarizes performance in a single number:

**AUC = 1.0:** Perfect separation

**AUC = 0.5:** Random classifier (diagonal line)

**AUC < 0.5:** Worse than random

AUC equals the probability that a random positive is ranked higher than a random negative.

---

## Multi-Class ROC

For multi-class problems, compute ROC curves using:

**One-vs-Rest (OvR):**
For each class, treat it as positive and all others as negative. Plot one curve per class.

**Micro-average:**
Aggregate all classes together, compute global TPR and FPR.

**Macro-average:**
Average the curves across classes.

---

## Implementation Notes

**Handling ties:**
When multiple samples have the same score, they should be grouped. Some implementations step through all of them at once.

**Efficient computation:**
Sort once, then sweep through in O(n) time. No need to recompute confusion matrix at each threshold.

**Returning values:**
Typically return arrays of FPR values, TPR values, and corresponding thresholds.

---

## Plotting Best Practices

**Include the diagonal:**
Draw the random classifier line for reference.

**Mark operating points:**
Indicate specific thresholds of interest.

**Show AUC:**
Include the AUC value in the plot title or legend.

**Equal axes:**
Use equal scaling for X and Y axes so the diagonal is at 45 degrees.
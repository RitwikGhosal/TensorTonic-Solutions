## What Is AUC-ROC?

AUC stands for **Area Under the Curve**. In machine learning, it typically refers to the area under the **ROC curve** (Receiver Operating Characteristic curve).

AUC-ROC measures how well a classifier ranks positive examples above negative examples, regardless of the classification threshold. It answers: "If I pick a random positive and a random negative, what is the probability that the model scores the positive higher?"

---

## The ROC Curve

The ROC curve plots:
- **X-axis:** False Positive Rate (FPR) = FP / (FP + TN)
- **Y-axis:** True Positive Rate (TPR) = TP / (TP + FN)

Each point on the curve corresponds to a different classification threshold.

**At threshold = 0 (predict everything positive):**
- TPR = 1 (all positives caught)
- FPR = 1 (all negatives incorrectly labeled positive)
- Point: (1, 1)

**At threshold = 1 (predict everything negative):**
- TPR = 0 (no positives caught)
- FPR = 0 (no false positives)
- Point: (0, 0)

The ROC curve traces from (0, 0) to (1, 1) as threshold decreases.

---

## Interpreting the ROC Curve

**Perfect classifier:** Goes from (0, 0) up to (0, 1), then across to (1, 1). It achieves 100% TPR with 0% FPR.

**Random classifier:** Diagonal line from (0, 0) to (1, 1). TPR always equals FPR.

**Good classifier:** Curve bows toward the top-left corner. High TPR with low FPR.

**Bad classifier (worse than random):** Curve below the diagonal. The model is anti-predictive.

---

## AUC: The Area Under the ROC Curve

AUC is the total area under the ROC curve:

$$
\text{AUC} = \int_0^1 \text{TPR}(\text{FPR}) \, d(\text{FPR})
$$

**AUC = 1.0:** Perfect classifier. The curve covers the entire upper-left region.

**AUC = 0.5:** Random classifier. The curve is the diagonal.

**AUC = 0.0:** Perfectly wrong classifier. Always predicts the opposite of the true label.

---

## The Probabilistic Interpretation

AUC equals the probability that a randomly chosen positive example is ranked higher than a randomly chosen negative example:

$$
\text{AUC} = P(\text{score}(x^+) > \text{score}(x^-))
$$

This interpretation is intuitive and threshold-independent. It measures **ranking quality**, not classification at any specific threshold.

---

## Computing AUC

**Method 1: Trapezoidal rule**

1. Sort all samples by predicted score (descending)
2. Compute TPR and FPR at each unique threshold
3. Sum the areas of trapezoids between consecutive points

**Method 2: Wilcoxon-Mann-Whitney statistic**

Count all positive-negative pairs where the positive has a higher score:

$$
\text{AUC} = \frac{\sum_{i \in \text{pos}} \sum_{j \in \text{neg}} \mathbf{1}[s_i > s_j]}{|\text{pos}| \times |\text{neg}|}
$$

Ties are often counted as 0.5.

---

## Worked Example

**Predictions (score, label):**
- (0.9, 1)
- (0.8, 1)
- (0.7, 0)
- (0.6, 1)
- (0.5, 0)
- (0.4, 0)

**Positives:** 3, **Negatives:** 3, **Total pairs:** 9

**Counting pairs where positive scores higher than negative:**

Positive 0.9 vs negatives (0.7, 0.5, 0.4): 3 wins

Positive 0.8 vs negatives (0.7, 0.5, 0.4): 3 wins

Positive 0.6 vs negatives (0.7, 0.5, 0.4): 2 wins (loses to 0.7)

Total wins: 3 + 3 + 2 = 8

AUC = 8 / 9 = 0.889

---

## AUC vs. Accuracy

**Accuracy:** Measures performance at a single threshold. Sensitive to class imbalance.

**AUC:** Measures ranking over all thresholds. Threshold-independent.

**Example showing the difference:**

Dataset: 95 negatives, 5 positives

Model A: Predicts all negative. Accuracy = 95%. AUC = 0.5 (random).

Model B: Ranks all positives in top 10. Accuracy depends on threshold. AUC = 1.0 (perfect ranking).

AUC reveals that Model B learned something useful; accuracy does not.

---

## When AUC Is Useful

**Imbalanced datasets:**
AUC is not affected by class proportions. A model with AUC = 0.9 on 1% positives is as impressive as AUC = 0.9 on 50% positives.

**When threshold is unknown:**
If the deployment threshold will be tuned later, AUC evaluates overall ranking quality.

**Comparing models:**
AUC provides a single number to compare models before choosing a threshold.

---

## When AUC Can Be Misleading

**When cost of errors differs:**
AUC treats FP and FN symmetrically. If false negatives are much worse than false positives (e.g., disease screening), AUC may not capture this.

**When you care about a specific operating point:**
If you will always use threshold = 0.5, precision/recall at that threshold may be more relevant than AUC.

**With highly imbalanced data (extreme):**
AUC can be high even when precision is very low. Consider AUC-PR (area under precision-recall curve) instead.

---

## AUC-PR: Area Under Precision-Recall Curve

For imbalanced datasets, the precision-recall curve is often more informative:

- **X-axis:** Recall = TP / (TP + FN)
- **Y-axis:** Precision = TP / (TP + FP)

AUC-PR focuses on the positive class. A random classifier has AUC-PR equal to the proportion of positives (e.g., 0.01 for 1% positives), making improvements easier to see.

---

## Multi-Class AUC

For multi-class problems:

**One-vs-Rest (OvR):**
Compute AUC for each class vs. all others. Average the AUCs (macro or weighted).

**One-vs-One (OvO):**
Compute AUC for each pair of classes. Average all pairwise AUCs.

---

## Interpreting AUC Values

- AUC > 0.9: Excellent discrimination
- 0.8 < AUC < 0.9: Good discrimination
- 0.7 < AUC < 0.8: Fair discrimination
- 0.6 < AUC < 0.7: Poor discrimination
- AUC < 0.6: Fail (close to random)

These are rough guidelines. Domain context matters.

---

## Computational Complexity

**Naive pairwise counting:** O(n_pos * n_neg)

**Efficient sorting-based:** O(n log n) where n is total samples

The sorting approach:
1. Sort by score
2. Count inversions using the sorted order
3. AUC = 1 - (inversions / total_pairs)
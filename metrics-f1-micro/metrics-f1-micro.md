## F1 Score Recap

The F1 score is the harmonic mean of precision and recall:

$$
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

For binary classification:
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)

F1 balances the trade-off between precision and recall in a single metric.

---

## The Multi-Class Challenge

With more than two classes, we need to aggregate F1 scores across all classes. There are three main approaches:

1. **Micro-averaged F1**
2. **Macro-averaged F1**
3. **Weighted F1**

Each gives different insights and is appropriate for different scenarios.

---

## Micro-Averaged F1

Micro-averaging computes **global** precision and recall by summing all true positives, false positives, and false negatives across all classes, then computes F1 from these totals.

$$
\text{Precision}_{\text{micro}} = \frac{\sum_i TP_i}{\sum_i (TP_i + FP_i)}
$$

$$
\text{Recall}_{\text{micro}} = \frac{\sum_i TP_i}{\sum_i (TP_i + FN_i)}
$$

$$
F1_{\text{micro}} = 2 \times \frac{\text{Precision}_{\text{micro}} \times \text{Recall}_{\text{micro}}}{\text{Precision}_{\text{micro}} + \text{Recall}_{\text{micro}}}
$$

---

## Key Insight: Micro F1 Equals Accuracy

For multi-class classification where each sample belongs to exactly one class:

$$
\sum_i TP_i = \text{total correct predictions}
$$

$$
\sum_i (TP_i + FP_i) = \sum_i (TP_i + FN_i) = \text{total samples}
$$

Therefore:

$$
\text{Precision}_{\text{micro}} = \text{Recall}_{\text{micro}} = \text{Accuracy}
$$

And:

$$
F1_{\text{micro}} = \text{Accuracy}
$$

Micro F1 and accuracy are the same for single-label multi-class problems.

---

## Worked Example

**3-class problem with 100 samples:**

Class A: 50 samples, 45 correct (TP=45, FN=5)

Class B: 30 samples, 24 correct (TP=24, FN=6)

Class C: 20 samples, 16 correct (TP=16, FN=4)

**Computing per-class metrics:**

For Class A:
- FP = samples predicted A but actually B or C
- Let us say FP_A = 3

For Class B: FP_B = 4

For Class C: FP_C = 8

**Micro-averaged precision:**

$$
\text{Precision}_{\text{micro}} = \frac{45 + 24 + 16}{(45+3) + (24+4) + (16+8)} = \frac{85}{100} = 0.85
$$

**Micro-averaged recall:**

$$
\text{Recall}_{\text{micro}} = \frac{45 + 24 + 16}{(45+5) + (24+6) + (16+4)} = \frac{85}{100} = 0.85
$$

**Micro F1:**

$$
F1_{\text{micro}} = 2 \times \frac{0.85 \times 0.85}{0.85 + 0.85} = 0.85
$$

This equals accuracy: 85 correct out of 100.

---

## Micro vs. Macro F1

**Micro F1:**
- Aggregates contributions from all classes
- Dominated by majority classes
- Good when you care about overall performance
- Equals accuracy for single-label classification

**Macro F1:**
- Computes F1 for each class, then averages
- Each class weighted equally regardless of size
- Good when all classes are equally important
- Sensitive to performance on minority classes

---

## When Classes Are Imbalanced

**Dataset:** 1000 samples
- Class A: 900 samples
- Class B: 80 samples
- Class C: 20 samples

**Model predicts everything as Class A:**

Micro F1 = 900/1000 = 0.90 (looks good!)

Macro F1:
- F1_A = high (most correct)
- F1_B = 0 (none predicted)
- F1_C = 0 (none predicted)
- Macro F1 = (high + 0 + 0) / 3 = low

Micro F1 hides the failure on minority classes. Macro F1 exposes it.

---

## The Formulas Side by Side

**Micro F1:**

$$
F1_{\text{micro}} = \frac{2 \times \sum_i TP_i}{2 \times \sum_i TP_i + \sum_i FP_i + \sum_i FN_i}
$$

**Macro F1:**

$$
F1_{\text{macro}} = \frac{1}{n} \sum_{i=1}^{n} F1_i
$$

**Weighted F1:**

$$
F1_{\text{weighted}} = \sum_{i=1}^{n} w_i \times F1_i
$$

where $w_i$ is the proportion of samples in class $i$.

---

## When to Use Micro F1

**All samples matter equally:**
If correctly classifying any sample is equally valuable, regardless of class.

**Class imbalance reflects real-world distribution:**
If the test distribution matches deployment, overall accuracy matters.

**Comparing to baselines:**
Micro F1 (accuracy) is a standard, widely understood metric.

---

## When to Use Macro F1 Instead

**All classes matter equally:**
In medical diagnosis, correctly identifying rare diseases is as important as common ones.

**Evaluating fairness:**
Ensure the model performs well across all groups, not just majority.

**Minority class performance is critical:**
When failing on rare classes is unacceptable.

---

## Multi-Label Classification

For multi-label problems (each sample can have multiple labels):

- Micro F1 aggregates TP, FP, FN across all samples and all labels
- Micro F1 no longer equals accuracy
- The formulas remain the same, but interpretation differs

**Example:** Sample has labels [A, B], model predicts [A, C]
- TP = 1 (A)
- FP = 1 (C)
- FN = 1 (B missed)

---

## Implementation Notes

**Computing micro F1:**

1. Build confusion matrix (n_classes x n_classes)
2. Sum diagonal for total TP
3. Sum each column for (TP + FP) per class
4. Sum each row for (TP + FN) per class
5. Apply micro formulas

**Efficient computation:**

For single-label classification:
- Micro F1 = (correct predictions) / (total predictions)
- Just count matches between predictions and labels

---

## Common Pitfalls

**Confusing micro and macro:**
They can give very different values on imbalanced data. Always specify which you are using.

**Ignoring class distribution:**
High micro F1 can mask poor minority class performance.

**Not reporting both:**
Best practice is to report both micro and macro (or weighted) to give a complete picture.
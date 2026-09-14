## The Problem with Simple Agreement

When two raters (or a model and ground truth) classify items, we might measure agreement by the percentage of items they agree on. But this raw agreement rate is misleading because some agreement happens **by chance**.

**Example:** Two raters classify 100 images as "cat" or "dog". If both raters randomly guess "cat" 50% of the time, they will agree about 50% of the time purely by chance (both say cat 25%, both say dog 25%).

A raw agreement of 60% sounds decent, but if chance agreement is 50%, the raters are barely doing better than random.

**Cohen's Kappa** accounts for chance agreement, measuring how much better the agreement is than random.

---

## The Formula

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

where:
- $p_o$ = observed agreement (proportion of items where raters agree)
- $p_e$ = expected agreement by chance
- $1 - p_e$ = maximum possible agreement beyond chance

**Interpretation:** Kappa measures the proportion of agreement beyond chance, relative to the maximum possible agreement beyond chance.

---

## Computing Observed Agreement

$p_o$ is straightforward: count how often the raters agree and divide by total items.

**Example confusion matrix:**

Rater B says Cat: Rater A says Cat = 20, Rater A says Dog = 10

Rater B says Dog: Rater A says Cat = 5, Rater A says Dog = 15

Total = 50 items

Agreements: 20 (both Cat) + 15 (both Dog) = 35

$p_o = 35 / 50 = 0.70$

---

## Computing Expected Agreement

$p_e$ is the agreement we would expect if both raters made independent random guesses with their observed marginal distributions.

**Step 1: Compute marginal probabilities**

Rater A says Cat: 20 + 5 = 25 out of 50 = 0.50

Rater A says Dog: 10 + 15 = 25 out of 50 = 0.50

Rater B says Cat: 20 + 10 = 30 out of 50 = 0.60

Rater B says Dog: 5 + 15 = 20 out of 50 = 0.40

**Step 2: Compute chance agreement for each category**

Chance both say Cat: $0.50 \times 0.60 = 0.30$

Chance both say Dog: $0.50 \times 0.40 = 0.20$

**Step 3: Sum chance agreements**

$p_e = 0.30 + 0.20 = 0.50$

---

## Computing Kappa

Using our example:

$p_o = 0.70$

$p_e = 0.50$

$$
\kappa = \frac{0.70 - 0.50}{1 - 0.50} = \frac{0.20}{0.50} = 0.40
$$

Despite 70% raw agreement, kappa is only 0.40 because half of that agreement was expected by chance.

---

## Interpreting Kappa Values

$$
-1 \leq \kappa \leq 1
$$

**$\kappa = 1$:** Perfect agreement. Raters agree on every item.

**$\kappa = 0$:** Agreement equals chance. No better than random guessing.

**$\kappa < 0$:** Agreement worse than chance. Raters systematically disagree.

**Common interpretation guidelines:**

- $\kappa > 0.80$: Almost perfect agreement
- $0.60 < \kappa \leq 0.80$: Substantial agreement
- $0.40 < \kappa \leq 0.60$: Moderate agreement
- $0.20 < \kappa \leq 0.40$: Fair agreement
- $0.00 < \kappa \leq 0.20$: Slight agreement
- $\kappa \leq 0.00$: Poor agreement (at or below chance)

---

## A Complete Worked Example

**Data:** 100 items classified by Model and Human

Model predicts Positive: Human says Positive = 40, Human says Negative = 15

Model predicts Negative: Human says Positive = 10, Human says Negative = 35

**Step 1: Observed agreement**

$p_o = (40 + 35) / 100 = 0.75$

**Step 2: Marginal probabilities**

Model Positive: $(40 + 15) / 100 = 0.55$

Model Negative: $(10 + 35) / 100 = 0.45$

Human Positive: $(40 + 10) / 100 = 0.50$

Human Negative: $(15 + 35) / 100 = 0.50$

**Step 3: Expected agreement**

$p_e = (0.55 \times 0.50) + (0.45 \times 0.50) = 0.275 + 0.225 = 0.50$

**Step 4: Kappa**

$\kappa = (0.75 - 0.50) / (1 - 0.50) = 0.25 / 0.50 = 0.50$

This indicates moderate agreement.

---

## Kappa for Multi-Class Problems

The formula extends naturally to more than two classes:

$$
p_e = \sum_{k=1}^{K} p_{A,k} \times p_{B,k}
$$

where $p_{A,k}$ is the proportion of items Rater A assigns to class $k$.

**Example with 3 classes:**

If Rater A's distribution is [0.3, 0.5, 0.2] and Rater B's is [0.4, 0.4, 0.2]:

$p_e = (0.3 \times 0.4) + (0.5 \times 0.4) + (0.2 \times 0.2)$

$p_e = 0.12 + 0.20 + 0.04 = 0.36$

---

## When to Use Kappa

**Comparing model to ground truth:**
Kappa measures classification agreement accounting for class imbalance.

**Inter-rater reliability:**
When multiple humans label the same data, kappa quantifies consistency.

**Imbalanced datasets:**
Accuracy is misleading when classes are imbalanced. A model predicting all "negative" on 95% negative data gets 95% accuracy but contributes nothing. Kappa penalizes this.

**Ordinal data:**
Weighted kappa (not covered here) extends to ordered categories where some disagreements are worse than others.

---

## Kappa vs. Accuracy

**Example of why accuracy is misleading:**

Dataset: 95 negatives, 5 positives

Model predicts all negative:
- Accuracy = 95%
- Kappa = 0 (no better than chance)

The model learned nothing useful. Kappa reveals this; accuracy hides it.

---

## Limitations of Kappa

**Prevalence dependence:**
Kappa can be low even with high agreement when one class dominates. This is sometimes called the "kappa paradox."

**No partial credit:**
Standard kappa treats all disagreements equally. Confusing class A with class B is the same as confusing A with class C.

**Two raters only:**
Cohen's kappa is for exactly two raters. For more raters, use Fleiss' kappa.

**Symmetric:**
Kappa treats both raters equally. If one is ground truth and one is a model, this symmetry may not match the problem structure.

---

## Relationship to Other Metrics

**Matthews Correlation Coefficient (MCC):**
For binary classification, MCC and kappa are closely related. MCC is often preferred in ML because it handles imbalanced data well.

**F1 Score:**
F1 focuses on the positive class. Kappa considers all classes symmetrically.

**AUC-ROC:**
Measures ranking quality, not agreement at a fixed threshold. Different use cases.
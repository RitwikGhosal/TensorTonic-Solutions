## What Is Gini Impurity?

Gini impurity measures the probability of **incorrectly classifying** a randomly chosen sample if it were labeled randomly according to the class distribution in the node.

**Low Gini:** Most samples belong to one class (pure node)

**High Gini:** Samples are evenly spread across classes (impure node)

Decision trees use Gini impurity to evaluate potential splits and choose the one that creates the purest child nodes.

---

## The Gini Impurity Formula

For a node with $C$ classes, where $p_i$ is the proportion of samples in class $i$:

$$
\text{Gini} = 1 - \sum_{i=1}^{C} p_i^2
$$

Alternatively written as:

$$
\text{Gini} = \sum_{i=1}^{C} p_i (1 - p_i)
$$

Both forms are equivalent.

---

## Understanding the Formula

The term $\sum p_i^2$ is the probability of correctly classifying a sample if we:
1. Pick a random sample from the node
2. Classify it according to the node's class distribution

$1 - \sum p_i^2$ is therefore the probability of **misclassification**.

---

## Binary Classification Examples

For 2 classes with proportions $(p, 1-p)$:

$$
\text{Gini} = 1 - p^2 - (1-p)^2 = 2p(1-p)
$$

**Example 1: Perfect purity**

All samples class A: $p = 1.0$

$\text{Gini} = 1 - 1^2 - 0^2 = 1 - 1 = 0$

Gini is 0. Pure node.

**Example 2: Maximum impurity**

Half class A, half class B: $p = 0.5$

$\text{Gini} = 1 - 0.5^2 - 0.5^2 = 1 - 0.25 - 0.25 = 0.5$

Gini is 0.5 (maximum for binary).

**Example 3: Moderate impurity**

70% class A, 30% class B: $p = 0.7$

$\text{Gini} = 1 - 0.7^2 - 0.3^2 = 1 - 0.49 - 0.09 = 0.42$

---

## Multi-Class Example

**Node with 100 samples:**
- Class A: 60 samples ($p_A = 0.6$)
- Class B: 25 samples ($p_B = 0.25$)
- Class C: 15 samples ($p_C = 0.15$)

$$
\text{Gini} = 1 - (0.6^2 + 0.25^2 + 0.15^2)
$$

$= 1 - (0.36 + 0.0625 + 0.0225)$

$= 1 - 0.445 = 0.555$

---

## Step-by-Step: Computing Gini Impurity

**Given labels:** [A, A, B, A, B, B, A, A]

**Step 1: Count each class**
- A: 5
- B: 3
- Total: 8

**Step 2: Compute proportions**
- $p_A = 5/8 = 0.625$
- $p_B = 3/8 = 0.375$

**Step 3: Square proportions**
- $p_A^2 = 0.625^2 = 0.391$
- $p_B^2 = 0.375^2 = 0.141$

**Step 4: Sum and subtract from 1**

$\text{Gini} = 1 - (0.391 + 0.141) = 1 - 0.532 = 0.468$

---

## Gini Impurity Bounds

**Minimum Gini:** 0 (all samples same class)

**Maximum Gini for C classes:** $1 - \frac{1}{C} = \frac{C-1}{C}$

- 2 classes: max = 0.5
- 3 classes: max = 0.667
- 4 classes: max = 0.75
- 10 classes: max = 0.9

Maximum occurs when all classes have equal proportions.

---

## Gini Gain (For Splitting)

When evaluating a split, compute the **weighted average Gini** of child nodes:

$$
\text{Gini}_{\text{split}} = \frac{n_{\text{left}}}{n_{\text{total}}} \text{Gini}_{\text{left}} + \frac{n_{\text{right}}}{n_{\text{total}}} \text{Gini}_{\text{right}}
$$

**Gini Gain:**

$$
\text{Gain} = \text{Gini}_{\text{parent}} - \text{Gini}_{\text{split}}
$$

Choose the split that maximizes Gini gain (biggest impurity reduction).

---

## Worked Example: Evaluating a Split

**Parent node:** 100 samples, 60 class A, 40 class B

$\text{Gini}_{\text{parent}} = 1 - 0.6^2 - 0.4^2 = 1 - 0.36 - 0.16 = 0.48$

**Split on feature X:**

Left child: 50 samples, 45 class A, 5 class B

$\text{Gini}_{\text{left}} = 1 - 0.9^2 - 0.1^2 = 1 - 0.81 - 0.01 = 0.18$

Right child: 50 samples, 15 class A, 35 class B

$\text{Gini}_{\text{right}} = 1 - 0.3^2 - 0.7^2 = 1 - 0.09 - 0.49 = 0.42$

**Weighted Gini after split:**

$\text{Gini}_{\text{split}} = \frac{50}{100} \times 0.18 + \frac{50}{100} \times 0.42 = 0.09 + 0.21 = 0.30$

**Gini Gain:**

$\text{Gain} = 0.48 - 0.30 = 0.18$

This is a good split (significant impurity reduction).

---

## Gini vs. Entropy

**Gini Impurity:**

$$
\text{Gini} = 1 - \sum_i p_i^2
$$

**Entropy:**

$$
H = -\sum_i p_i \log_2(p_i)
$$

**Comparison:**

- Gini is computationally simpler (no logarithms)
- Gini ranges [0, 0.5] for binary; Entropy ranges [0, 1]
- Both are concave functions maximized at uniform distribution
- Empirically produce very similar trees
- Gini is slightly faster, entropy is slightly more sensitive

**Default choice:** Most implementations (including scikit-learn) use Gini by default.

---

## The Gini Curve (Binary Case)

For binary classification with proportion $p$:

$\text{Gini}(p) = 2p(1-p)$

- $\text{Gini}(0) = 0$
- $\text{Gini}(0.5) = 0.5$ (maximum)
- $\text{Gini}(1) = 0$

The curve is a downward parabola, symmetric around $p = 0.5$.

---

## Edge Cases

**Single class:**

Labels: [A, A, A, A, A]

$p_A = 1.0$

$\text{Gini} = 1 - 1^2 = 0$

**Single sample:**

Labels: [B]

$p_B = 1.0$

$\text{Gini} = 0$

**Empty node:**

No samples: Gini is undefined (avoid this case in practice).

---

## Why "Gini"?

Named after Italian statistician Corrado Gini who developed the Gini coefficient for measuring inequality (1912). The formulas are related:

**Gini coefficient (inequality):** Measures how unequal a distribution is

**Gini impurity:** Measures how mixed classes are

Both capture the idea of "concentration" vs. "spread" in a distribution.

---

## Using Gini in CART

CART (Classification and Regression Trees) uses Gini impurity:

1. For each feature and each possible split point:
   - Compute Gini impurity of resulting child nodes
   - Compute weighted average Gini

2. Choose the split with minimum weighted Gini (maximum Gini gain)

3. Repeat recursively until stopping criteria met

This greedy approach builds the tree top-down, always choosing the locally optimal split.

---

## Implementation Notes

**Computing Gini:**

1. Count samples per class
2. Compute proportions: $p_i = \text{count}_i / \text{total}$
3. Square each proportion
4. Sum the squares
5. Subtract from 1

**Vectorized:** If labels are one-hot encoded, $\sum p_i^2 = ||p||_2^2$ where $p$ is the proportion vector.

**Numerical stability:** Gini involves only squares, no logarithms. Very stable numerically.
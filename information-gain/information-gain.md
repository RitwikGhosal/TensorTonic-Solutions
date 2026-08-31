## What Is Information Gain?

Information Gain measures how much a split **reduces uncertainty** about the class labels. It is the primary criterion used in ID3 and C4.5 decision tree algorithms.

$$
\text{Information Gain} = H(\text{parent}) - H(\text{children})
$$

where $H$ is entropy. Higher information gain means a better split.

---

## The Formula

For a parent node with entropy $H(S)$ that is split into children $S_1, S_2, ..., S_k$:

$$
\text{IG}(S, A) = H(S) - \sum_{i=1}^{k} \frac{|S_i|}{|S|} H(S_i)
$$

where:
- $S$ is the set of samples at the parent
- $A$ is the attribute/feature used for splitting
- $S_i$ is the subset of samples in child $i$
- $|S_i|/|S|$ is the proportion of samples going to child $i$

---

## Step-by-Step Process

**Step 1:** Compute entropy of the parent node

**Step 2:** For each possible split:
- Compute entropy of each child node
- Compute weighted average entropy of children
- Information Gain = Parent entropy - Weighted child entropy

**Step 3:** Choose the split with maximum information gain

---

## Worked Example

**Parent node:** 14 samples
- 9 "Yes" labels
- 5 "No" labels

**Parent entropy:**

$p_{yes} = 9/14 = 0.643$, $p_{no} = 5/14 = 0.357$

$H(\text{parent}) = -0.643 \log_2(0.643) - 0.357 \log_2(0.357)$

$= -0.643 \times (-0.637) - 0.357 \times (-1.486)$

$= 0.410 + 0.530 = 0.940$

**Split on feature "Outlook":**

Sunny: 5 samples (2 Yes, 3 No)
$H = -0.4 \log_2(0.4) - 0.6 \log_2(0.6) = 0.971$

Overcast: 4 samples (4 Yes, 0 No)
$H = -1 \log_2(1) - 0 = 0$ (pure!)

Rainy: 5 samples (3 Yes, 2 No)
$H = -0.6 \log_2(0.6) - 0.4 \log_2(0.4) = 0.971$

**Weighted child entropy:**

$H(\text{children}) = \frac{5}{14}(0.971) + \frac{4}{14}(0) + \frac{5}{14}(0.971)$

$= 0.347 + 0 + 0.347 = 0.694$

**Information Gain:**

$\text{IG} = 0.940 - 0.694 = 0.246$

---

## Comparing Multiple Splits

Evaluate information gain for each candidate feature:

- Split on "Outlook": IG = 0.246
- Split on "Temperature": IG = 0.029
- Split on "Humidity": IG = 0.151
- Split on "Wind": IG = 0.048

**Best split:** "Outlook" with highest IG (0.246)

---

## Why Information Gain Works

**High IG splits:**
- Create purer children (lower entropy)
- Separate classes effectively
- Move toward leaf nodes with single classes

**Low IG splits:**
- Children have similar entropy to parent
- Classes remain mixed
- Not helpful for classification

---

## Information Gain for Binary Splits

For numeric features or binary splits (left/right):

$$
\text{IG} = H(S) - \frac{|S_L|}{|S|} H(S_L) - \frac{|S_R|}{|S|} H(S_R)
$$

**Example:** Split on "Age > 30"

Parent: 100 samples, H = 0.88

Left (Age <= 30): 40 samples, H = 0.72

Right (Age > 30): 60 samples, H = 0.65

$\text{IG} = 0.88 - \frac{40}{100}(0.72) - \frac{60}{100}(0.65)$

$= 0.88 - 0.288 - 0.390 = 0.202$

---

## Properties of Information Gain

**Non-negative:** IG is always $\geq 0$ (entropy cannot increase from splitting)

**Zero IG:** Split does not separate classes at all

**Maximum IG:** Equals parent entropy when children are perfectly pure

**Bounded:** $0 \leq \text{IG} \leq H(\text{parent})$

---

## Bias Toward Many-Valued Attributes

Information gain has a known bias: it favors attributes with many values.

**Example:** An ID attribute with unique values for each sample gives perfect splits (each child has one sample), achieving maximum IG but learning nothing useful.

**Solution:** Use **Gain Ratio** (C4.5):

$$
\text{Gain Ratio} = \frac{\text{IG}(S, A)}{\text{SplitInfo}(S, A)}
$$

where SplitInfo penalizes attributes with many branches.

---

## Information Gain vs. Gini Gain

**Information Gain (uses entropy):**
- Used in ID3, C4.5
- Slightly more computationally expensive (logarithms)
- More sensitive to probability distribution changes

**Gini Gain (uses Gini impurity):**
- Used in CART
- Faster to compute
- Default in scikit-learn

Both produce similar trees in practice.

---

## Recursive Application

Decision trees apply information gain recursively:

1. At root: Find best split for all data
2. Create children based on split
3. At each child: Find best split for that subset
4. Repeat until stopping criteria (max depth, min samples, pure nodes)

Each split maximizes local information gain, building the tree greedily.

---

## Handling Continuous Features

For continuous features, consider all possible threshold splits:

1. Sort samples by feature value
2. Consider splits between each pair of consecutive values
3. Compute IG for each threshold
4. Choose threshold with maximum IG

Efficient: Only need to consider $n-1$ thresholds for $n$ unique values.
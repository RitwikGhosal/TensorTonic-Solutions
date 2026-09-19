## What Is a Random Forest?

A Random Forest is an **ensemble** of decision trees. Instead of relying on a single tree, it trains many trees on different subsets of the data and combines their predictions.

The key insight: individual trees may overfit or make errors, but averaging across many diverse trees reduces variance and improves generalization.

---

## How Random Forest Makes Predictions

**For classification:**

Each tree votes for a class. The final prediction is the **majority vote** across all trees.

$$
\hat{y} = \text{mode}(\hat{y}_1, \hat{y}_2, ..., \hat{y}_T)
$$

where $\hat{y}_t$ is the prediction of tree $t$ and $T$ is the total number of trees.

**For regression:**

Each tree outputs a value. The final prediction is the **average** across all trees.

$$
\hat{y} = \frac{1}{T} \sum_{t=1}^{T} \hat{y}_t
$$

---

## The Voting Process for Classification

**Step 1:** Pass the input sample to each tree in the forest

**Step 2:** Each tree traverses its structure and outputs a class prediction

**Step 3:** Count votes for each class

**Step 4:** Return the class with the most votes

If there is a tie, common strategies include:
- Random selection among tied classes
- Choose the class with lower index
- Use probability estimates to break ties

---

## Worked Example: Classification

**Setup:** 5 trees, 3 classes (A, B, C)

**Tree predictions for sample X:**
- Tree 1: Class A
- Tree 2: Class B
- Tree 3: Class A
- Tree 4: Class A
- Tree 5: Class B

**Vote count:**
- Class A: 3 votes
- Class B: 2 votes
- Class C: 0 votes

**Final prediction:** Class A (majority with 3 out of 5 votes)

---

## Worked Example: Regression

**Setup:** 5 trees predicting house prices

**Tree predictions for sample X:**
- Tree 1: $250,000
- Tree 2: $270,000
- Tree 3: $245,000
- Tree 4: $260,000
- Tree 5: $255,000

**Final prediction:**

$$
\hat{y} = \frac{250000 + 270000 + 245000 + 260000 + 255000}{5} = \frac{1280000}{5} = 256000
$$

The predicted price is $256,000.

---

## Why Voting Works: Wisdom of Crowds

Consider a simple model: each tree has 60% accuracy (better than random guessing at 50%).

**Single tree:** 60% chance of correct prediction

**Majority of 5 trees:** Need at least 3 correct

$$
P(\text{majority correct}) = \sum_{k=3}^{5} \binom{5}{k} (0.6)^k (0.4)^{5-k}
$$

$= \binom{5}{3}(0.6)^3(0.4)^2 + \binom{5}{4}(0.6)^4(0.4)^1 + \binom{5}{5}(0.6)^5$

$= 10(0.216)(0.16) + 5(0.1296)(0.4) + 1(0.07776)$

$= 0.3456 + 0.2592 + 0.07776 = 0.683$

**Result:** Ensemble accuracy 68.3% vs individual 60%

With more trees (assuming independence), accuracy approaches 100%.

---

## Soft Voting vs Hard Voting

**Hard voting:** Each tree casts one vote for its predicted class. Final prediction is the mode.

**Soft voting:** Each tree outputs class probabilities. Average the probabilities, then pick the class with highest average probability.

**Example of soft voting:**

Tree 1 probabilities: [0.7, 0.2, 0.1] for classes [A, B, C]

Tree 2 probabilities: [0.4, 0.5, 0.1]

Tree 3 probabilities: [0.6, 0.3, 0.1]

**Average probabilities:**
- Class A: $(0.7 + 0.4 + 0.6)/3 = 0.567$
- Class B: $(0.2 + 0.5 + 0.3)/3 = 0.333$
- Class C: $(0.1 + 0.1 + 0.1)/3 = 0.1$

**Final prediction:** Class A (highest average probability)

Soft voting often performs better because it considers prediction confidence.

---

## Why Random Forests Work

**Bagging (Bootstrap Aggregating):**

Each tree is trained on a bootstrap sample (random sample with replacement) of the training data. This creates diversity among trees.

**Feature randomization:**

At each split, only a random subset of features is considered. This further decorrelates the trees.

**Variance reduction:**

Averaging over many decorrelated trees reduces variance without increasing bias. The expected error of the average is lower than the expected error of individuals.

$$
\text{Var}(\bar{X}) = \frac{\text{Var}(X)}{n} \text{ (for independent variables)}
$$

---

## Number of Trees

**More trees:**
- Better accuracy (diminishing returns after a point)
- More stable predictions
- Longer training and prediction time
- More memory usage

**Typical values:** 100 to 500 trees

**Rule of thumb:** Keep adding trees until out-of-bag error stabilizes.

---

## Handling Ties in Voting

When two or more classes have equal votes:

**Option 1:** Random selection
- Pick randomly among tied classes
- Non-deterministic

**Option 2:** Class with lower index
- Deterministic but arbitrary

**Option 3:** Use probabilities
- If trees output probabilities, average them to break ties
- More principled approach

**Option 4:** Weighted voting
- Give trees different weights (e.g., based on validation accuracy)
- Higher weight = more influence

---

## Out-of-Bag Predictions

Since each tree is trained on a bootstrap sample, about 37% of samples are "out-of-bag" (not used) for each tree.

**OOB prediction:** For each sample, aggregate votes only from trees that did not train on it.

This provides a built-in validation estimate without needing a separate validation set.

---

## Feature Importance from Voting

Random forests can estimate feature importance:

**Mean Decrease in Impurity:** Average reduction in impurity (Gini or entropy) across all splits on that feature, across all trees.

**Permutation Importance:** Shuffle the feature values and measure how much accuracy drops. Important features cause large drops.

---

## Comparing Voting Strategies

**Majority voting (mode):**
- Simple and intuitive
- Each tree has equal say
- Works well when trees have similar accuracy

**Weighted voting:**
- Better trees have more influence
- Requires estimating tree quality
- Can improve accuracy when tree quality varies

**Probability averaging (soft voting):**
- Uses more information (confidence)
- Often outperforms hard voting
- Requires trees that output probabilities

---

## Computational Considerations

**Prediction time:** $O(T \cdot d)$ where $T$ is number of trees and $d$ is average tree depth

**Parallelizable:** Trees are independent, so predictions can be computed in parallel

**Memory:** Must store all trees in memory

For a forest with 100 trees of depth 10 with 1000 leaves each, memory can be significant but usually manageable.
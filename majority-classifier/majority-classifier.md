## What Is a Majority Classifier?

A majority classifier (also called a "most frequent class" or "zero-rule" classifier) is the simplest possible classification model. It ignores all input features and always predicts the **most common class** in the training data.

$$
\hat{y} = \arg\max_c \sum_{i=1}^{n} \mathbb{1}[y_i = c]
$$

where $\mathbb{1}[\cdot]$ is the indicator function that equals 1 when the condition is true.

---

## Why Use a Majority Classifier?

**1. Baseline performance**

Any useful model should beat the majority classifier. If your model has 70% accuracy but the majority class represents 70% of the data, your model is not learning anything beyond the class distribution.

**2. Sanity check**

A quick way to verify your evaluation pipeline is working correctly.

**3. Handling edge cases**

When no features are available or all features are missing, fall back to the majority class.

---

## Step-by-Step Process

**Step 1:** Count the occurrences of each class in the training data

**Step 2:** Identify the class with the highest count

**Step 3:** Store this class as the prediction

**Step 4:** For any new input, predict this stored class (ignoring input features)

---

## Worked Example

**Training data labels:** [A, B, A, A, C, B, A, A, B, C]

**Step 1: Count classes**
- Class A: 5 occurrences
- Class B: 3 occurrences
- Class C: 2 occurrences
- Total: 10 samples

**Step 2: Find majority**

Class A has the most occurrences (5 out of 10).

**Step 3: Prediction rule**

For any input $x$, predict class A.

**Training accuracy:**
$5/10 = 50\%$

This is the expected accuracy of the majority classifier on the training set.

---

## Handling Ties

When two or more classes have the same highest count:

**Option 1:** Choose randomly among tied classes

**Option 2:** Choose the class with the smallest index or alphabetically first

**Option 3:** Use class priors if available

**Example with tie:**

Labels: [A, B, A, B, C, C]
- Class A: 2
- Class B: 2
- Class C: 2

All classes are tied. A deterministic implementation might choose A (first alphabetically).

---

## Probability Estimates

A majority classifier can also output class probabilities based on training frequencies:

$$
P(y = c) = \frac{\text{count}(c)}{n}
$$

**Example:**

Labels: [A, A, A, B, B, C]
- $P(A) = 3/6 = 0.5$
- $P(B) = 2/6 = 0.333$
- $P(C) = 1/6 = 0.167$

**Prediction:** Class A (highest probability)

**Probability output:** [0.5, 0.333, 0.167]

---

## Expected Accuracy

The expected accuracy of a majority classifier equals the proportion of the majority class:

$$
\text{Accuracy} = \frac{\text{count}(\text{majority class})}{n} = \max_c P(c)
$$

**Imbalanced datasets:** If 90% of samples are class A, the majority classifier achieves 90% accuracy without learning anything useful.

**Balanced datasets:** If classes are evenly distributed, majority classifier accuracy equals $1/C$ where $C$ is the number of classes.

---

## Majority Classifier as a Baseline

Always compare your model against the majority classifier:

**Scenario 1:** Binary classification with 95% negative, 5% positive
- Majority classifier accuracy: 95%
- Your model with 96% accuracy: marginal improvement
- Need to look at other metrics (precision, recall, F1)

**Scenario 2:** 3-class problem with equal distribution
- Majority classifier accuracy: 33%
- Your model with 80% accuracy: significant improvement

**Scenario 3:** Your model underperforms majority classifier
- Bug in your code or data
- Model is learning spurious patterns
- Features are not informative

---

## Relationship to Other Models

**Decision stump:** A one-level decision tree that makes one split. Majority classifier is a decision stump with no splits (just the root).

**Naive Bayes with uniform likelihood:** If all features have the same distribution across classes, Naive Bayes reduces to the majority classifier.

**Prior classifier:** The majority classifier essentially uses only the class prior $P(y)$ and ignores the likelihood $P(x|y)$.

---

## Multi-label Extension

For multi-label classification (where each sample can have multiple labels):

**Per-label majority:** For each label, predict 1 if that label appears in more than 50% of training samples, else 0.

**Example:**

Sample labels:
- Sample 1: [1, 0, 1]
- Sample 2: [1, 1, 0]
- Sample 3: [1, 0, 0]
- Sample 4: [0, 1, 0]

Label frequencies:
- Label 0: 3/4 = 75% (predict 1)
- Label 1: 2/4 = 50% (predict 0 or 1 depending on threshold)
- Label 2: 1/4 = 25% (predict 0)

---

## Computational Complexity

**Training:**
- Count occurrences: $O(n)$
- Find maximum: $O(C)$ where $C$ is number of classes
- Total: $O(n + C)$

**Prediction:**
- Return stored class: $O(1)$

The majority classifier is extremely efficient in both training and inference.

---

## When Majority Classifier Performs Well

**Highly imbalanced data:** When one class dominates (>90%), majority classifier is hard to beat without careful modeling.

**No signal in features:** If features are random noise, majority classifier is optimal.

**Early stopping in trees:** Decision trees often have majority classifier nodes as leaves.

---

## Limitations

**Ignores all features:** Cannot adapt to different inputs

**Blind to minority classes:** Never predicts rare classes

**Poor for balanced problems:** Only achieves $1/C$ accuracy when classes are equally distributed

**Not useful for decision making:** Provides no insight into why a prediction was made
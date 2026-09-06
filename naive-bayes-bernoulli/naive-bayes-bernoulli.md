## What Is Bernoulli Naive Bayes?

Bernoulli Naive Bayes is a classification algorithm for **binary features**. Each feature is either present (1) or absent (0). The model assumes features are independent given the class label and follow a Bernoulli distribution.

Common applications include:
- Text classification with binary word indicators (word present or not)
- Spam detection
- Document categorization

---

## The Probabilistic Model

For a sample $x = (x_1, x_2, ..., x_d)$ where each $x_i \in \{0, 1\}$:

$$
P(x | y = c) = \prod_{i=1}^{d} P(x_i | y = c)
$$

Each feature follows a Bernoulli distribution:

$$
P(x_i | y = c) = p_{ic}^{x_i} (1 - p_{ic})^{1 - x_i}
$$

where $p_{ic} = P(x_i = 1 | y = c)$ is the probability that feature $i$ is present given class $c$.

---

## The Classification Rule

Using Bayes' theorem, we classify based on the posterior probability:

$$
P(y = c | x) \propto P(y = c) \prod_{i=1}^{d} P(x_i | y = c)
$$

The predicted class is:

$$
\hat{y} = \arg\max_c P(y = c) \prod_{i=1}^{d} P(x_i | y = c)
$$

---

## Expanding the Likelihood

For Bernoulli features:

$$
P(x | y = c) = \prod_{i=1}^{d} p_{ic}^{x_i} (1 - p_{ic})^{1 - x_i}
$$

This means:
- If $x_i = 1$ (feature present): contribute $p_{ic}$
- If $x_i = 0$ (feature absent): contribute $1 - p_{ic}$

**Important:** Unlike Multinomial Naive Bayes, Bernoulli considers the **absence** of features as informative.

---

## Log-Probability Form

To avoid numerical underflow with many features, use log probabilities:

$$
\log P(y = c | x) = \log P(y = c) + \sum_{i=1}^{d} \left[ x_i \log p_{ic} + (1 - x_i) \log(1 - p_{ic}) \right]
$$

Simplifying:

$$
= \log P(y = c) + \sum_{i=1}^{d} \left[ x_i \log \frac{p_{ic}}{1 - p_{ic}} + \log(1 - p_{ic}) \right]
$$

---

## Parameter Estimation

**Class prior:**

$$
P(y = c) = \frac{N_c}{N}
$$

where $N_c$ is the number of samples in class $c$ and $N$ is total samples.

**Feature probability:**

$$
p_{ic} = P(x_i = 1 | y = c) = \frac{\text{count}(x_i = 1 \text{ in class } c)}{N_c}
$$

This is the fraction of class $c$ samples where feature $i$ is present.

---

## Laplace Smoothing

To avoid zero probabilities when a feature never appears in a class:

$$
p_{ic} = \frac{\text{count}(x_i = 1 \text{ in class } c) + \alpha}{N_c + 2\alpha}
$$

where $\alpha$ is the smoothing parameter (commonly $\alpha = 1$).

The denominator uses $2\alpha$ because there are 2 possible values (0 or 1) for each feature.

---

## Worked Example

**Training data:** 4 documents, 2 classes (spam/not spam), 3 binary features

- Doc 1: Features = [1, 0, 1], Class = spam
- Doc 2: Features = [1, 1, 0], Class = spam
- Doc 3: Features = [0, 1, 0], Class = not spam
- Doc 4: Features = [0, 0, 1], Class = not spam

**Step 1: Class priors**

$P(\text{spam}) = 2/4 = 0.5$

$P(\text{not spam}) = 2/4 = 0.5$

**Step 2: Feature probabilities (with Laplace smoothing, $\alpha = 1$)**

For spam (2 samples):
- $p_{1,\text{spam}} = (2 + 1)/(2 + 2) = 3/4 = 0.75$
- $p_{2,\text{spam}} = (1 + 1)/(2 + 2) = 2/4 = 0.5$
- $p_{3,\text{spam}} = (1 + 1)/(2 + 2) = 2/4 = 0.5$

For not spam (2 samples):
- $p_{1,\text{not spam}} = (0 + 1)/(2 + 2) = 1/4 = 0.25$
- $p_{2,\text{not spam}} = (1 + 1)/(2 + 2) = 2/4 = 0.5$
- $p_{3,\text{not spam}} = (1 + 1)/(2 + 2) = 2/4 = 0.5$

---

**Step 3: Classify a new document**

New document: $x = [1, 0, 1]$

**For spam:**

$P(\text{spam} | x) \propto 0.5 \times 0.75^1 \times (1-0.5)^1 \times 0.5^1$

$= 0.5 \times 0.75 \times 0.5 \times 0.5 = 0.09375$

**For not spam:**

$P(\text{not spam} | x) \propto 0.5 \times 0.25^1 \times (1-0.5)^1 \times 0.5^1$

$= 0.5 \times 0.25 \times 0.5 \times 0.5 = 0.03125$

**Normalization:**

$P(\text{spam} | x) = 0.09375 / (0.09375 + 0.03125) = 0.75$

$P(\text{not spam} | x) = 0.03125 / (0.09375 + 0.03125) = 0.25$

**Prediction:** spam (75% probability)

---

## Why Feature Absence Matters

In Bernoulli Naive Bayes, the absence of a feature provides information.

**Example:** Classifying emails

If the word "free" appears in 80% of spam but only 10% of legitimate emails:
- Presence of "free" strongly suggests spam
- **Absence** of "free" suggests legitimate email

The term $(1 - p_{ic})$ captures this information.

---

## Bernoulli vs Multinomial Naive Bayes

**Bernoulli:**
- Features are binary (present/absent)
- Explicitly models feature absence
- Document represented as binary word vector
- Better for short documents

**Multinomial:**
- Features are counts (word frequencies)
- Only considers present features
- Document represented as word count vector
- Better for longer documents

---

## The Naive Bayes Assumption

"Naive" refers to the assumption that features are **conditionally independent** given the class:

$$
P(x_1, x_2, ..., x_d | y) = \prod_{i=1}^{d} P(x_i | y)
$$

This assumption is often violated in practice (words are correlated), but Naive Bayes still works well empirically.

---

## Decision Boundary

Bernoulli Naive Bayes defines a **linear decision boundary** in the binary feature space.

The log-odds ratio between two classes:

$$
\log \frac{P(y=1|x)}{P(y=0|x)} = \log \frac{P(y=1)}{P(y=0)} + \sum_{i=1}^{d} x_i \log \frac{p_{i1}(1-p_{i0})}{p_{i0}(1-p_{i1})} + \text{const}
$$

This is linear in $x$, making Bernoulli Naive Bayes a linear classifier.

---

## Handling Non-Binary Features

If features are not naturally binary:

**Thresholding:** Convert numeric features to binary using a threshold
- $x_i = 1$ if value > threshold, else $x_i = 0$

**Binarization:** For text, use presence/absence instead of counts
- $x_i = 1$ if word appears, regardless of frequency

**Multiple thresholds:** Create multiple binary features from one numeric feature

---

## Computational Complexity

**Training:**
- Compute class counts: $O(N)$
- Compute feature counts per class: $O(N \times d)$
- Total: $O(N \times d)$

**Prediction:**
- Compute log-probability for each class: $O(d)$
- For all classes: $O(C \times d)$

Very efficient for high-dimensional sparse data.

---

## Advantages and Limitations

**Advantages:**
- Simple and fast
- Works well with small training data
- Handles high-dimensional data
- Interpretable probabilities
- Feature absence is informative

**Limitations:**
- Assumes feature independence
- Binary features only (need preprocessing for continuous)
- Sensitive to imbalanced classes (use balanced priors)
- Cannot capture feature interactions
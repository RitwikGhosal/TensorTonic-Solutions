## What Is a Bernoulli Distribution?

The Bernoulli distribution is the simplest discrete probability distribution. It models a single experiment with exactly **two possible outcomes**: success (1) or failure (0).

Named after Swiss mathematician Jacob Bernoulli, this distribution is the building block for many other distributions, including the Binomial, Geometric, and Negative Binomial.

---

## Real-World Examples

Many real-world scenarios follow a Bernoulli distribution:

- Flipping a coin (heads or tails)
- A patient responding to treatment (responds or does not respond)
- A website visitor clicking an ad (clicks or does not click)
- A manufactured item being defective (defective or not defective)
- A student passing an exam (pass or fail)
- An email being spam (spam or not spam)

Any situation with a binary outcome and a fixed probability of success can be modeled as Bernoulli.

---

## The Parameter $p$

The Bernoulli distribution has a single parameter:

$$
p = P(X = 1) = \text{probability of success}
$$

The probability of failure is:

$$
q = 1 - p = P(X = 0)
$$

**Constraints:**
- $0 \leq p \leq 1$
- $p$ and $q$ must sum to 1

---

## The Probability Mass Function (PMF)

The PMF gives the probability of each possible outcome:

$$
P(X = x) = p^x (1-p)^{1-x}
$$

where $x \in \{0, 1\}$.

**Expanding this formula:**

When $x = 1$ (success):

$$
P(X = 1) = p^1 (1-p)^{1-1} = p \cdot (1-p)^0 = p \cdot 1 = p
$$

When $x = 0$ (failure):

$$
P(X = 0) = p^0 (1-p)^{1-0} = 1 \cdot (1-p) = 1 - p
$$

---

## Alternative PMF Notation

The PMF can also be written as a piecewise function:

$$
P(X = x) = \begin{cases} p & \text{if } x = 1 \\ 1 - p & \text{if } x = 0 \\ 0 & \text{otherwise} \end{cases}
$$

Both notations are equivalent and commonly used.

---

## Worked Example: Coin Flip

**Setup:** A fair coin has $p = 0.5$ for heads.

**PMF values:**

$P(X = 1) = 0.5$ (probability of heads)

$P(X = 0) = 1 - 0.5 = 0.5$ (probability of tails)

**Verification:** $P(X = 0) + P(X = 1) = 0.5 + 0.5 = 1$ ✓

---

## Worked Example: Biased Coin

**Setup:** A biased coin has $p = 0.7$ for heads.

**PMF values:**

$P(X = 1) = 0.7$ (probability of heads)

$P(X = 0) = 1 - 0.7 = 0.3$ (probability of tails)

**Using the formula:**

$P(X = 1) = 0.7^1 \cdot 0.3^0 = 0.7 \cdot 1 = 0.7$

$P(X = 0) = 0.7^0 \cdot 0.3^1 = 1 \cdot 0.3 = 0.3$

---

## Worked Example: Click-Through Rate

**Setup:** An ad has a 3% click-through rate, so $p = 0.03$.

**PMF values:**

$P(X = 1) = 0.03$ (probability user clicks)

$P(X = 0) = 0.97$ (probability user does not click)

For any single user viewing the ad, there is a 3% chance they click.

---

## Expected Value (Mean)

The expected value of a Bernoulli random variable is:

$$
E[X] = \sum_{x} x \cdot P(X = x) = 0 \cdot (1-p) + 1 \cdot p = p
$$

The mean equals the probability of success.

**Interpretation:** If you repeat the experiment many times, the average outcome converges to $p$.

**Example:** For a fair coin ($p = 0.5$), the expected value is 0.5. Over many flips, roughly half will be heads.

---

## Variance

The variance measures the spread of outcomes:

$$
\text{Var}(X) = E[X^2] - (E[X])^2
$$

First, compute $E[X^2]$:

$$
E[X^2] = 0^2 \cdot (1-p) + 1^2 \cdot p = p
$$

Note that $E[X^2] = E[X] = p$ because $X$ only takes values 0 and 1.

Therefore:

$$
\text{Var}(X) = p - p^2 = p(1-p)
$$

---

## Understanding the Variance

The variance $p(1-p)$ is maximized when $p = 0.5$:

$$
\text{Var}(X) = 0.5 \times 0.5 = 0.25
$$

This makes intuitive sense: maximum uncertainty occurs when success and failure are equally likely.

As $p$ approaches 0 or 1, the variance approaches 0:
- If $p = 0.01$: $\text{Var}(X) = 0.01 \times 0.99 = 0.0099$ (almost always 0)
- If $p = 0.99$: $\text{Var}(X) = 0.99 \times 0.01 = 0.0099$ (almost always 1)

---

## Standard Deviation

The standard deviation is:

$$
\sigma = \sqrt{\text{Var}(X)} = \sqrt{p(1-p)}
$$

**Example:** For $p = 0.5$:

$$
\sigma = \sqrt{0.5 \times 0.5} = \sqrt{0.25} = 0.5
$$

---

## Cumulative Distribution Function (CDF)

The CDF gives the probability that $X$ is less than or equal to a value:

$$
F(x) = P(X \leq x) = \begin{cases} 0 & \text{if } x < 0 \\ 1 - p & \text{if } 0 \leq x < 1 \\ 1 & \text{if } x \geq 1 \end{cases}
$$

The CDF is a step function with jumps at $x = 0$ and $x = 1$.

---

## Moment Generating Function

The moment generating function (MGF) is:

$$
M_X(t) = E[e^{tX}] = e^{t \cdot 0} (1-p) + e^{t \cdot 1} p = (1-p) + pe^t
$$

The MGF can be used to derive moments:
- $M'_X(0) = E[X] = p$
- $M''_X(0) = E[X^2] = p$

---

## Relationship to Other Distributions

**Binomial distribution:**

The sum of $n$ independent Bernoulli trials with the same $p$ follows a Binomial$(n, p)$ distribution:

$$
Y = \sum_{i=1}^{n} X_i \sim \text{Binomial}(n, p)
$$

A Bernoulli distribution is a special case: Binomial$(1, p)$.

**Geometric distribution:**

The number of Bernoulli trials until the first success follows a Geometric distribution.

**Beta distribution:**

The Beta distribution is the conjugate prior for the Bernoulli parameter $p$ in Bayesian inference.

---

## Multiple Independent Bernoulli Trials

If $X_1, X_2, ..., X_n$ are independent Bernoulli$(p)$ random variables:

**Sum:**

$$
\sum_{i=1}^{n} X_i \sim \text{Binomial}(n, p)
$$

**Sample mean:**

$$
\bar{X} = \frac{1}{n} \sum_{i=1}^{n} X_i
$$

$E[\bar{X}] = p$ and $\text{Var}(\bar{X}) = \frac{p(1-p)}{n}$

As $n \to \infty$, $\bar{X} \to p$ (Law of Large Numbers).

---

## Maximum Likelihood Estimation

Given $n$ observations $x_1, x_2, ..., x_n$ from a Bernoulli distribution, the MLE of $p$ is:

$$
\hat{p} = \frac{1}{n} \sum_{i=1}^{n} x_i = \frac{\text{number of successes}}{n}
$$

This is simply the sample proportion of successes.

**Derivation:**

Likelihood: $L(p) = \prod_{i=1}^{n} p^{x_i}(1-p)^{1-x_i} = p^{\sum x_i}(1-p)^{n - \sum x_i}$

Log-likelihood: $\ell(p) = \sum x_i \log p + (n - \sum x_i) \log(1-p)$

Setting $\frac{d\ell}{dp} = 0$ and solving gives $\hat{p} = \frac{\sum x_i}{n}$.

---

## Bernoulli in Machine Learning

The Bernoulli distribution appears throughout machine learning:

**Binary classification:**
The predicted probability $\hat{p}$ represents $P(Y = 1 | X)$.

**Logistic regression:**
Models $p = \sigma(w^T x)$ where $\sigma$ is the sigmoid function.

**Binary cross-entropy loss:**

$$
L = -[y \log \hat{p} + (1-y) \log(1-\hat{p})]
$$

This is the negative log-likelihood of a Bernoulli distribution.

**Bernoulli Naive Bayes:**
Assumes features are binary and independently Bernoulli distributed given the class.

**Dropout:**
Each neuron is kept with probability $p$, following a Bernoulli distribution.

---

## Properties Summary

- **Support:** $\{0, 1\}$
- **Parameter:** $p \in [0, 1]$
- **PMF:** $P(X = x) = p^x(1-p)^{1-x}$
- **Mean:** $E[X] = p$
- **Variance:** $\text{Var}(X) = p(1-p)$
- **Skewness:** $\frac{1-2p}{\sqrt{p(1-p)}}$
- **MGF:** $M_X(t) = (1-p) + pe^t$
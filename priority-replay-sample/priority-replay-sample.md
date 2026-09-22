## What Is Prioritized Experience Replay?

Prioritized Experience Replay (PER) is an enhancement to standard experience replay that samples transitions based on their **importance** rather than uniformly at random.

The key insight: not all experiences are equally useful for learning. Transitions with high TD error indicate surprising outcomes that the model can learn more from.

Introduced by Schaul et al. (2016) and used in many modern deep RL algorithms.

---

## Motivation

**Standard uniform replay:**

All transitions sampled with equal probability $P(i) = 1/N$.

**Problem:** Many transitions are already well-predicted and provide little learning signal. Rare, surprising transitions may be undersampled.

**Solution:** Sample transitions with high learning potential more frequently.

---

## Measuring Priority: TD Error

The most common priority measure is the magnitude of the TD error:

$$
\delta_i = |r_i + \gamma \max_{a'} Q(s_i', a') - Q(s_i, a_i)|
$$

**High $|\delta_i|$:** Large prediction error. The model was wrong about this transition. High learning potential.

**Low $|\delta_i|$:** Small prediction error. The model already predicts this well. Less to learn.

---

## Priority Computation

The priority of transition $i$ is:

$$
p_i = |\delta_i| + \epsilon
$$

where $\epsilon > 0$ is a small constant (e.g., $10^{-6}$) ensuring non-zero probability for all transitions.

**Alternative:** Use exponentiated TD error:

$$
p_i = (|\delta_i| + \epsilon)^\alpha
$$

where $\alpha$ controls how much prioritization affects sampling.

---

## Sampling Probability

The probability of sampling transition $i$:

$$
P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}
$$

**$\alpha = 0$:** Uniform sampling (no prioritization)

**$\alpha = 1$:** Full prioritization (proportional to priority)

**$\alpha \in (0, 1)$:** Interpolation between uniform and full prioritization

**Typical value:** $\alpha = 0.6$

---

## Worked Example

**Buffer with 5 transitions:**

- Transition 1: $|\delta_1| = 0.5$
- Transition 2: $|\delta_2| = 2.0$
- Transition 3: $|\delta_3| = 0.1$
- Transition 4: $|\delta_4| = 1.5$
- Transition 5: $|\delta_5| = 0.3$

**Parameters:** $\epsilon = 0.01$, $\alpha = 1.0$

**Priorities:**
- $p_1 = 0.5 + 0.01 = 0.51$
- $p_2 = 2.0 + 0.01 = 2.01$
- $p_3 = 0.1 + 0.01 = 0.11$
- $p_4 = 1.5 + 0.01 = 1.51$
- $p_5 = 0.3 + 0.01 = 0.31$

**Total:** $\sum p_i = 4.45$

**Sampling probabilities:**
- $P(1) = 0.51/4.45 = 0.115$ (11.5%)
- $P(2) = 2.01/4.45 = 0.452$ (45.2%)
- $P(3) = 0.11/4.45 = 0.025$ (2.5%)
- $P(4) = 1.51/4.45 = 0.339$ (33.9%)
- $P(5) = 0.31/4.45 = 0.070$ (7.0%)

Transition 2 (highest TD error) is sampled most frequently.

---

## The Bias Problem

Prioritized sampling introduces **bias** because frequently sampled transitions contribute more to the gradient.

The expected gradient is no longer an unbiased estimate of the true gradient:

$$
E_{PER}[\nabla L] \neq E_{uniform}[\nabla L]
$$

This can cause the learned Q-function to be biased.

---

## Importance Sampling Correction

To correct for the sampling bias, we weight each transition's contribution:

$$
w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta
$$

where $N$ is the buffer size and $\beta \in [0, 1]$ controls the amount of correction.

**Normalized weights:**

$$
w_i^{norm} = \frac{w_i}{\max_j w_j}
$$

Normalization ensures weights are in $[0, 1]$, preventing large gradient updates.

---

## The $\beta$ Parameter

**$\beta = 0$:** No correction. Full bias from prioritization.

**$\beta = 1$:** Full correction. Unbiased estimates.

**$\beta$ annealing:** Start with $\beta_0 < 1$ and increase to $\beta = 1$ during training.

**Typical schedule:** Linear annealing from $\beta_0 = 0.4$ to $\beta = 1.0$.

**Rationale:** Early in training, some bias is acceptable and speeds up learning. Near convergence, we want unbiased updates.

---

## Weighted Loss Function

The TD loss with importance sampling weights:

$$
L = \frac{1}{B} \sum_{i=1}^{B} w_i \cdot (y_i - Q(s_i, a_i))^2
$$

where:
- $B$ is batch size
- $y_i = r_i + \gamma \max_{a'} Q(s_i', a')$ is the TD target
- $w_i$ is the importance sampling weight

---

## Priority Update

After sampling and computing TD errors, update the priorities:

**For each sampled transition $i$:**

1. Compute new TD error: $\delta_i = |y_i - Q(s_i, a_i)|$
2. Update priority: $p_i = |\delta_i| + \epsilon$

This ensures priorities reflect current model predictions.

---

## Efficient Implementation: Sum Tree

Computing $\sum_k p_k$ for every sample is expensive ($O(N)$).

**Sum Tree:** A binary tree data structure that enables:
- $O(\log N)$ priority updates
- $O(\log N)$ proportional sampling
- $O(1)$ total priority sum

**Structure:**
- Leaf nodes store transition priorities
- Internal nodes store sum of children
- Root stores total priority

---

## Sum Tree Sampling

To sample proportional to priority:

1. Generate random number $s \in [0, \sum_i p_i]$
2. Traverse tree from root:
   - If $s <$ left child sum, go left
   - Else subtract left sum from $s$, go right
3. Return leaf index

This efficiently finds the transition where cumulative priority exceeds $s$.

---

## Rank-Based Prioritization

An alternative to proportional prioritization:

**Rank by TD error magnitude**

Priority based on rank rather than absolute value:

$$
p_i = \frac{1}{\text{rank}(i)}
$$

**Advantages:**
- More robust to outliers
- Heavy-tailed distribution

**Disadvantages:**
- Requires sorting or maintaining sorted structure
- More complex implementation

---

## Comparing Prioritization Schemes

**Proportional ($p_i \propto |\delta_i|$):**
- Simple to implement
- Sensitive to outliers
- Used in original PER paper

**Rank-based ($p_i \propto 1/\text{rank}$):**
- Robust to outliers
- Heavier tail (more exploration)
- Slightly better empirically

---

## When Priorities Become Stale

As the Q-network updates, TD errors change. Stored priorities become outdated.

**Solutions:**

**1. Update on sample:**
Recompute TD error when transition is sampled.

**2. Periodic refresh:**
Periodically recalculate priorities for all transitions.

**3. Lazy updates:**
Accept some staleness; priorities approximately correct.

In practice, updating sampled transitions is sufficient.

---

## Initial Priority for New Transitions

When a new transition is added, its TD error is unknown.

**Option 1: Maximum priority**

$$
p_{new} = \max_i p_i
$$
Ensures new transitions are sampled at least once.

**Option 2: Fixed high priority**

Use a constant high value.

**Option 3: Compute immediately**

Calculate TD error before storing (more expensive).

Maximum priority is the standard approach.

---

## Hyperparameters Summary

**$\alpha$ (prioritization exponent):**
- Controls how much prioritization affects sampling
- Typical: 0.6
- Range: [0, 1]

**$\beta$ (importance sampling exponent):**
- Controls bias correction
- Start: 0.4, anneal to 1.0
- Range: [0, 1]

**$\epsilon$ (priority offset):**
- Ensures non-zero sampling probability
- Typical: $10^{-6}$ to $10^{-5}$

---

## PER Algorithm Summary

**Storing:**
1. Compute TD error for new transition
2. Set priority $p = |\delta| + \epsilon$
3. Add to buffer with priority

**Sampling:**
1. Sample batch proportional to priorities
2. Compute importance weights $w_i$
3. Normalize weights

**Learning:**
1. Compute TD targets and errors
2. Weight loss by importance weights
3. Update network
4. Update priorities for sampled transitions

---

## Advantages of PER

**1. Sample efficiency:**
Learn more from each interaction by focusing on informative transitions.

**2. Faster learning:**
Difficult transitions are revisited more often.

**3. Better final performance:**
More thorough exploration of challenging regions.

---

## Disadvantages of PER

**1. Implementation complexity:**
Sum tree and importance sampling add overhead.

**2. Computational cost:**
Priority updates and tree operations.

**3. Hyperparameter sensitivity:**
$\alpha$ and $\beta$ require tuning.

**4. Potential for overfitting:**
May overfit to high-priority transitions.

---

## PER in Modern Algorithms

**Rainbow DQN:**
Combines PER with other improvements (double DQN, dueling, etc.)

**Distributed RL (Ape-X):**
PER with distributed actors and learner.

**Soft Actor-Critic:**
Can use PER for off-policy learning.

PER is a standard component in state-of-the-art deep RL systems.
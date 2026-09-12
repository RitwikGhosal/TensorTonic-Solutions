## What Is Novelty Score?

Novelty score measures how unexpected or unfamiliar recommended items are to users. It quantifies the degree to which recommendations introduce users to items they would not have discovered on their own.

High novelty means recommending items that are less popular, less well-known, or further from the user's typical choices.

---

## Why Novelty Matters

**Discovery value:**

Users often want to discover new things, not just see items they already know about.

**Beyond accuracy:**

A system that only recommends popular items may be accurate but not useful for discovery.

**Long-tail exposure:**

Novelty encourages recommending niche items, benefiting both users and content creators.

**User engagement:**

Novel recommendations can increase user interest and time spent exploring.

---

## Popularity-Based Novelty

The most common definition uses item popularity as a proxy for familiarity:

$$
\text{novelty}(i) = -\log_2 p(i)
$$

where $p(i)$ is the probability that a random user has seen/rated item $i$.

$$
p(i) = \frac{|U_i|}{|U|}
$$

**Interpretation:** Less popular items have higher novelty (they are less likely to be known).

---

## Novelty Formula Derivation

The formula $-\log_2 p(i)$ comes from information theory:

**Self-information:**

The "surprise" of an event with probability $p$ is $-\log_2 p$.

**Low probability = High surprise = High novelty**

**High probability = Low surprise = Low novelty**

If everyone knows an item (p ≈ 1), novelty ≈ 0.
If almost no one knows an item (p ≈ 0), novelty is high.

---

## Worked Example

**System with 1000 users:**

- Item A: rated by 500 users, $p(A) = 0.5$
- Item B: rated by 100 users, $p(B) = 0.1$
- Item C: rated by 10 users, $p(C) = 0.01$

**Novelty scores:**

- $\text{novelty}(A) = -\log_2(0.5) = 1.0$
- $\text{novelty}(B) = -\log_2(0.1) \approx 3.32$
- $\text{novelty}(C) = -\log_2(0.01) \approx 6.64$

Item C (least popular) has the highest novelty.

---

## Average Novelty of a Recommendation List

For a recommendation list $R_u$ of K items for user $u$:

$$
\text{Novelty}(R_u) = \frac{1}{K} \sum_{i \in R_u} -\log_2 p(i)
$$

This averages the novelty of all recommended items.

**Example:**

$R_u = \{A, B, C\}$ with novelty scores $\{1.0, 3.32, 6.64\}$

$$
\text{Novelty}(R_u) = \frac{1.0 + 3.32 + 6.64}{3} = 3.65
$$

---

## System-Wide Novelty

Average novelty across all users:

$$
\text{Novelty}_{system} = \frac{1}{|U|} \sum_{u \in U} \text{Novelty}(R_u)
$$

This measures how novel the system's recommendations are overall.

---

## Alternative Novelty Definitions

**Distance-based novelty:**

How far is the recommended item from items the user has already seen?

$$
\text{novelty}(i; u) = \min_{j \in I_u} d(i, j)
$$

where $d(i, j)$ is distance between items in feature space.

**Content-based novelty:**

How different is the recommended item from the user's profile?

**Temporal novelty:**

How recently was the item added to the system?

---

## Novelty vs Serendipity

**Novelty:**

Is the item unknown/unexpected to the user?

**Serendipity:**

Is the item both novel AND relevant/enjoyable?

Serendipity = Novelty + Relevance

A completely random recommendation might be novel but not serendipitous if the user dislikes it.

---

## Novelty vs Diversity

**Novelty:**

How unfamiliar are individual items to the user?

**Diversity:**

How different are the recommended items from each other?

Both are beyond-accuracy metrics, but they measure different aspects. A list can be novel but not diverse (all niche items from same genre) or diverse but not novel (popular items from different genres).

---

## The Accuracy-Novelty Tradeoff

**High accuracy systems tend toward low novelty:**

- Accurate predictions favor popular, well-rated items
- Popular items have more data, making predictions more reliable
- Safe, popular recommendations are often accurate

**Increasing novelty may decrease accuracy:**

- Recommending obscure items is riskier
- Less data means less confident predictions

**The goal:** Find novel items that the user will actually like.

---

## Incorporating Novelty into Recommendations

**Re-ranking:**

Generate candidates by relevance, then re-rank to include more novel items.

**Multi-objective optimization:**

Optimize for both relevance and novelty:

$$
\text{score}(i) = \alpha \cdot \text{relevance}(i) + (1-\alpha) \cdot \text{novelty}(i)
$$

**Diversification algorithms:**

Explicitly balance relevance and novelty in the selection process.

---

## Measuring Novelty in Evaluation

**Offline evaluation:**

Compute novelty of recommended items using historical popularity.

**Online evaluation:**

Track whether users click on/engage with novel recommendations.

**User studies:**

Ask users if recommendations helped them discover new things.

---

## Popularity Bias and Novelty

Many recommendation algorithms have popularity bias:

- Popular items appear in more training examples
- Algorithms learn to prefer them
- Resulting low novelty

**Addressing popularity bias:**

- Inverse propensity scoring
- Popularity debiasing during training
- Post-hoc re-ranking for novelty

---

## Novelty at Different K Values

Novelty changes with list length:

**Short lists (K=5):**

Algorithms typically include mostly relevant (often popular) items.

**Longer lists (K=50):**

More room for novel items after covering obvious choices.

**Novelty@K:**

Compute novelty only for top-K recommendations.

---

## User-Specific Novelty

Novelty can be personalized:

**Based on user history:**

Item $i$ is novel for user $u$ if unlike items $u$ has seen.

$$
\text{novelty}_u(i) = 1 - \max_{j \in I_u} \text{sim}(i, j)
$$

**Based on user expectations:**

Different users expect different levels of novelty.

- New users: prefer familiar items
- Power users: crave novelty

---

## Reporting Novelty Metrics

When reporting novelty:

**Specify the formula used:**

Popularity-based, distance-based, etc.

**Report alongside accuracy:**

Novelty alone is not sufficient. Random recommendations have high novelty.

**Show tradeoff curves:**

Plot accuracy vs novelty as a tuning parameter varies.
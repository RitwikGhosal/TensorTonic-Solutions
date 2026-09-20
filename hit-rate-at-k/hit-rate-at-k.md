## What Is Hit Rate at K?

Hit Rate at K (HR@K) is a metric that measures the proportion of users for whom at least one relevant item appears in their top-K recommendations. It answers the question: "For how many users did we successfully recommend something they actually wanted?"

$$
\text{HR@K} = \frac{\text{Number of users with at least one hit in top-K}}{\text{Total number of users}}
$$

A "hit" occurs when a recommended item matches a known relevant item for that user.

---

## The Hit Rate Formula

$$
\text{HR@K} = \frac{1}{|U|} \sum_{u \in U} \mathbb{1}[|R_u^K \cap T_u| > 0]
$$

where:
- $U$ is the set of all users
- $R_u^K$ is the set of top-K recommended items for user $u$
- $T_u$ is the set of relevant (ground truth) items for user $u$
- $\mathbb{1}[\cdot]$ is the indicator function (1 if true, 0 if false)

---

## Understanding Hit Rate

Hit rate is a binary per-user metric:

- **Hit (1):** At least one of the K recommendations is relevant
- **Miss (0):** None of the K recommendations are relevant

It does not matter if 1 or all K recommendations are relevant; a user either has a hit or does not.

---

## Worked Example

**5 users with their relevant items and top-3 recommendations:**

User A:
- Relevant: {101, 102}
- Top-3: {101, 205, 310}
- Hit? Yes (101 is in both)

User B:
- Relevant: {201}
- Top-3: {205, 210, 215}
- Hit? No (201 not in top-3)

User C:
- Relevant: {301, 302, 303}
- Top-3: {302, 303, 305}
- Hit? Yes (302 and 303 are hits)

User D:
- Relevant: {401}
- Top-3: {410, 420, 430}
- Hit? No

User E:
- Relevant: {501, 502}
- Top-3: {503, 504, 501}
- Hit? Yes (501 is in both)

**Calculation:**

$$
\text{HR@3} = \frac{3}{5} = 0.6 = 60\%
$$

60% of users got at least one relevant recommendation in their top-3.

---

## Hit Rate vs Precision and Recall

**Hit Rate:**

Binary success per user. Did the user get ANY relevant item?

**Precision@K:**

What fraction of the K recommendations are relevant?

$$
\text{Precision@K} = \frac{|R_u^K \cap T_u|}{K}
$$

**Recall@K:**

What fraction of relevant items appear in top-K?

$$
\text{Recall@K} = \frac{|R_u^K \cap T_u|}{|T_u|}
$$

Hit rate treats all users equally regardless of how many relevant items they have or how many hits they got.

---

## When to Use Hit Rate

**Leave-one-out evaluation:**

A common evaluation protocol holds out exactly one item per user. Hit rate then measures: "Did we correctly recommend the held-out item?"

$$
\text{HR@K}_{leave-one-out} = \frac{\text{Users whose held-out item is in top-K}}{\text{Total users}}
$$

**Session-based recommendations:**

Did the user find something they clicked on / purchased in the recommendations?

**Any-hit scenarios:**

When the goal is simply to show at least one good item, not to maximize the number of good items.

---

## Hit Rate at Different K Values

HR@K increases with K (larger K means more chances for a hit):

- HR@1: Strictest. The single top recommendation must be relevant.
- HR@5: More lenient. Any of top-5 must be relevant.
- HR@10: Even more lenient.
- HR@N (N = catalog size): Would be 100% if every user has at least one relevant item.

**Typical reporting:** HR@1, HR@5, HR@10, HR@20

---

## Computing HR@K Step by Step

**Step 1:** For each user, generate the top-K recommendations

**Step 2:** For each user, check if any recommendation is in their relevant set

**Step 3:** Count users with at least one hit

**Step 4:** Divide by total users

---

## Relationship to Recall@K

Hit rate is related to recall when each user has exactly one relevant item:

If $|T_u| = 1$ for all users:

$$
\text{HR@K} = \text{Average Recall@K}
$$

When users have multiple relevant items, hit rate is less granular than recall.

---

## Hit Rate in Leave-One-Out Evaluation

The leave-one-out protocol:

1. For each user, hide one of their interactions (the test item)
2. Train on remaining data
3. Generate top-K recommendations for each user
4. Check if the hidden item appears in top-K

$$
\text{HR@K} = \frac{\text{Users whose hidden item is in top-K}}{\text{Total users}}
$$

This is a very common evaluation approach in research papers.

---

## Normalized Hit Rate

To compare across datasets with different catalog sizes, normalize:

$$
\text{Normalized HR@K} = \frac{\text{HR@K}}{\text{HR@K}_{random}}
$$

where $\text{HR@K}_{random}$ is the expected hit rate from random recommendations.

For leave-one-out with catalog size $|I|$:

$$
\text{HR@K}_{random} = \frac{K}{|I|}
$$

---

## Hit Rate vs Mean Reciprocal Rank (MRR)

Both use the leave-one-out setting, but:

**Hit Rate@K:**

Binary. Did the item appear anywhere in top-K?

**MRR:**

Position matters. Where did the item appear?

$$
\text{MRR} = \frac{1}{|U|} \sum_{u} \frac{1}{\text{rank}_u}
$$

MRR gives more credit for recommending the relevant item at position 1 than position K.

---

## Variants and Extensions

**Hit Rate with Threshold:**

Instead of binary relevance, use a rating threshold.

Item is relevant if $r_{ui} \geq \tau$.

**Weighted Hit Rate:**

Weight users differently (e.g., by activity level or business value).

$$
\text{Weighted HR@K} = \frac{\sum_{u} w_u \cdot \mathbb{1}[\text{hit}_u]}{\sum_{u} w_u}
$$

---

## Interpreting Hit Rate Values

**HR@10 = 0.3:** 30% of users found at least one relevant item in top-10.

**Contextual interpretation:**

- For exploratory browsing: 30% might be acceptable
- For targeted search: 30% is likely too low
- For rare items: 30% might be excellent

Compare to baselines (random, popularity) and business requirements.

---

## Hit Rate in A/B Testing

In production A/B tests:

**Online hit rate:**

$$
\text{HR} = \frac{\text{Users who clicked/purchased a recommended item}}{\text{Users who saw recommendations}}
$$

This measures actual user engagement, not just offline relevance.

---

## Computing Confidence Intervals

Hit rate is a proportion, so confidence intervals follow binomial distribution:

**Standard error:**

$$
SE = \sqrt{\frac{\text{HR}(1 - \text{HR})}{n}}
$$

**95% confidence interval:**

$$
\text{HR} \pm 1.96 \cdot SE
$$

This helps determine if differences between systems are statistically significant.

---

## Group-Level Hit Rate

Compute hit rate for different user or item segments:

**By user activity:**

- Active users (many ratings): HR = 0.7
- Casual users (few ratings): HR = 0.4

**By item category:**

- Popular categories: HR = 0.6
- Niche categories: HR = 0.2

This reveals where the system succeeds or fails.
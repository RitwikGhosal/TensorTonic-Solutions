## What Is Adjusted Cosine Similarity?

Adjusted cosine similarity is a modification of standard cosine similarity designed specifically for item-based collaborative filtering. It accounts for the fact that different users have different rating scales by subtracting each user's mean rating before computing similarity.

This adjustment addresses the problem that some users consistently rate high (generous raters) while others rate low (harsh raters).

---

## The Problem with Standard Cosine Similarity

Standard cosine similarity between items $i$ and $j$:

$$
\cos(i, j) = \frac{\sum_{u \in U_{ij}} r_{ui} \cdot r_{uj}}{\sqrt{\sum_{u \in U_{ij}} r_{ui}^2} \cdot \sqrt{\sum_{u \in U_{ij}} r_{uj}^2}}
$$

**The issue:** If user A rates everything 4-5 stars and user B rates everything 1-2 stars, their raw ratings are not comparable. A "4" from user B means something different than a "4" from user A.

---

## The Adjusted Cosine Formula

Subtract each user's mean rating before computing similarity:

$$
\text{sim}_{adj}(i, j) = \frac{\sum_{u \in U_{ij}} (r_{ui} - \bar{r}_u)(r_{uj} - \bar{r}_u)}{\sqrt{\sum_{u \in U_{ij}} (r_{ui} - \bar{r}_u)^2} \cdot \sqrt{\sum_{u \in U_{ij}} (r_{uj} - \bar{r}_u)^2}}
$$

where:
- $r_{ui}$ is user $u$'s rating of item $i$
- $\bar{r}_u$ is user $u$'s mean rating across all items they rated
- $U_{ij}$ is the set of users who rated both items $i$ and $j$

---

## Understanding the Adjustment

By subtracting $\bar{r}_u$, we transform ratings to deviations from each user's personal average:

**User with mean rating 4.0:**
- Original rating 5 becomes: 5 - 4.0 = +1.0 (above average)
- Original rating 3 becomes: 3 - 4.0 = -1.0 (below average)

**User with mean rating 2.5:**
- Original rating 4 becomes: 4 - 2.5 = +1.5 (above average)
- Original rating 2 becomes: 2 - 2.5 = -0.5 (below average)

Now both users' ratings are on a comparable scale of "how much above or below their personal average."

---

## Worked Example

**User ratings:**

User 1 (mean = 4.0):
- Item A: 5
- Item B: 3

User 2 (mean = 2.0):
- Item A: 3
- Item B: 1

User 3 (mean = 3.0):
- Item A: 4
- Item B: 2

**Step 1: Compute adjusted ratings**

User 1: Item A = 5 - 4 = 1, Item B = 3 - 4 = -1

User 2: Item A = 3 - 2 = 1, Item B = 1 - 2 = -1

User 3: Item A = 4 - 3 = 1, Item B = 2 - 3 = -1

**Step 2: Compute similarity**

Numerator: $(1)(−1) + (1)(−1) + (1)(−1) = −3$

Denominator for A: $\sqrt{1^2 + 1^2 + 1^2} = \sqrt{3}$

Denominator for B: $\sqrt{(−1)^2 + (−1)^2 + (−1)^2} = \sqrt{3}$

$$
\text{sim}_{adj}(A, B) = \frac{-3}{\sqrt{3} \cdot \sqrt{3}} = \frac{-3}{3} = -1.0
$$

Items A and B are perfectly negatively correlated after adjustment.

---

## Interpretation of Values

The adjusted cosine similarity ranges from -1 to +1:

**+1:** Items are rated identically relative to each user's mean. When a user rates one above average, they rate the other above average too.

**0:** No linear relationship between how users rate the two items relative to their means.

**-1:** Items are rated oppositely. When a user rates one above average, they rate the other below average.

---

## Comparison with Pearson Correlation

Adjusted cosine similarity is mathematically equivalent to Pearson correlation computed across users:

$$
\text{sim}_{adj}(i, j) = \text{Pearson}(\mathbf{r}_i, \mathbf{r}_j)
$$

where $\mathbf{r}_i$ and $\mathbf{r}_j$ are the rating vectors for items $i$ and $j$.

The Pearson correlation also centers data by subtracting means, producing the same result.

---

## Why Use Adjusted Cosine for Items?

In item-based collaborative filtering, we compare items based on how users rate them.

**Different users have different baselines:**

- Generous user: rates most things 4-5
- Critical user: rates most things 2-3
- Middle user: rates most things 3

**Without adjustment:** A generous user's "4" and a critical user's "4" are treated equally, even though the generous user considers it below average while the critical user considers it excellent.

**With adjustment:** Both become deviations from personal average, making them comparable.

---

## Computing User Means

The user mean $\bar{r}_u$ is computed over all items user $u$ has rated:

$$
\bar{r}_u = \frac{1}{|I_u|} \sum_{i \in I_u} r_{ui}
$$

where $I_u$ is the set of items rated by user $u$.

**Important:** Use the mean over ALL items the user rated, not just items $i$ and $j$.

---

## Handling Edge Cases

**Only one common user:**

If only one user rated both items, the denominator involves a single term. The similarity is either +1, -1, or undefined (if that user's adjusted ratings are both zero).

**Zero variance:**

If all of a user's adjusted ratings for the common items are zero (user rated both items exactly at their mean), that user contributes zero to both numerator and denominator.

**No common users:**

If $U_{ij}$ is empty, similarity is undefined. Often set to 0 or handled specially.

---

## Adjusted Cosine vs Standard Cosine: Example

**User ratings (both rated items X and Y):**

- User A (mean 4.5): X = 5, Y = 4
- User B (mean 2.0): X = 3, Y = 1

**Standard cosine similarity:**

$$
\cos(X, Y) = \frac{5 \cdot 4 + 3 \cdot 1}{\sqrt{5^2 + 3^2} \cdot \sqrt{4^2 + 1^2}} = \frac{23}{\sqrt{34} \cdot \sqrt{17}} \approx 0.96
$$

**Adjusted cosine similarity:**

Adjusted ratings:
- User A: X = 0.5, Y = -0.5
- User B: X = 1, Y = -1

$$
\text{sim}_{adj}(X, Y) = \frac{(0.5)(-0.5) + (1)(-1)}{\sqrt{0.25 + 1} \cdot \sqrt{0.25 + 1}} = \frac{-1.25}{1.25} = -1.0
$$

The adjustment reveals that both users actually rate X above their average and Y below, showing perfect negative correlation.

---

## Use in Item-Based CF Prediction

After computing adjusted cosine similarities, predict user $u$'s rating for item $i$:

$$
\hat{r}_{ui} = \frac{\sum_{j \in N(i;u)} \text{sim}_{adj}(i, j) \cdot r_{uj}}{\sum_{j \in N(i;u)} |\text{sim}_{adj}(i, j)|}
$$

where $N(i;u)$ is the set of items similar to $i$ that user $u$ has rated.

Some formulations also adjust the prediction:

$$
\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{j \in N(i;u)} \text{sim}_{adj}(i, j) \cdot (r_{uj} - \bar{r}_u)}{\sum_{j \in N(i;u)} |\text{sim}_{adj}(i, j)|}
$$

---

## Computational Considerations

**Precompute user means:**

Calculate $\bar{r}_u$ once for each user before computing any similarities.

**Sparse data:**

Most user-item pairs have no rating. Only iterate over users who rated both items.

**Symmetry:**

$\text{sim}_{adj}(i, j) = \text{sim}_{adj}(j, i)$, so compute each pair only once.

**Storage:**

For $n$ items, there are $\binom{n}{2}$ pairs. With many items, store only top-k most similar items per item.

---

## When Adjusted Cosine Excels

**Heterogeneous user populations:**

When users have very different rating behaviors (different means and scales).

**Explicit ratings:**

Works best with explicit numerical ratings (1-5 stars) rather than implicit feedback.

**Sufficient co-ratings:**

Need enough users who rated both items to get reliable similarity estimates.

---

## Relationship to Other Similarities

**Cosine similarity:** Does not account for user bias. Works on raw ratings.

**Pearson correlation (user-based):** Centers by item mean, used in user-based CF.

**Adjusted cosine (item-based):** Centers by user mean, used in item-based CF.

The choice of centering depends on whether you're comparing users or items.
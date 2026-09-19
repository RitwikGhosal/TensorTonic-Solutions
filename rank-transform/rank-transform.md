## What Is Rank Transform?

Rank transform (also called ranking or ordinal transform) replaces each value with its position when all values are sorted. The smallest value gets rank 1, the second smallest gets rank 2, and so on.

This transformation converts numerical data to ordinal data, preserving only the relative ordering of values while discarding information about the actual magnitudes.

---

## Why Use Rank Transform?

**1. Extreme outlier resistance:**

Outliers become just another rank, eliminating their disproportionate influence.

**2. Handle any distribution:**

Works regardless of the original distribution shape.

**3. Make data comparable:**

Different features with different scales become comparable when ranked.

**4. Non-parametric analysis:**

Enables statistical methods that do not assume normal distribution.

**5. Reduce skewness:**

Transforms any distribution to approximately uniform.

---

## Basic Ranking Process

**Step 1:** Sort all values in ascending order

**Step 2:** Assign ranks starting from 1

**Step 3:** Replace each original value with its rank

**Original data:** [50, 30, 90, 10, 70]

**Sorted:** [10, 30, 50, 70, 90]

**Ranks:** [1, 2, 3, 4, 5]

**Mapped back:**

- 50 (3rd smallest) gets rank 3
- 30 (2nd smallest) gets rank 2
- 90 (5th smallest) gets rank 5
- 10 (1st smallest) gets rank 1
- 70 (4th smallest) gets rank 4

**Result:** [3, 2, 5, 1, 4]

---

## Handling Ties

When values are equal, how should ranks be assigned?

**Original data with ties:** [30, 20, 30, 10, 30]

**Methods for handling ties:**

**1. Average rank:**

Tied values get the average of their ranks.

- 10 gets rank 1
- 20 gets rank 2
- Three 30s would get ranks 3, 4, 5. Average = 4
- All 30s get rank 4

Result: [4, 2, 4, 1, 4]

**2. Minimum rank:**

Tied values get the lowest rank.

- Three 30s all get rank 3

Result: [3, 2, 3, 1, 3]

**3. Maximum rank:**

Tied values get the highest rank.

- Three 30s all get rank 5

Result: [5, 2, 5, 1, 5]

**4. Dense rank:**

No gaps in ranks; ties get same rank, next value gets next integer.

- 10 gets rank 1
- 20 gets rank 2
- All 30s get rank 3

Result: [3, 2, 3, 1, 3]

**5. Ordinal (first occurrence):**

Ties broken by position in original data.

- First 30 gets rank 3, second 30 gets rank 4, third 30 gets rank 5

Result: [3, 2, 4, 1, 5]

---

## Average Rank Example in Detail

**Data:** [100, 200, 200, 300, 400]

**Without ties, ranks would be:** 1, 2, 3, 4, 5

**With ties (average method):**

- 100: rank 1
- First 200 would be rank 2, second 200 would be rank 3
- Average of 2 and 3 is 2.5
- Both 200s get rank 2.5
- 300: rank 4
- 400: rank 5

**Result:** [1, 2.5, 2.5, 4, 5]

---

## Percentile Rank

Instead of absolute ranks, use percentile ranks:

$$
\text{percentile rank} = \frac{\text{rank} - 1}{n - 1} \times 100
$$

Or normalized to [0, 1]:

$$
\text{normalized rank} = \frac{\text{rank} - 1}{n - 1}
$$

**Example:** Data with 5 values, ranks [1, 2, 3, 4, 5]

- Rank 1: (1-1)/(5-1) = 0
- Rank 2: (2-1)/(5-1) = 0.25
- Rank 3: (3-1)/(5-1) = 0.5
- Rank 4: (4-1)/(5-1) = 0.75
- Rank 5: (5-1)/(5-1) = 1.0

**Result:** [0, 0.25, 0.5, 0.75, 1.0]

---

## Quantile Transform (Related)

Quantile transform maps data to a specified distribution:

**To uniform distribution:**

$$
x_{transformed} = F(x)
$$

where $F$ is the cumulative distribution function (estimated from data).

**To normal distribution:**

$$
x_{transformed} = \Phi^{-1}(F(x))
$$

where $\Phi^{-1}$ is the inverse normal CDF.

Rank transform is essentially quantile transform to uniform distribution.

---

## Effect on Outliers

**Original data:** [10, 20, 30, 40, 10000]

**With regular scaling (min-max):**

Values would be approximately [0, 0.001, 0.002, 0.003, 1.0]

The outlier dominates.

**With rank transform:**

Ranks: [1, 2, 3, 4, 5]

The outlier is just rank 5, one step above 40.

**Extreme outlier resistance:** Rank transform treats the gap between 40 and 10000 the same as the gap between 10 and 20.

---

## Effect on Distribution

**Before rank transform:**

Any distribution (normal, skewed, bimodal, etc.)

**After rank transform:**

Approximately uniform distribution.

**Percentile ranks:** Perfectly uniform on [0, 1]

**Quantile transform to normal:** Normal distribution

---

## Rank Transform for Feature Scaling

Using ranks instead of original values:

**Benefits:**

- All features become comparable
- No feature dominates due to scale
- Robust to outliers in any feature

**When to use:**

- Features have very different distributions
- Features have heavy tails or outliers
- Distribution-free comparison is needed

---

## Rank in Non-Parametric Statistics

Rank transform is fundamental to many statistical tests:

**Spearman correlation:**

Pearson correlation computed on ranks.

$$
\rho_s = \text{correlation of ranks}(X, Y)
$$

**Mann-Whitney U test:**

Tests if two groups differ, based on ranks.

**Kruskal-Wallis test:**

Non-parametric ANOVA using ranks.

---

## Rank Transform in Machine Learning

**For tree-based models:**

Trees only care about order, so rank transform does not change tree splits. However, it can help with:
- Numerical stability
- Faster training (smaller values)

**For linear models:**

Rank transform can help when:
- Relationship is monotonic but not linear
- Outliers would otherwise dominate

**For neural networks:**

Rank transform provides uniform input distribution, which can help training.

---

## Preserving Information

**What is preserved:**

- Relative ordering of values
- Monotonic relationships

**What is lost:**

- Actual magnitudes
- Distance information (100 vs 200 becomes same as 200 vs 10000)
- Non-monotonic patterns

**Trade-off:** Gaining robustness at the cost of information loss.

---

## Applying to New Data

**Challenge:** New data may have values outside training range.

**Options:**

**1. Use training quantiles:**

Map new values to closest quantile from training data.

**2. Extrapolate:**

Values below training min get rank 0, above max get rank n+1.

**3. Refit:**

If distribution shifts significantly, consider refitting.

---

## Rank Transform vs Standardization

**Standardization (Z-score):**

$$
x_{std} = \frac{x - \mu}{\sigma}
$$

- Assumes roughly normal distribution
- Sensitive to outliers (affects mean and std)
- Preserves relative distances

**Rank transform:**

$$
x_{rank} = \text{rank}(x)
$$

- Works for any distribution
- Robust to outliers
- Preserves only ordering

---

## Rank Transform vs Min-Max Scaling

**Min-Max:**

$$
x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}
$$

- Single outlier can compress all other values
- Preserves relative distances

**Rank transform:**

- Outliers get extreme ranks but do not compress others
- Only preserves order, not distances

---

## Worked Example: Complete Process

**Original data:** [15, 8, 42, 23, 8, 100, 31]

**Step 1: Sort and determine ranks**

Sorted: [8, 8, 15, 23, 31, 42, 100]

With average rank for ties:
- 8 appears twice, would be ranks 1 and 2, average = 1.5
- 15: rank 3
- 23: rank 4
- 31: rank 5
- 42: rank 6
- 100: rank 7

**Step 2: Map back to original positions**

- 15: rank 3
- 8: rank 1.5
- 42: rank 6
- 23: rank 4
- 8: rank 1.5
- 100: rank 7
- 31: rank 5

**Result:** [3, 1.5, 6, 4, 1.5, 7, 5]

---

## Reversibility

Rank transform is **not reversible**.

**Lost information:**

- Original values cannot be recovered from ranks
- Multiple original datasets can produce the same ranks

**Example:**

[1, 2, 3] and [10, 20, 30] and [1, 1000, 1000000]

All produce ranks [1, 2, 3]

---

## Feature-Wise Ranking

For multiple features, rank each column independently:

**Feature A:** [100, 200, 300] becomes [1, 2, 3]

**Feature B:** [5, 3, 8] becomes [2, 1, 3]

Each feature is ranked within its own column.

---

## Sample-Wise Ranking

Alternatively, rank values within each sample (row):

**Sample 1:** Features [100, 5, 50] becomes [3, 1, 2]

**Use case:** When you care about relative importance of features within each sample.

---

## Rank Transform and Monotonic Relationships

If the true relationship is monotonic, rank transform preserves it:

**Monotonically increasing function $f$:**

$$
x_1 < x_2 \implies f(x_1) < f(x_2)
$$

**After rank transform:**

$$
\text{rank}(x_1) < \text{rank}(x_2)
$$

The monotonic relationship is preserved.

**Non-monotonic relationships are distorted** because rank only captures order.

---

## Practical Considerations

**1. Handle ties consistently:**

Choose a method (average, min, max) and stick with it.

**2. Memory for large datasets:**

Sorting requires $O(n \log n)$ time and can be memory-intensive.

**3. Streaming data:**

Ranking requires seeing all data, making it challenging for streaming applications.

**4. Reproducibility:**

Document the tie-breaking method used.

---

## Common Mistakes

**1. Ignoring ties:**

Not specifying how to handle equal values.

**2. Ranking across dataset instead of per feature:**

Usually want to rank within each feature column.

**3. Expecting reversibility:**

Cannot recover original values from ranks.

**4. Using for non-monotonic patterns:**

Rank transform destroys non-monotonic relationships.

---

## Best Practices

**1. Visualize before and after:**

Confirm the transform achieves desired effect.

**2. Choose appropriate tie method:**

Average is most common for continuous data.

**3. Document the transformation:**

Record the method used for reproducibility.

**4. Consider quantile transform:**

For mapping to specific distributions, quantile transform may be more appropriate.

**5. Validate on holdout data:**

Ensure rank transform improves model performance.
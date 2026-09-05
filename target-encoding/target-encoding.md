## What Is Target Encoding?

Target encoding (also called mean encoding or likelihood encoding) is a technique for converting categorical features into numerical values by replacing each category with a statistic computed from the target variable. Most commonly, each category is replaced by the mean of the target for samples in that category.

For a category $c$:

$$
\text{encoded}(c) = \text{mean}(y | \text{category} = c)
$$

---

## Why Use Target Encoding?

**1. Captures predictive information:**

The encoding directly reflects the relationship between category and target.

**2. Reduces dimensionality:**

Converts a categorical feature to a single numerical column (versus k columns for one-hot).

**3. Handles high cardinality:**

Works well for features with many unique categories.

**4. Can improve model performance:**

The encoded values are predictive by construction.

---

## Basic Target Encoding Process

**Step 1:** For each category, identify all samples with that category

**Step 2:** Compute the mean target value for those samples

**Step 3:** Replace the category with this mean value

**Step 4:** Handle regularization to prevent overfitting

---

## Worked Example: Binary Classification

**Data:**

Sample 1: City = New York, Target = 1

Sample 2: City = New York, Target = 1

Sample 3: City = New York, Target = 0

Sample 4: City = Boston, Target = 0

Sample 5: City = Boston, Target = 0

Sample 6: City = Chicago, Target = 1

**Target means by city:**

- New York: (1 + 1 + 0) / 3 = 0.667
- Boston: (0 + 0) / 2 = 0.0
- Chicago: (1) / 1 = 1.0

**Encoded data:**

- Sample 1: 0.667
- Sample 2: 0.667
- Sample 3: 0.667
- Sample 4: 0.0
- Sample 5: 0.0
- Sample 6: 1.0

---

## Worked Example: Regression

**Data:**

Sample 1: Brand = A, Price = 100

Sample 2: Brand = A, Price = 120

Sample 3: Brand = A, Price = 110

Sample 4: Brand = B, Price = 200

Sample 5: Brand = B, Price = 220

**Target means by brand:**

- Brand A: (100 + 120 + 110) / 3 = 110
- Brand B: (200 + 220) / 2 = 210

**Encoded data:**

- Sample 1: 110
- Sample 2: 110
- Sample 3: 110
- Sample 4: 210
- Sample 5: 210

---

## The Overfitting Problem

Target encoding has a serious risk: **data leakage**.

**The problem:**

When computing the encoded value for a sample, you are using that sample's own target value in the calculation.

**Example:** Category X appears only once with target = 1

Encoded value = 1 (based on that single sample)

When training, the model sees encoded = 1 and target = 1, learning a perfect but meaningless correlation.

**This causes:**

- Overfitting to training data
- Poor generalization to new data
- Overly optimistic cross-validation scores

---

## Regularization: Smoothing

Blend the category mean with the global mean using a smoothing factor:

$$
\text{encoded}(c) = \alpha \cdot \text{mean}_c + (1 - \alpha) \cdot \text{mean}_{global}
$$

where $\alpha$ depends on category count:

$$
\alpha = \frac{n_c}{n_c + m}
$$

- $n_c$: Number of samples with category $c$
- $m$: Smoothing parameter (hyperparameter)

**Effect:**

- Categories with many samples: $\alpha \approx 1$, use category mean
- Categories with few samples: $\alpha \approx 0$, use global mean

---

## Smoothing Example

**Global mean:** 0.5

**Category A:** 100 samples, mean = 0.8

**Category B:** 2 samples, mean = 1.0

**With smoothing parameter m = 10:**

Category A:

$$
\alpha = \frac{100}{100 + 10} = 0.909
$$

$$
\text{encoded} = 0.909 \times 0.8 + 0.091 \times 0.5 = 0.773
$$

Category B:

$$
\alpha = \frac{2}{2 + 10} = 0.167
$$

$$
\text{encoded} = 0.167 \times 1.0 + 0.833 \times 0.5 = 0.583
$$

Category B is pulled strongly toward global mean due to small sample size.

---

## Leave-One-Out Target Encoding

Compute the mean excluding the current sample:

$$
\text{encoded}_i = \frac{\sum_{j \neq i, c_j = c_i} y_j}{n_c - 1}
$$

**Benefits:**

- Reduces overfitting
- No data leakage from current sample

**Implementation:**

For each sample, the encoded value is the mean of the target for all other samples with the same category.

---

## Leave-One-Out Example

**Category A with 3 samples:**

- Sample 1: target = 1
- Sample 2: target = 1
- Sample 3: target = 0

**Standard encoding:** mean = (1 + 1 + 0) / 3 = 0.667 for all

**Leave-one-out encoding:**

- Sample 1: mean of others = (1 + 0) / 2 = 0.5
- Sample 2: mean of others = (1 + 0) / 2 = 0.5
- Sample 3: mean of others = (1 + 1) / 2 = 1.0

Each sample gets a different encoded value.

---

## K-Fold Target Encoding

Use cross-validation folds to compute encodings:

**Process:**

1. Split training data into k folds
2. For each fold, compute category means from the other k-1 folds
3. Apply those means to encode the current fold
4. For test data, use means from all training data

**Benefits:**

- More robust than leave-one-out
- No data leakage
- Works well in practice

---

## K-Fold Example

**5 folds:** Fold 1, Fold 2, Fold 3, Fold 4, Fold 5

**For samples in Fold 1:**

Compute category means using Folds 2, 3, 4, 5 only.

**For samples in Fold 2:**

Compute category means using Folds 1, 3, 4, 5 only.

Each fold gets encoded using independent data.

---

## Target Encoding for Multi-Class Classification

**Option 1: Use class probability**

For each category, compute the probability of each class:

$$
\text{encoded}(c, \text{class}_k) = P(\text{class}_k | \text{category} = c)
$$

Creates k-1 new features (one per class, excluding reference).

**Option 2: One-vs-rest**

Create separate binary encodings for each class.

---

## Comparison with Other Encodings

**One-hot encoding:**

- No target information used
- No overfitting risk from target
- Creates many features for high cardinality

**Label encoding:**

- Arbitrary integers
- No target information
- Single column but no predictive content

**Frequency encoding:**

- Uses category frequency
- No target information
- May or may not be predictive

**Target encoding:**

- Uses target information
- Highly predictive
- Risk of overfitting

---

## Handling New Categories

When a category appears in test data but not in training:

**Option 1: Use global mean**

$$
\text{encoded}(c_{new}) = \text{mean}_{global}
$$

**Option 2: Use prior (if smoothing is used)**

The smoothed encoding for a category with count 0:

$$
\text{encoded} = 0 \times \text{mean}_c + 1 \times \text{mean}_{global} = \text{mean}_{global}
$$

---

## Handling Missing Values

**Option 1: Treat as separate category**

Compute mean target for samples where category is missing.

**Option 2: Use global mean**

Replace missing with global mean.

**Option 3: Impute category first**

Fill missing category, then encode.

---

## Bayesian Target Encoding

A principled approach using Bayesian updating:

**Prior:** Global mean with prior strength $m$

**Likelihood:** Category-specific observations

**Posterior:**

$$
\text{encoded}(c) = \frac{n_c \cdot \text{mean}_c + m \cdot \text{mean}_{global}}{n_c + m}
$$

This is equivalent to the smoothing formula but with Bayesian interpretation.

---

## When to Use Target Encoding

**Good scenarios:**

- High-cardinality categorical features
- Features with clear relationship to target
- When dimensionality reduction is needed
- Tree-based models that do not overfit easily

**Use with caution:**

- Small datasets (high overfitting risk)
- Categories with very few samples
- When interpretability is important

---

## Target Encoding in Different Models

**Tree-based models (Random Forest, XGBoost):**

- Less prone to overfitting from target encoding
- Can benefit significantly from high-cardinality handling
- Often the best use case

**Linear models:**

- More prone to overfitting
- Use strong regularization (smoothing, K-fold)
- Consider alternatives like one-hot

**Neural networks:**

- Embeddings may be better alternative
- Target encoding can help for very high cardinality

---

## Practical Tips

**1. Always use regularization:**

Never use naive target encoding without smoothing or K-fold.

**2. Choose smoothing parameter carefully:**

Cross-validate to find optimal $m$ value. Common range: 1 to 100.

**3. Use K-fold for production:**

More reliable than leave-one-out in practice.

**4. Validate carefully:**

Use proper time-based or stratified splits.

**5. Combine with other encodings:**

Target encoding plus one-hot can capture different signals.

---

## Alternative Statistics

Instead of mean, you can use other statistics:

**Median:**

$$
\text{encoded}(c) = \text{median}(y | c)
$$

More robust to outliers in target.

**Standard deviation:**

$$
\text{encoded}(c) = \text{std}(y | c)
$$

Captures variability within category.

**Percentiles:**

Use various quantiles for richer encoding.

---

## Ordinal Target Encoding

For ordinal targets, preserve the ordering:

**Weighted mean by rank:**

Weight target values by their ordinal position.

**Median:**

More appropriate for ordinal targets.

---

## Time-Based Target Encoding

For time series, compute means using only past data:

**Expanding mean:**

For time t, use mean of targets from times < t.

**Rolling mean:**

Use mean from a sliding window of past observations.

**Prevents leakage:**

Never uses future information.

---

## Common Mistakes

**1. No regularization:**

Using naive mean encoding leads to severe overfitting.

**2. Using test data:**

Computing encodings using any test data information.

**3. Wrong cross-validation:**

Not using proper K-fold for encoding before CV for model selection.

**4. Ignoring rare categories:**

Not handling categories with very few samples.

**5. Single point estimates:**

Not considering uncertainty for rare categories.

---

## Best Practices

**1. Always smooth:**

Use smoothing with reasonable prior strength.

**2. Use K-fold encoding:**

Especially for smaller datasets.

**3. Validate properly:**

Ensure no leakage in cross-validation setup.

**4. Compare with alternatives:**

Test against one-hot, frequency encoding.

**5. Document the parameters:**

Record smoothing parameter and K-fold setup for reproducibility.
## What Are Polynomial Features?

Polynomial features are new features created by raising existing features to powers and computing all cross-products up to a specified degree. This allows linear models to learn non-linear relationships by fitting a polynomial function to the data.

For a single feature $x$, degree-2 polynomial features are: $1, x, x^2$

For two features $x_1, x_2$, degree-2 polynomial features are: $1, x_1, x_2, x_1^2, x_1 x_2, x_2^2$

---

## Why Use Polynomial Features?

**1. Capture non-linear relationships:**

Real-world relationships are often non-linear. Polynomial features allow linear models to fit curves.

**2. Model interactions:**

Cross-product terms ($x_1 x_2$) capture how features interact.

**3. Improve model flexibility:**

Adding polynomial features increases the hypothesis space of the model.

**4. Simple implementation:**

Feature engineering that does not require domain expertise.

---

## Mathematical Foundation

A polynomial of degree $d$ in one variable:

$$
f(x) = \beta_0 + \beta_1 x + \beta_2 x^2 + ... + \beta_d x^d
$$

This is linear in the coefficients $\beta_i$ but non-linear in $x$.

By treating each $x^k$ as a separate feature, we can use linear regression to fit polynomial curves.

---

## Single Variable Example

**Original feature:** $x$

**Degree 2 polynomial features:**

$$
[1, x, x^2]
$$

**Degree 3 polynomial features:**

$$
[1, x, x^2, x^3]
$$

**Degree d polynomial features:**

$$
[1, x, x^2, ..., x^d]
$$

---

## Worked Example: Single Variable

**Original data:** $x = [1, 2, 3, 4, 5]$

**Degree 2 transformation:**

For $x = 2$:
- $x^0 = 1$ (bias term)
- $x^1 = 2$
- $x^2 = 4$

**Full transformation:**

- $x = 1$: $[1, 1, 1]$
- $x = 2$: $[1, 2, 4]$
- $x = 3$: $[1, 3, 9]$
- $x = 4$: $[1, 4, 16]$
- $x = 5$: $[1, 5, 25]$

---

## Two Variables: Degree 2

**Original features:** $x_1, x_2$

**Degree 2 polynomial features:**

$$
[1, x_1, x_2, x_1^2, x_1 x_2, x_2^2]
$$

**Components:**

- $1$: Bias (constant term)
- $x_1, x_2$: Original features (degree 1)
- $x_1^2, x_2^2$: Squared terms (degree 2)
- $x_1 x_2$: Interaction term (degree 2)

---

## Worked Example: Two Variables

**Original features:** $x_1 = 2$, $x_2 = 3$

**Degree 2 polynomial features:**

- $1 = 1$
- $x_1 = 2$
- $x_2 = 3$
- $x_1^2 = 4$
- $x_1 x_2 = 6$
- $x_2^2 = 9$

**Result:** $[1, 2, 3, 4, 6, 9]$

---

## General Formula for Feature Count

For $n$ input features and degree $d$:

**Number of polynomial features (including bias):**

$$
\binom{n + d}{d} = \frac{(n + d)!}{n! \cdot d!}
$$

**Examples:**

- $n = 2$, $d = 2$: $\binom{4}{2} = 6$ features
- $n = 2$, $d = 3$: $\binom{5}{3} = 10$ features
- $n = 3$, $d = 2$: $\binom{5}{2} = 10$ features
- $n = 10$, $d = 2$: $\binom{12}{2} = 66$ features
- $n = 10$, $d = 3$: $\binom{13}{3} = 286$ features

---

## Feature Explosion Warning

Polynomial features grow rapidly with degree and number of input features:

**$n = 20$ features:**

- Degree 2: 231 features
- Degree 3: 1,771 features
- Degree 4: 10,626 features

**Consequences:**

- Increased training time
- Higher memory usage
- Risk of overfitting
- May need more training data

---

## Degree 2: Most Common Choice

Degree 2 is the most commonly used because:

**1. Captures most important non-linearities:**

Quadratic terms and pairwise interactions handle many real-world patterns.

**2. Manageable feature count:**

Growth is $O(n^2)$ rather than higher.

**3. Interpretable:**

Squared terms and interactions have clear meanings.

**4. Less prone to overfitting:**

Higher degrees often memorize training data.

---

## Polynomial Regression Model

With degree-2 polynomial features for one variable:

$$
y = \beta_0 + \beta_1 x + \beta_2 x^2
$$

This fits a parabola to the data.

**Interpretation:**

- $\beta_0$: Intercept (value when $x = 0$)
- $\beta_1$: Linear effect
- $\beta_2$: Curvature (positive = U-shape, negative = inverted U)

---

## With Interactions

For two variables with degree 2:

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_1^2 + \beta_4 x_1 x_2 + \beta_5 x_2^2
$$

**$\beta_4 x_1 x_2$ captures:**

How the effect of $x_1$ on $y$ changes depending on the value of $x_2$ (and vice versa).

---

## Interaction-Only Option

Sometimes you want interactions without squared terms:

**Original:** $x_1, x_2, x_3$

**Interaction-only (no powers):**

$$
[1, x_1, x_2, x_3, x_1 x_2, x_1 x_3, x_2 x_3]
$$

This is useful when individual feature effects are already captured but interactions are needed.

---

## Scaling Before Polynomial Transformation

**Important:** Scale features before computing polynomial features.

**Without scaling:**

- $x = 1000$
- $x^2 = 1,000,000$
- $x^3 = 1,000,000,000$

Large values can cause numerical instability.

**With scaling to [0, 1] first:**

- $x = 0.5$
- $x^2 = 0.25$
- $x^3 = 0.125$

Values remain manageable.

---

## Regularization with Polynomial Features

More features increase overfitting risk. Use regularization:

**Ridge regression (L2):**

$$
\text{minimize } ||y - X\beta||^2 + \lambda ||\beta||^2
$$

**Lasso (L1):**

$$
\text{minimize } ||y - X\beta||^2 + \lambda ||\beta||_1
$$

Lasso can set some polynomial coefficients to zero, performing feature selection.

---

## Bias-Variance Tradeoff

**Low degree (underfitting):**

- High bias, low variance
- Cannot capture complex patterns
- Generalizes well but misses true relationships

**High degree (overfitting):**

- Low bias, high variance
- Captures training data perfectly
- Generalizes poorly to new data

**Optimal degree:**

Use cross-validation to find the right balance.

---

## Choosing the Degree

**Methods:**

1. **Cross-validation:** Test degrees 1, 2, 3, ... and select based on validation error

2. **Domain knowledge:** Physics or domain constraints may suggest appropriate degree

3. **Visual inspection:** Plot data and fitted curves to assess fit quality

4. **Information criteria:** AIC or BIC penalize model complexity

---

## Polynomial Features vs Other Non-Linear Models

**Polynomial features + Linear model:**

- Interpretable coefficients
- Manual feature engineering
- Can overfit with high degree
- Works with any linear model

**Decision trees:**

- Automatic non-linearity capture
- No need for explicit polynomial terms
- Can overfit too
- Less interpretable for complex trees

**Neural networks:**

- Learn non-linearities automatically
- More flexible
- Require more data
- Less interpretable

---

## Computational Considerations

**Memory:**

Polynomial transformation can vastly increase data size.

**Sparse data:**

If original data is sparse, polynomial features may still be sparse (products of zeros are zero).

**Online computation:**

For streaming data, polynomial features can be computed on the fly.

---

## Example Application: Price Prediction

**Original features:**

- $x_1$: Square footage
- $x_2$: Number of bedrooms

**Polynomial degree 2:**

- $x_1$: Linear effect of size
- $x_2$: Linear effect of bedrooms
- $x_1^2$: Diminishing/increasing returns to size
- $x_1 x_2$: Interaction (bedroom effect depends on size)
- $x_2^2$: Diminishing returns to bedrooms

**Model can learn:**

Large houses might not benefit as much from additional bedrooms (captured by interaction term).

---

## Centered Polynomial Features

To reduce correlation between terms, center features first:

$$
x' = x - \bar{x}
$$

Then compute polynomials of $x'$.

**Benefit:**

- Linear term and squared term less correlated
- Coefficients more stable
- Easier interpretation

---

## Orthogonal Polynomials

An alternative to raw polynomials:

**Raw polynomials:** $1, x, x^2, x^3, ...$

**Orthogonal polynomials:** $P_0(x), P_1(x), P_2(x), ...$

Examples: Legendre, Chebyshev, Hermite polynomials

**Benefit:** Orthogonal features are uncorrelated, improving numerical stability and interpretability.

---

## When to Use Polynomial Features

**Good scenarios:**

- Relationship appears curved (U-shaped, S-shaped)
- Physical laws suggest polynomial relationship
- Interactions between features are suspected
- Using linear models that cannot capture non-linearity

**Avoid when:**

- High dimensional data (feature explosion)
- Using models that handle non-linearity (trees, neural networks)
- Limited training data (overfitting risk)

---

## Common Mistakes

**1. Not scaling first:**

Large polynomial terms cause numerical issues.

**2. Too high degree:**

Degree 10 polynomial almost always overfits.

**3. Ignoring multicollinearity:**

Polynomial terms are highly correlated; use regularization.

**4. All features with high degree:**

Apply polynomial only to features where non-linearity is suspected.

**5. Forgetting to apply to test data:**

Same transformation must be applied to new data.

---

## Best Practices

**1. Start with degree 2:**

Most non-linear patterns can be captured with quadratic terms.

**2. Scale features first:**

Normalize or standardize before polynomial expansion.

**3. Use regularization:**

Ridge or Lasso to prevent overfitting.

**4. Cross-validate the degree:**

Do not assume higher is better.

**5. Consider selective expansion:**

Apply polynomials only to features that need it.
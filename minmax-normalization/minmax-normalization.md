## What is Min-Max Normalization?

Min-Max normalization rescales data to fall within a specified range, most commonly [0, 1]. For each feature, the transformation maps the smallest value to 0 and the largest value to 1, with all other values linearly distributed between them. This is a fundamental preprocessing technique for making features comparable.

---

## The Core Formula

For a single value $x$ in a feature with minimum $x_{min}$ and maximum $x_{max}$:

$$
x_{normalized} = \frac{x - x_{min}}{x_{max} - x_{min}}
$$

This can be understood as two operations:
1. **Shift**: Subtract minimum to make the range start at 0
2. **Scale**: Divide by range to make the maximum equal to 1

---

## Why Normalization Matters

**Equal contribution**: Without normalization, features with large values dominate distance calculations and gradient updates.

**Convergence speed**: Optimization algorithms converge faster when features are on similar scales.

**Numerical stability**: Very large or very small values can cause overflow or underflow in computations.

**Algorithm requirements**: Many algorithms assume or work better with normalized inputs.

---

## Handling 1D vs 2D Arrays

**1D array** (single feature):
- Apply normalization across the entire array
- min and max computed from all elements

**2D array** (multiple features):
- Apply normalization independently to each column (feature)
- Each column has its own min and max
- Rows represent samples, columns represent features

**Key insight**: The axis of normalization matters. For a 2D array with shape (n_samples, n_features):
- Compute min/max along axis 0 (across samples)
- Each feature (column) gets its own normalization parameters

---

## Vectorized Implementation Concept

Instead of looping through columns:
1. Compute all column minimums in one operation
2. Compute all column maximums in one operation
3. Broadcast and divide in one operation

**Conceptual steps for a 2D array**:
- mins = minimum of each column (shape: 1 x n_features)
- maxs = maximum of each column (shape: 1 x n_features)
- ranges = maxs - mins (shape: 1 x n_features)
- normalized = (X - mins) / ranges (broadcasting handles the rest)

---

## The Epsilon Parameter

When the range is zero (all values identical), division by zero occurs:

$$
\frac{x - x_{min}}{0} = \text{undefined}
$$

**Solution**: Add a small epsilon to the denominator:

$$
x_{normalized} = \frac{x - x_{min}}{x_{max} - x_{min} + \epsilon}
$$

**Choosing epsilon**:
- Common values: $10^{-8}$, $10^{-10}$
- Must be small enough not to affect normal calculations
- Must be large enough to prevent numerical issues

**Result when range is zero**: With epsilon, all identical values normalize to approximately 0 (since numerator is 0).

---

## Worked Example: 1D Array

**Original data**: [2, 4, 6, 8, 10]

**Step 1 - Compute min and max**:
- $x_{min} = 2$
- $x_{max} = 10$
- Range = $10 - 2 = 8$

**Step 2 - Apply formula**:
- $(2 - 2) / 8 = 0.0$
- $(4 - 2) / 8 = 0.25$
- $(6 - 2) / 8 = 0.5$
- $(8 - 2) / 8 = 0.75$
- $(10 - 2) / 8 = 1.0$

**Result**: [0.0, 0.25, 0.5, 0.75, 1.0]

---

## Worked Example: 2D Array

**Original data** (3 samples, 2 features):
- Sample 0: [100, 0.1]
- Sample 1: [200, 0.2]
- Sample 2: [300, 0.5]

**Step 1 - Compute column statistics**:

Column 0: min=100, max=300, range=200
Column 1: min=0.1, max=0.5, range=0.4

**Step 2 - Normalize each column**:

Column 0:
- $(100 - 100) / 200 = 0.0$
- $(200 - 100) / 200 = 0.5$
- $(300 - 100) / 200 = 1.0$

Column 1:
- $(0.1 - 0.1) / 0.4 = 0.0$
- $(0.2 - 0.1) / 0.4 = 0.25$
- $(0.5 - 0.1) / 0.4 = 1.0$

**Normalized result**:
- Sample 0: [0.0, 0.0]
- Sample 1: [0.5, 0.25]
- Sample 2: [1.0, 1.0]

---

## Worked Example: Zero Range Column

**Original data** (3 samples, 2 features):
- Sample 0: [100, 5]
- Sample 1: [200, 5]
- Sample 2: [300, 5]

Column 1 has all identical values (range = 0).

**With epsilon = 1e-8**:

Column 1 normalization:
- Range = $5 - 5 + 10^{-8} = 10^{-8}$
- $(5 - 5) / 10^{-8} = 0$

All values in Column 1 become 0.

---

## Properties of Min-Max Normalization

**Preserves proportional relationships**: If $a$ was twice as far from the minimum as $b$, this relationship holds after normalization.

**Does not center data**: Unlike Z-score standardization, the mean of normalized data is not necessarily 0 or 0.5.

**Bounded output**: Values are guaranteed to be in [0, 1] for training data. Test data may exceed this range if it has values outside the training min/max.

**Invertible**: Can recover original values given the min and max: $x = x_{normalized} \cdot (x_{max} - x_{min}) + x_{min}$

---

## Common Pitfalls

**Forgetting axis**: Normalizing along the wrong axis produces incorrect results.

**Integer division**: In some languages, dividing integers truncates. Ensure floating-point arithmetic.

**Storing parameters**: Must save min and max from training data to apply the same transformation to test data.

**Assuming [0,1] output**: Test data can produce values outside [0, 1] if it exceeds training data range.

---

## Where Min-Max Normalization Shows Up

- **Image Processing**: Scaling pixel values from [0, 255] to [0, 1]

- **Neural Networks**: Input normalization for sigmoid and tanh activations

- **Data Visualization**: Normalizing values for color mapping

- **Similarity Measures**: Preparing features for distance calculations

- **Feature Aggregation**: Combining features from different sources

- **Time Series Analysis**: Normalizing multiple series for comparison

- **Signal Processing**: Amplitude normalization

- **Preprocessing Pipelines**: Standard step before many ML algorithms

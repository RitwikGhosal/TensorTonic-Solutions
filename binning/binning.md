## What is Binning?

Binning, also called discretization or bucketing, transforms continuous numerical variables into discrete categorical ones by grouping values into intervals (bins). This technique converts a continuous feature like age (0-100) into categories like "young", "middle-aged", and "senior".

---

## Why Discretize Continuous Data?

- **Handling Non-Linear Relationships**: Linear models assume linear relationships. Binning captures non-linear patterns - for example, insurance risk might spike after age 65 rather than increasing linearly with age.

- **Reducing Noise**: Small measurement variations become irrelevant when grouped. Temperature readings of 36.7°C vs 36.8°C likely indicate the same health state.

- **Algorithm Requirements**: Some algorithms like Naive Bayes work better with categorical data. Decision trees can benefit from pre-binned features.

- **Interpretability**: "High income" is more interpretable than "$127,453.21" for business stakeholders.

- **Handling Outliers**: Extreme values get grouped into terminal bins rather than dominating the feature range.

---

## Types of Binning

### Equal-Width Binning

Divides the range into intervals of equal size.

**Formula**: Given minimum value $x_{min}$, maximum value $x_{max}$, and $k$ bins:

$\text{bin\_width} = \frac{x_{max} - x_{min}}{k}$

**Bin assignment**: For a value $x$, its bin index is:

$\text{bin}(x) = \lfloor \frac{x - x_{min}}{\text{bin\_width}} \rfloor$

**Characteristics**:
- Simple and predictable bin boundaries
- Easy to explain and implement
- Poor performance on skewed distributions (some bins may be empty while others overflow)

**Example**: Ages 0-100 with 5 equal-width bins creates boundaries at 0, 20, 40, 60, 80, 100

---

### Equal-Frequency Binning (Quantile Binning)

Each bin contains approximately the same number of samples.

**Approach**: With $N$ samples and $k$ bins, each bin gets roughly $N/k$ samples. Bin edges are determined by quantiles (percentiles).

**Characteristics**:
- Balanced sample distribution across bins
- Handles skewed data effectively
- Bin widths vary - narrow where data is dense, wide where sparse
- Better for creating categorical features for machine learning

**Example**: For 4 bins, use the 25th, 50th, and 75th percentiles as boundaries (quartile binning)

---

### Custom Binning

Domain knowledge defines meaningful boundaries.

**Examples**:
- Age groups for marketing: [0-18), [18-35), [35-55), [55+)
- BMI categories: underweight (<18.5), normal (18.5-25), overweight (25-30), obese (30+)
- Temperature: freezing (<0°C), cold (0-15°C), mild (15-25°C), hot (>25°C)

**Advantages**: Bins have semantic meaning aligned with domain expertise

---

## Mathematical Formulation

For equal-width binning with $k$ bins:

**Bin edges**: $e_i = x_{min} + i \cdot \frac{x_{max} - x_{min}}{k}$ for $i = 0, 1, ..., k$

**Bin assignment function**:

$b(x) = \min\left(\lfloor \frac{x - x_{min}}{\text{width}} \rfloor, k-1\right)$

The $\min$ operation ensures the maximum value falls into the last bin rather than creating an out-of-bounds index.

---

## Edge Cases and Boundary Handling

- **Right Edge Inclusion**: The maximum value must be included in the last bin. Typical convention: bins are [left, right) except the final bin which is [left, right].

- **Out-of-Range Values**: New data outside training range needs strategy:
  - Clip to nearest bin (most common)
  - Assign to special "outlier" bin
  - Return NaN or raise error

- **Identical Values**: If many samples share the same value, they must go in the same bin regardless of frequency targets. This can make equal-frequency binning impossible to achieve exactly.

- **Empty Bins**: Equal-width binning on skewed data may produce empty bins in sparse regions.

---

## Worked Example: Equal-Width Binning

**Data**: [2, 7, 12, 18, 23, 31, 45, 52, 67, 89]

**Step 1** - Compute range:
- min = 2, max = 89, range = 87

**Step 2** - Calculate bin width for 4 bins:
- width = 87 / 4 = 21.75

**Step 3** - Determine bin edges:
- [2, 23.75, 45.5, 67.25, 89]

**Step 4** - Assign values:
- Values 2, 7, 12, 18, 23 → Bin 0 (range [2, 23.75))
- Values 31, 45 → Bin 1 (range [23.75, 45.5))
- Values 52, 67 → Bin 2 (range [45.5, 67.25))
- Value 89 → Bin 3 (range [67.25, 89])

**Result**: [0, 0, 0, 0, 0, 1, 1, 2, 2, 3]

**Observation**: Distribution is uneven - 5 samples in Bin 0 but only 1 in Bin 3

---

## Worked Example: Quantile Binning

**Data**: [1, 2, 3, 4, 5, 6, 7, 8, 9, 100] (note the outlier)

**For 4 bins**, compute quartile boundaries:
- 25th percentile: ~3.25
- 50th percentile: ~5.5
- 75th percentile: ~7.75

**Bin assignments**:
- Bin 0: [1, 2, 3] - values below 25th percentile
- Bin 1: [4, 5] - values between 25th and 50th percentile
- Bin 2: [6, 7] - values between 50th and 75th percentile
- Bin 3: [8, 9, 100] - values above 75th percentile

**Observation**: The outlier (100) is grouped with normal high values, preventing a sparse outlier bin. Each bin has roughly equal count despite extreme skew.

---

## Information Loss Considerations

Binning sacrifices granularity for simplicity:

- Two samples with values 20.1 and 39.9 in the same bin become indistinguishable
- The information loss increases with fewer bins
- Original continuous variable has theoretically infinite precision
- With $k$ bins, maximum representable information is $\log_2(k)$ bits

**Tradeoff**: Fewer bins = more generalization, less overfitting, but potential underfitting. More bins = finer granularity but approaching the original continuous feature.

---

## Where Binning Shows Up

- **Credit Scoring**: FICO scores binned into "Poor" (300-579), "Fair" (580-669), "Good" (670-739), "Very Good" (740-799), "Excellent" (800-850)

- **Histogram Construction**: Visualizing distributions requires binning continuous data into bar widths

- **Feature Engineering**: Converting continuous features to categorical for models that handle categories better, or to create meaningful interaction features

- **Data Privacy**: Binning ages into ranges (18-25, 26-35) protects exact ages while preserving analytical utility

- **Medical Diagnosis**: Blood pressure readings categorized as "normal", "elevated", "high stage 1", "high stage 2", "hypertensive crisis"

- **Survey Design**: Converting precise measurements to response categories for easier human interpretation

- **Recommendation Systems**: User activity levels binned into "casual", "regular", "power user" for segment-based recommendations

- **Risk Assessment**: Insurance premiums based on binned age, income, or driving history categories

## What is Robust Scaling?

Robust scaling transforms features using statistics that are robust to outliers: the median and the interquartile range (IQR). Unlike standard scaling (which uses mean and standard deviation), robust scaling is not unduly influenced by extreme values, making it ideal for datasets with outliers.

---

## Why Robust Scaling?

**Outlier resistance**: Mean and standard deviation are heavily affected by outliers. Median and IQR are not.

**Preserves distribution shape**: The bulk of the data is scaled consistently even with extreme values present.

**No outlier removal needed**: Can be applied directly without preprocessing to remove outliers.

**Better for real-world data**: Many real datasets contain measurement errors, data entry mistakes, or genuine extreme values.

---

## The Robust Scaling Formula

For a feature column $X$ with median $Q_2$ and interquartile range $IQR = Q_3 - Q_1$:

$$
x_{scaled} = \frac{x - Q_2}{IQR}
$$

Where:
- $Q_1$ = 25th percentile (first quartile)
- $Q_2$ = 50th percentile (median)
- $Q_3$ = 75th percentile (third quartile)
- $IQR = Q_3 - Q_1$ = interquartile range

---

## Understanding Quartiles

Quartiles divide a sorted dataset into four equal parts:

**Q1 (25th percentile)**: 25% of values are below this point

**Q2 (50th percentile)**: The median; 50% of values are below

**Q3 (75th percentile)**: 75% of values are below this point

**IQR**: The range containing the middle 50% of the data

---

## Comparison with Standard Scaling

**Standard (Z-score) scaling**:

$$
x_{scaled} = \frac{x - \mu}{\sigma}
$$

Uses mean $\mu$ and standard deviation $\sigma$, both sensitive to outliers.

**Robust scaling**: Uses median and IQR, resistant to outliers.

**Example impact of outliers**:

Data: [1, 2, 3, 4, 5, 100]

Standard scaling statistics:
- Mean = 19.17 (pulled high by outlier)
- Std = 38.28 (inflated by outlier)

Robust scaling statistics:
- Median = 3.5 (unaffected)
- IQR = 4 - 2 = 2 (unaffected)

---

## Worked Example

**Data**: [10, 20, 30, 40, 50, 60, 70, 80, 90, 1000]

Note: 1000 is a clear outlier.

**Step 1 - Compute quartiles**:
- Q1 (25th percentile): 27.5
- Q2 (median): 55
- Q3 (75th percentile): 82.5

**Step 2 - Compute IQR**:

$$
IQR = Q3 - Q1 = 82.5 - 27.5 = 55
$$

**Step 3 - Apply robust scaling**:

For value 10:

$$
x_{scaled} = \frac{10 - 55}{55} = \frac{-45}{55} = -0.82
$$

For value 50:

$$
x_{scaled} = \frac{50 - 55}{55} = \frac{-5}{55} = -0.09
$$

For value 1000 (outlier):

$$
x_{scaled} = \frac{1000 - 55}{55} = \frac{945}{55} = 17.18
$$

**Observation**: The outlier (1000) gets a scaled value of 17.18, clearly marking it as extreme. The bulk of the data falls in a reasonable range around 0.

---

## Properties of Robust Scaled Data

**Centered at median**: The median value maps to 0

**Scaled by IQR**: A value exactly at Q3 maps to 0.5; exactly at Q1 maps to -0.5

**Unbounded**: Unlike min-max scaling, values are not constrained to a specific range

**Outliers remain outliers**: Extreme values are clearly identifiable as far from 0

---

## Handling Zero IQR

When Q1 = Q3 (all middle 50% of values are identical):

$$
IQR = 0 \Rightarrow \frac{x - Q_2}{0} = \text{undefined}
$$

**Solutions**:
- Return 0 for all values (since they are near the median)
- Use a small epsilon: $IQR + \epsilon$
- Fall back to standard scaling for that feature

---

## Computing Percentiles

Two main interpolation methods for percentiles:

**Linear interpolation**: Interpolate between adjacent values when the percentile falls between data points

**Nearest rank**: Round to the nearest data point

Different methods can give slightly different results, especially for small datasets.

---

## Centering Options

The formula can be modified:

**With centering** (default):

$$
x_{scaled} = \frac{x - Q_2}{IQR}
$$

**Without centering**:

$$
x_{scaled} = \frac{x}{IQR}
$$

The centered version is more common as it places the median at 0.

---

## When to Use Robust Scaling

**Good for**:
- Data with known or suspected outliers
- Distributions with heavy tails
- Real-world measurements prone to errors
- When outlier removal is undesirable or infeasible

**Less suitable for**:
- Clean data without outliers (standard scaling is simpler)
- When exact [0, 1] range is required (use min-max)
- Very small datasets where percentile estimates are unreliable

---

## Column-wise Application

Like other scaling methods, robust scaling is applied independently to each feature:

**Steps for 2D data**:
1. For each column, compute Q1, Q2 (median), Q3
2. Compute IQR for each column
3. Apply formula to each element using its column statistics

**Result**: Each feature is centered at its median and scaled by its IQR

---

## Train-Test Split Considerations

**Important**: Compute Q1, Q2, Q3 from training data only

**Apply to test data** using training statistics:

$$
x_{test,scaled} = \frac{x_{test} - Q_{2,train}}{IQR_{train}}
$$

**Rationale**: Prevents data leakage from test set

---

## Robust Scaling vs Winsorization

**Robust scaling**: Transforms all values, outliers remain but are clearly extreme

**Winsorization**: Caps outliers at specified percentiles, then scales

**Combined approach**: Winsorize first to cap extreme values, then apply robust scaling

---

## Where Robust Scaling Shows Up

- **Financial Data**: Stock prices and returns often have extreme movements

- **Sensor Data**: Measurement errors and equipment failures create outliers

- **Medical Data**: Patient measurements can have recording errors

- **Web Analytics**: Traffic spikes and bot activity create outliers

- **Survey Data**: Extreme responses may be errors or genuine outliers

- **Image Processing**: Pixel intensity outliers from noise

- **Scientific Measurements**: Experimental data with occasional anomalies

- **Preprocessing for ML**: When data quality is uncertain

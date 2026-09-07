## What Are Percentiles?

A percentile indicates the **relative standing** of a value within a dataset. The $p$th percentile is a value below which $p$ percent of the data falls.

**Example:** If your test score is at the 90th percentile, you scored higher than 90% of test-takers.

Percentiles are used to understand distributions, identify outliers, and compare values across different scales.

---

## Formal Definition

The $p$th percentile (where $0 \leq p \leq 100$) is a value $x_p$ such that:

- At least $p\%$ of the data is $\leq x_p$
- At least $(100-p)\%$ of the data is $\geq x_p$

**Note:** Different methods exist for computing percentiles, especially when the percentile falls between data points.

---

## Special Percentiles

**Quartiles divide data into four parts:**
- Q1 (25th percentile): First quartile
- Q2 (50th percentile): Second quartile = Median
- Q3 (75th percentile): Third quartile

**Deciles divide data into ten parts:**
- 10th, 20th, 30th, ..., 90th percentiles

**The median is the 50th percentile:**
- Half the data is below, half is above

---

## Computing Percentiles: Basic Method

**Step 1:** Sort the data in ascending order

**Step 2:** Calculate the rank position:

$$
L = \frac{p}{100} \times (n + 1)
$$

where $p$ is the percentile and $n$ is the sample size.

**Step 3:**
- If $L$ is an integer, the percentile is the value at position $L$
- If $L$ is not an integer, interpolate between adjacent values

---

## Worked Example: Computing Percentiles

**Data:** [15, 20, 35, 40, 50] (n = 5, already sorted)

**Find the 25th percentile (Q1):**

$L = \frac{25}{100} \times (5 + 1) = 0.25 \times 6 = 1.5$

Position 1.5 means: interpolate between positions 1 and 2

$P_{25} = x_1 + 0.5 \times (x_2 - x_1) = 15 + 0.5 \times (20 - 15) = 15 + 2.5 = 17.5$

**Find the 50th percentile (Median):**

$L = \frac{50}{100} \times 6 = 3$

Position 3 is exactly the 3rd value.

$P_{50} = 35$

**Find the 75th percentile (Q3):**

$L = \frac{75}{100} \times 6 = 4.5$

$P_{75} = x_4 + 0.5 \times (x_5 - x_4) = 40 + 0.5 \times (50 - 40) = 45$

---

## Alternative Calculation Methods

There are multiple conventions for computing percentiles. Common methods include:

**Method 1: Linear interpolation (most common)**

Used in the example above. Interpolates between adjacent data points.

**Method 2: Nearest rank**

Round $L$ to the nearest integer and take that value. No interpolation.

**Method 3: Exclusive method**

$L = \frac{p}{100} \times (n + 1)$

**Method 4: Inclusive method**

$L = \frac{p}{100} \times (n - 1) + 1$

Different software uses different methods. Results may differ slightly for small samples.

---

## The Interquartile Range (IQR)

The IQR measures the spread of the middle 50% of data:

$$
\text{IQR} = Q3 - Q1 = P_{75} - P_{25}
$$

**Properties:**
- Robust measure of spread
- Not affected by outliers
- Used in box plots

**Example:** If $Q1 = 17.5$ and $Q3 = 45$:

$$
\text{IQR} = 45 - 17.5 = 27.5
$$

---

## Using IQR to Detect Outliers

A common rule defines outliers as values outside:

**Lower fence:** $Q1 - 1.5 \times \text{IQR}$

**Upper fence:** $Q3 + 1.5 \times \text{IQR}$

**Example:** With $Q1 = 17.5$, $Q3 = 45$, $\text{IQR} = 27.5$:

Lower fence = $17.5 - 1.5 \times 27.5 = 17.5 - 41.25 = -23.75$

Upper fence = $45 + 1.5 \times 27.5 = 45 + 41.25 = 86.25$

Values below $-23.75$ or above $86.25$ would be flagged as outliers.

---

## Five-Number Summary

The five-number summary consists of:

1. Minimum
2. Q1 (25th percentile)
3. Median (50th percentile)
4. Q3 (75th percentile)
5. Maximum

This summary captures the distribution's shape and spread and is the basis for box plots.

**Example:** For data [15, 20, 35, 40, 50]:
- Min = 15
- Q1 = 17.5
- Median = 35
- Q3 = 45
- Max = 50

---

## Box Plots (Box-and-Whisker Plots)

Box plots visualize the five-number summary:

- **Box:** Spans from Q1 to Q3
- **Line in box:** The median
- **Whiskers:** Extend to min/max (or to fences, with outliers shown as points)

Box plots allow quick comparison of distributions across groups.

---

## Percentile Rank

The percentile rank of a value $x$ tells what percentage of data falls at or below $x$:

$$
\text{Percentile Rank}(x) = \frac{\text{number of values} \leq x}{n} \times 100
$$

**Example:** In data [15, 20, 35, 40, 50], what is the percentile rank of 35?

3 values are $\leq 35$ (15, 20, 35)

Percentile rank = $\frac{3}{5} \times 100 = 60\%$

The value 35 is at the 60th percentile.

---

## Percentiles vs Quantiles

**Percentiles:** Divide data into 100 parts (0th to 100th)

**Quartiles:** Divide data into 4 parts (Q1, Q2, Q3)

**Deciles:** Divide data into 10 parts

**Quantiles:** General term for any division
- The 0.25 quantile = 25th percentile = Q1
- The 0.5 quantile = 50th percentile = Median

---

## Percentiles of Common Distributions

**Normal distribution:**

For $N(\mu, \sigma^2)$:
- 50th percentile = $\mu$
- 84th percentile $\approx \mu + \sigma$
- 97.5th percentile $\approx \mu + 2\sigma$
- 16th percentile $\approx \mu - \sigma$
- 2.5th percentile $\approx \mu - 2\sigma$

These correspond to standard scores (z-scores).

---

## Z-Scores and Percentiles

For a Normal distribution, the z-score tells how many standard deviations from the mean:

$$
z = \frac{x - \mu}{\sigma}
$$

**Common z-scores and percentiles:**
- $z = -2$: 2.3rd percentile
- $z = -1$: 15.9th percentile
- $z = 0$: 50th percentile
- $z = 1$: 84.1st percentile
- $z = 2$: 97.7th percentile

---

## Applications of Percentiles

**Standardized testing:**
- SAT, GRE report percentile ranks
- "90th percentile" means you outperformed 90% of test-takers

**Income and wealth:**
- "Top 1%" refers to 99th percentile
- Median income is more informative than mean

**Growth charts:**
- Child's height/weight reported as percentile
- "25th percentile for height" means 25% of children that age are shorter

**Website performance:**
- 95th percentile response time
- Captures typical user experience better than mean

---

## Percentiles in Machine Learning

**Feature scaling:**
- Percentile-based scaling (e.g., scale to [0, 100] based on percentile rank)
- Robust to outliers compared to min-max scaling

**Quantile regression:**
- Predict different percentiles, not just the mean
- Useful for prediction intervals

**Anomaly detection:**
- Flag values outside certain percentiles as anomalies
- e.g., values below 1st or above 99th percentile

**Model evaluation:**
- Report percentiles of error distribution
- 90th percentile error shows worst-case performance

---

## Computing Percentiles Efficiently

**For small datasets:**
- Sort the data: $O(n \log n)$
- Access desired positions: $O(1)$

**For single percentile:**
- Selection algorithm (quickselect): $O(n)$ average case
- No need to sort entire dataset

**For streaming data:**
- Approximate algorithms (t-digest, quantile sketches)
- Maintain approximate percentiles with bounded memory

---

## Percentiles vs Mean and Standard Deviation

**Mean and SD:**
- Assume symmetric distribution
- Sensitive to outliers
- Useful for Normal-like data

**Percentiles:**
- Make no distribution assumptions
- Robust to outliers
- Capture asymmetry in distribution

For skewed distributions, reporting Q1, median, Q3 is often more informative than mean and SD.
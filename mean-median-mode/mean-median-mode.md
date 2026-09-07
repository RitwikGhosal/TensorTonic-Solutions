## Measures of Central Tendency

Central tendency describes where the "center" of a dataset lies. The three most common measures are:

- **Mean:** The arithmetic average
- **Median:** The middle value when data is sorted
- **Mode:** The most frequently occurring value

Each measure has different properties and is appropriate in different situations.

---

## The Mean (Arithmetic Average)

The mean is the sum of all values divided by the count:

$$
\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i = \frac{x_1 + x_2 + ... + x_n}{n}
$$

**Properties:**
- Uses all data points
- Sensitive to outliers
- Minimizes sum of squared deviations
- Has nice mathematical properties (unbiased estimator of population mean)

---

## Computing the Mean: Example

**Data:** [4, 8, 6, 5, 3, 9, 7]

**Step 1:** Sum the values

$$
4 + 8 + 6 + 5 + 3 + 9 + 7 = 42
$$

**Step 2:** Count the values

$$
n = 7
$$

**Step 3:** Divide

$$
\bar{x} = \frac{42}{7} = 6
$$

The mean is 6.

---

## The Median

The median is the middle value when data is sorted in order:

**For odd $n$:** The median is the value at position $\frac{n+1}{2}$.

**For even $n$:** The median is the average of values at positions $\frac{n}{2}$ and $\frac{n}{2} + 1$.

**Properties:**
- Robust to outliers
- Minimizes sum of absolute deviations
- Good for skewed distributions
- 50th percentile

---

## Computing the Median: Odd $n$

**Data:** [4, 8, 6, 5, 3, 9, 7]

**Step 1:** Sort the data

$$
[3, 4, 5, 6, 7, 8, 9]
$$

**Step 2:** Find the middle position

$$
\text{position} = \frac{n+1}{2} = \frac{7+1}{2} = 4
$$

**Step 3:** The median is the 4th value

$$
\text{median} = 6
$$

---

## Computing the Median: Even $n$

**Data:** [4, 8, 6, 5, 3, 9]

**Step 1:** Sort the data

$$
[3, 4, 5, 6, 8, 9]
$$

**Step 2:** Find the two middle positions

$$
\text{positions} = \frac{n}{2} \text{ and } \frac{n}{2} + 1 = 3 \text{ and } 4
$$

**Step 3:** Average the 3rd and 4th values

$$
\text{median} = \frac{5 + 6}{2} = 5.5
$$

---

## The Mode

The mode is the value that appears most frequently:

**Properties:**
- Can be used with categorical data
- May not exist (if all values are unique)
- May not be unique (multimodal distributions)
- Not affected by outliers

---

## Computing the Mode: Example 1

**Data:** [4, 8, 6, 5, 3, 6, 7, 6, 9]

**Count occurrences:**
- 3 appears 1 time
- 4 appears 1 time
- 5 appears 1 time
- 6 appears 3 times
- 7 appears 1 time
- 8 appears 1 time
- 9 appears 1 time

**Mode = 6** (appears most frequently)

---

## Computing the Mode: No Mode

**Data:** [4, 8, 6, 5, 3, 9, 7]

Each value appears exactly once. There is **no mode** (or we say all values are modes).

---

## Computing the Mode: Multiple Modes

**Data:** [4, 8, 6, 5, 3, 6, 8, 9]

**Count occurrences:**
- 6 appears 2 times
- 8 appears 2 times
- All others appear 1 time

**Modes = 6 and 8** (bimodal distribution)

---

## Comparison: Mean vs Median

**Symmetric distributions:**
- Mean $\approx$ Median
- Either measure works well

**Right-skewed distributions (long right tail):**
- Mean > Median
- Median is often preferred

**Left-skewed distributions (long left tail):**
- Mean < Median
- Median is often preferred

**With outliers:**
- Mean is pulled toward outliers
- Median is robust to outliers

---

## Effect of Outliers: Example

**Data without outlier:** [10, 12, 11, 13, 12, 11, 14]

Mean = $(10+12+11+13+12+11+14)/7 = 83/7 = 11.86$

Median = 12 (middle of sorted [10, 11, 11, 12, 12, 13, 14])

**Data with outlier:** [10, 12, 11, 13, 12, 11, 100]

Mean = $(10+12+11+13+12+11+100)/7 = 169/7 = 24.14$

Median = 12 (middle of sorted [10, 11, 11, 12, 12, 13, 100])

The outlier dramatically affects the mean but not the median.

---

## When to Use Each Measure

**Use Mean when:**
- Data is roughly symmetric
- No significant outliers
- You need to use the value in further calculations
- You want to account for all values

**Use Median when:**
- Data is skewed
- Outliers are present
- Reporting "typical" values (e.g., income, house prices)
- Working with ordinal data

**Use Mode when:**
- Data is categorical
- You want the most common value
- Describing distribution shape (number of peaks)

---

## Relationship in Different Distributions

**Symmetric distribution:**

$$
\text{Mean} = \text{Median} = \text{Mode}
$$

**Right-skewed distribution:**

$$
\text{Mode} < \text{Median} < \text{Mean}
$$

**Left-skewed distribution:**

$$
\text{Mean} < \text{Median} < \text{Mode}
$$

This relationship is approximate and does not always hold exactly.

---

## Weighted Mean

When data points have different importances (weights):

$$
\bar{x}_w = \frac{\sum_{i=1}^{n} w_i x_i}{\sum_{i=1}^{n} w_i}
$$

**Example:** Calculate GPA where courses have different credits.

Grades: A(4), B(3), A(4), C(2)
Credits: 3, 4, 2, 3

$$
\text{GPA} = \frac{4(3) + 3(4) + 4(2) + 2(3)}{3 + 4 + 2 + 3} = \frac{12 + 12 + 8 + 6}{12} = \frac{38}{12} = 3.17
$$

---

## Trimmed Mean

The trimmed mean removes a percentage of extreme values before computing the mean:

**Process:**
1. Sort the data
2. Remove the lowest $k\%$ and highest $k\%$
3. Compute mean of remaining values

**Example:** 10% trimmed mean of 20 values removes the 2 smallest and 2 largest.

This provides robustness to outliers while still using most data.

---

## Geometric Mean

For positive values, the geometric mean is:

$$
\bar{x}_g = \sqrt[n]{x_1 \cdot x_2 \cdot ... \cdot x_n} = \left(\prod_{i=1}^{n} x_i\right)^{1/n}
$$

**Use cases:**
- Growth rates and ratios
- Values spanning orders of magnitude
- Proportional data

**Example:** Growth rates of 10%, 20%, 30%

Geometric mean = $\sqrt[3]{1.1 \times 1.2 \times 1.3} = \sqrt[3]{1.716} = 1.197$

Average growth rate $\approx 19.7\%$

---

## Harmonic Mean

The harmonic mean is:

$$
\bar{x}_h = \frac{n}{\sum_{i=1}^{n} \frac{1}{x_i}}
$$

**Use cases:**
- Averaging rates (speed, productivity)
- F1-score in machine learning

**Example:** Drive 60 mph one way, 40 mph return

Harmonic mean = $\frac{2}{\frac{1}{60} + \frac{1}{40}} = \frac{2}{\frac{2+3}{120}} = \frac{2 \times 120}{5} = 48$ mph

The average speed is 48 mph, not 50 mph.

---

## Relationship Between Means

For positive values:

$$
\text{Harmonic Mean} \leq \text{Geometric Mean} \leq \text{Arithmetic Mean}
$$

Equality holds only when all values are equal.

---

## Central Tendency in Machine Learning

**Imputation:**
- Missing values often imputed with mean or median
- Median is preferred for skewed features

**Loss functions:**
- Mean squared error is minimized by the mean
- Mean absolute error is minimized by the median

**Baseline models:**
- Predicting the mean (regression) or mode (classification) as baseline

**Feature engineering:**
- Mean, median, mode can be features in aggregations

---

## Population vs Sample Notation

**Population mean:** $\mu = \frac{1}{N}\sum_{i=1}^{N} x_i$

**Sample mean:** $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$

The sample mean $\bar{x}$ estimates the population mean $\mu$.

**Key property:** $E[\bar{X}] = \mu$ (unbiased estimator)

---

## Computational Considerations

**Numerical stability for mean:**

For very large datasets, use running mean:

$$
\bar{x}_n = \bar{x}_{n-1} + \frac{x_n - \bar{x}_{n-1}}{n}
$$

**Efficiency for median:**
- Full sort: $O(n \log n)$
- Selection algorithm: $O(n)$ average case

**Mode:**
- Using hash map: $O(n)$ time, $O(k)$ space where $k$ is unique values
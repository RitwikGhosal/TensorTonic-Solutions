## What Is Frequency Encoding?

Frequency encoding (also called count encoding) is a technique for converting categorical features into numerical features by replacing each category with its frequency of occurrence in the dataset.

Instead of creating multiple binary columns (one-hot encoding), frequency encoding creates a single numerical column representing how common each category is.

---

## The Core Idea

Each category is replaced by the count or proportion of times it appears in the training data:

**Count-based:**

$$
\text{encoded}(c) = \text{count}(c)
$$

**Proportion-based:**

$$
\text{encoded}(c) = \frac{\text{count}(c)}{n}
$$

where $n$ is the total number of samples.

---

## Why Use Frequency Encoding?

**1. Dimensionality reduction:**

High-cardinality categorical features (many unique values) would create too many one-hot columns. Frequency encoding creates just one column.

**2. Captures category importance:**

Rare categories get low values, common categories get high values. This can be predictive in many scenarios.

**3. Handles unseen categories:**

New categories can be assigned a frequency of 0 or a small default value.

**4. No increase in feature count:**

Unlike one-hot encoding, the number of features stays the same.

---

## Step-by-Step Process

**Step 1:** Count occurrences of each category in the training data

**Step 2:** Create a mapping from category to count (or proportion)

**Step 3:** Replace each category with its corresponding frequency value

**Step 4:** For new data, use the same mapping learned from training

---

## Worked Example: Count-Based

**Training data:** City column

- New York (appears 4 times)
- Los Angeles (appears 3 times)
- Chicago (appears 2 times)
- Houston (appears 1 time)

Total: 10 samples

**Frequency mapping:**

- New York: 4
- Los Angeles: 3
- Chicago: 2
- Houston: 1

**Original data:**

[New York, Los Angeles, Chicago, New York, Houston, Los Angeles, New York, Chicago, New York, Los Angeles]

**Encoded data:**

[4, 3, 2, 4, 1, 3, 4, 2, 4, 3]

---

## Worked Example: Proportion-Based

Using the same data with $n = 10$:

**Frequency mapping (proportions):**

- New York: 4/10 = 0.4
- Los Angeles: 3/10 = 0.3
- Chicago: 2/10 = 0.2
- Houston: 1/10 = 0.1

**Encoded data:**

[0.4, 0.3, 0.2, 0.4, 0.1, 0.3, 0.4, 0.2, 0.4, 0.3]

---

## Mathematical Formulation

Given a categorical feature $X$ with categories $\{c_1, c_2, ..., c_k\}$:

**Count encoding:**

$$
f(x_i) = \sum_{j=1}^{n} \mathbb{1}[x_j = x_i]
$$

where $\mathbb{1}[\cdot]$ is the indicator function.

**Proportion encoding:**

$$
f(x_i) = \frac{1}{n} \sum_{j=1}^{n} \mathbb{1}[x_j = x_i]
$$

---

## Handling High-Cardinality Features

Frequency encoding is especially useful for features with many unique values:

**Example: Product IDs**

- 10,000 unique products
- One-hot encoding: 10,000 columns
- Frequency encoding: 1 column

The frequency indicates product popularity, which is often predictive.

**Example: ZIP codes**

- Thousands of unique codes
- Frequency indicates population density or data collection patterns

---

## When Categories Have Equal Frequencies

If multiple categories have the same frequency, they will have the same encoded value:

**Example:**

- Category A: 5 occurrences
- Category B: 5 occurrences
- Category C: 3 occurrences

**Encoded:**

- A and B both become 5
- C becomes 3

**Implication:** The model cannot distinguish between A and B based on frequency alone. This may or may not be acceptable depending on the use case.

---

## Handling Unseen Categories

When a category appears in test data but not in training:

**Option 1: Assign zero**

$$
\text{encoded}(c_{new}) = 0
$$

**Option 2: Assign minimum frequency**

$$
\text{encoded}(c_{new}) = \min(\text{frequencies})
$$

**Option 3: Assign a small constant**

$$
\text{encoded}(c_{new}) = \epsilon
$$

Choose based on whether rarity or novelty should be emphasized.

---

## Comparison with One-Hot Encoding

**One-hot encoding:**

- Creates k binary columns for k categories
- No information about category frequency
- Treats all categories as equally different
- High dimensionality for high-cardinality features

**Frequency encoding:**

- Creates 1 numerical column
- Encodes how common each category is
- Categories with same frequency become identical
- Low dimensionality regardless of cardinality

---

## Comparison with Label Encoding

**Label encoding:** Assigns arbitrary integers (0, 1, 2, ...)

**Frequency encoding:** Assigns counts or proportions

**Key difference:** Frequency encoding has semantic meaning (popularity), while label encoding is arbitrary.

Label encoding creates artificial ordering (2 > 1) that may not be meaningful. Frequency encoding's ordering (more common > less common) often is meaningful.

---

## Use Cases Where Frequency Matters

**E-commerce:**

- Popular products may have different behavior patterns
- Frequently purchased items might have lower return rates

**Fraud detection:**

- Rare merchant categories might indicate unusual transactions
- Frequency of card usage location matters

**Customer segmentation:**

- Common customer types vs rare ones
- Frequent buyers vs occasional buyers

**Natural language processing:**

- Word frequency (term frequency) is highly predictive
- Rare words often carry more specific meaning

---

## Logarithmic Frequency Encoding

For highly skewed frequency distributions, apply log transformation:

$$
f(x) = \log(1 + \text{count}(x))
$$

**Benefits:**

- Compresses the range of values
- Reduces impact of extremely common categories
- Often improves model performance

**Example:**

- Category with count 1000 becomes log(1001) ≈ 6.91
- Category with count 10 becomes log(11) ≈ 2.40
- The ratio is compressed from 100:1 to about 2.9:1

---

## Normalized Frequency Encoding

Scale frequencies to a specific range:

**Min-max normalized:**

$$
f_{norm}(x) = \frac{\text{count}(x) - \text{min}}{\text{max} - \text{min}}
$$

**Z-score normalized:**

$$
f_{norm}(x) = \frac{\text{count}(x) - \mu}{\sigma}
$$

where $\mu$ and $\sigma$ are mean and standard deviation of counts.

---

## Frequency Encoding vs Target Encoding

**Frequency encoding:**

- Uses only the feature itself (count of category)
- No information about target variable
- No risk of target leakage

**Target encoding:**

- Uses relationship with target variable
- More predictive but risk of overfitting
- Requires careful regularization

Frequency encoding is simpler and safer but may be less predictive.

---

## Combining Frequency with Other Encodings

You can use frequency encoding alongside other techniques:

**Frequency + One-hot:**

For moderate cardinality, use both:
- One-hot captures category identity
- Frequency captures how common it is

**Frequency + Target encoding:**

- Frequency as a baseline
- Target encoding for categories where target relationship exists

---

## Grouped Frequency Encoding

For hierarchical categories, compute frequency at different levels:

**Example: Product category hierarchy**

- Electronics > Phones > Smartphones

Encode:
- Smartphone frequency
- Phones category frequency
- Electronics category frequency

This captures information at multiple granularities.

---

## Time-Aware Frequency Encoding

In time-series contexts, compute frequency using only past data:

**Expanding window:**

For each time point, count occurrences up to that point only.

**Rolling window:**

Count occurrences in the last $w$ time periods.

This prevents look-ahead bias and data leakage.

---

## Handling Rare Categories

Rare categories with frequency 1 or very low counts can be problematic:

**Option 1: Group rare categories**

Combine all categories with count < threshold into "Other"

**Option 2: Smoothing**

$$
f_{smooth}(x) = \frac{\text{count}(x) + \alpha}{n + \alpha \cdot k}
$$

where $\alpha$ is a smoothing parameter and $k$ is number of categories.

---

## Advantages of Frequency Encoding

**1. Simple and interpretable:**

The encoded value directly represents how common a category is.

**2. Low dimensionality:**

Always produces one feature regardless of number of categories.

**3. Handles high cardinality:**

Works well when there are thousands of unique values.

**4. No target leakage:**

Does not use the target variable in encoding.

**5. Fast computation:**

Just counting, no complex calculations.

---

## Disadvantages of Frequency Encoding

**1. Loss of category identity:**

Categories with same frequency become indistinguishable.

**2. May not be predictive:**

Frequency might not correlate with the target variable.

**3. Sensitivity to data size:**

Frequencies change with sample size, which can affect model stability.

**4. Same value for different categories:**

If A and B both appear 100 times, they get the same encoding.

---

## Best Practices

**1. Use proportion over count:**

Proportions are more stable and comparable across datasets.

**2. Consider log transformation:**

Especially for power-law distributed categories.

**3. Save the mapping:**

Store the frequency dictionary for consistent encoding of new data.

**4. Validate on holdout:**

Ensure frequency encoding improves performance on unseen data.

**5. Combine with other features:**

Do not rely on frequency alone for important categorical features.

---

## When to Avoid Frequency Encoding

**When category identity matters:**

If each category has distinct behavior unrelated to its frequency.

**When categories are balanced:**

If all categories have similar frequencies, the encoding provides little information.

**When order matters:**

For ordinal categories where the ranking is important, ordinal encoding is more appropriate.
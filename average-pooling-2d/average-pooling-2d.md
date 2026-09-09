## What Is Average Pooling?

Average pooling is a downsampling operation that reduces spatial dimensions by computing the mean value within each pooling region. Unlike max pooling which selects the strongest activation, average pooling captures the overall presence of features.

---

## The Average Pooling Formula

For pool size $p \times p$ with non-overlapping windows:

$$
\text{output}[i][j] = \frac{1}{p^2} \sum_{a=0}^{p-1} \sum_{b=0}^{p-1} \text{input}[i \cdot p + a][j \cdot p + b]
$$

This computes the arithmetic mean of all values in each window.

---

## Step-by-Step Example

**Input (4x4):**

4  2  6  8
0  4  2  4
8  6  2  0
2  4  6  8

**Pool size: 2x2**

**Top-left window:** (4 + 2 + 0 + 4) / 4 = 10/4 = 2.5
**Top-right window:** (6 + 8 + 2 + 4) / 4 = 20/4 = 5.0
**Bottom-left window:** (8 + 6 + 2 + 4) / 4 = 20/4 = 5.0
**Bottom-right window:** (2 + 0 + 6 + 8) / 4 = 16/4 = 4.0

**Output (2x2):**

2.5  5.0
5.0  4.0

---

## Output Dimensions

Same as max pooling:

$$
H_{out} = \left\lfloor \frac{H}{p} \right\rfloor
$$

$$
W_{out} = \left\lfloor \frac{W}{p} \right\rfloor
$$

For a 6x6 input with 2x2 pooling: output is 3x3.

---

## Average Pooling vs. Max Pooling

**Average pooling:**
- Uses all values in the window
- Smooths the feature map
- Sensitive to all activations
- Better for dense, distributed features

**Max pooling:**
- Uses only the maximum value
- Preserves sharp features
- Invariant to most activations
- Better for sparse, localized features

---

## When to Use Average Pooling

**Global average pooling (final layer):**
- Replace fully connected layers
- Average entire feature map to single value per channel
- Used in GoogLeNet, ResNet, and modern architectures
- Reduces parameters significantly

**Intermediate layers:**
- Less common than max pooling
- Used when smoothing is desirable
- Some architectures mix both types

**Dense prediction tasks:**
- Segmentation, depth estimation
- Where every input region matters
- May preserve more information

---

## The Gradient (Backpropagation)

During backpropagation, the gradient is distributed equally:

**Forward pass:**
- Compute mean of p^2 values

**Backward pass:**
- Each input element receives: (incoming gradient) / p^2
- Gradient is uniformly distributed across the window

Compared to max pooling where only the max element receives gradient, average pooling provides gradient to all elements.

---

## Global Average Pooling

A special case where pool size equals the spatial dimensions:

$$
\text{output}_c = \frac{1}{H \times W} \sum_{i=0}^{H-1} \sum_{j=0}^{W-1} \text{input}[i][j][c]
$$

For input H x W x C, output is 1 x 1 x C (or just a vector of length C).

**Benefits:**
- No parameters (unlike fully connected layer)
- Works with any input size
- Reduces overfitting
- Often followed by single FC layer for classification

---

## Implementation Notes

**Division placement:**
- Divide by p^2 at the end (after summing)
- Or accumulate running mean
- Both give same result

**Floating point output:**
- Unlike max pooling, average pooling typically produces floats
- Even if input is integers, output is usually float

**Handling remainders:**
- If H is not divisible by p, extra rows/columns are ignored
- Same truncation behavior as max pooling

---

## Numerical Example with Stride

For general stride $s$ (not equal to pool size):

$$
\text{output}[i][j] = \frac{1}{p^2} \sum_{a=0}^{p-1} \sum_{b=0}^{p-1} \text{input}[i \cdot s + a][j \cdot s + b]
$$

This allows overlapping windows (s < p) or gaps (s > p), similar to max pooling.

---

## Average Pooling in Modern Architectures

**ResNet:**
- Uses global average pooling before final FC
- Followed by single 1000-way classifier

**EfficientNet:**
- Global average pooling at the end
- Followed by dropout and final FC

**MobileNet:**
- Global average pooling
- Designed for efficiency

The trend is: minimal use of intermediate average pooling, but global average pooling at the end is standard.
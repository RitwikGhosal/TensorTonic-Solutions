## What Is Max Pooling?

Max pooling is a downsampling operation that reduces the spatial size of feature maps. It works by:
1. Dividing the input into non-overlapping rectangular regions
2. Taking the maximum value from each region

The result is a smaller feature map that retains the most prominent features.

---

## Why Use Max Pooling?

**Dimensionality reduction:**
- Reduces computation for subsequent layers
- A 2x2 pool with stride 2 reduces spatial dimensions by 75%

**Translation invariance:**
- Small shifts in the input produce the same output
- If the max value shifts within a pool region, the output is unchanged

**Feature selection:**
- Keeps only the strongest activation in each region
- Discards weaker activations (which may be noise)

---

## The Max Pooling Operation

For a pool size of $p \times p$ with stride equal to pool size:

$$
\text{output}[i][j] = \max_{0 \le a < p,; 0 \le b < p} \text{input}[i \cdot p + a][j \cdot p + b]
$$

This takes the maximum over each non-overlapping $p \times p$ window.

---

## Step-by-Step Example

**Input (4x4):**

1  3  2  4
5  6  7  8
9  2  1  3
4  5  6  7

**Pool size: 2x2**

**Top-left window:** max(1, 3, 5, 6) = 6
**Top-right window:** max(2, 4, 7, 8) = 8
**Bottom-left window:** max(9, 2, 4, 5) = 9
**Bottom-right window:** max(1, 3, 6, 7) = 7

**Output (2x2):**

6  8
9  7

The output is 1/4 the size of the input.

---

## Output Dimensions

For input size $H \times W$, pool size $p$, and stride $s$:

$$
H_{out} = \left\lfloor \frac{H - p}{s} \right\rfloor + 1
$$

$$
W_{out} = \left\lfloor \frac{W - p}{s} \right\rfloor + 1
$$

For non-overlapping pooling (stride = pool size):

$$
H_{out} = \left\lfloor \frac{H}{p} \right\rfloor
$$

---

## Common Configurations

**2x2 pool, stride 2:**
- Most common configuration
- Reduces each dimension by half
- 75% reduction in spatial size
- Used in VGG, AlexNet

**3x3 pool, stride 2:**
- Overlapping pooling
- Slightly different receptive field
- Used in some older architectures

**Global max pooling:**
- Pool size equals entire feature map
- Produces a single value per channel
- Often used before fully connected layers

---

## Max Pooling vs. Average Pooling

**Max pooling:**
- Takes maximum value
- Preserves strongest activations
- Good for detecting presence of features
- More commonly used in classification

**Average pooling:**
- Takes mean value
- Smooths activations
- Good for preserving overall information
- Sometimes preferred in final layers

In practice, max pooling is more common in CNNs because it better preserves distinctive features.

---

## The Gradient (Backpropagation)

During backpropagation, the gradient flows only through the maximum element:

**Forward pass:**
- Record which element was the maximum (the "mask")

**Backward pass:**
- Gradient at the max position: equals incoming gradient
- Gradient at other positions: zero

This is why max pooling creates sparse gradients.

---

## Max Pooling in Modern Architectures

**Classic CNNs (VGG, AlexNet):**
- Heavy use of 2x2 max pooling
- Multiple pooling layers throughout

**ResNet:**
- One max pooling early on
- Relies more on strided convolutions

**Modern trend:**
- Less max pooling, more strided convolutions
- Strided conv learns how to downsample
- Some architectures eliminate pooling entirely

---

## Handling Edge Cases

When input size is not divisible by pool size:

**Option 1: Truncate**
- Only pool complete windows
- Some input values ignored
- Output is floor(H/p)

**Option 2: Pad**
- Add zeros or replicate border
- All input values contribute
- May introduce artifacts

Most implementations use truncation (floor division).

---

## Multi-Channel Max Pooling

For inputs with multiple channels:
- Apply max pooling independently to each channel
- No interaction between channels
- Output has same number of channels as input

If input is H x W x C, output is (H/p) x (W/p) x C.
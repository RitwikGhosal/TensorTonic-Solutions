## What Global Average Pooling Does

Global Average Pooling (GAP) takes a feature map and computes the average of all spatial positions for each channel. It collapses the spatial dimensions (height and width) into a single value per channel.

**Input:** Feature map of shape (channels, height, width)

**Output:** Vector of shape (channels,)

Each output value is the mean of all pixels in that channel's feature map.

---

## The Formula

For a feature map $F$ with shape $(C, H, W)$:

$$
\text{GAP}(F)_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} F_{c,i,j}
$$

Each channel $c$ produces one output value: the average of its $H \times W$ spatial locations.

---

## Worked Example

**Input feature map:** Shape (2, 3, 3) - 2 channels, 3x3 spatial

Channel 0:

$$
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{bmatrix}
$$

Channel 1:

$$
\begin{bmatrix}
9 & 8 & 7 \\
6 & 5 & 4 \\
3 & 2 & 1
\end{bmatrix}
$$

**Global Average Pooling:**

Channel 0: $(1+2+3+4+5+6+7+8+9) / 9 = 45 / 9 = 5.0$

Channel 1: $(9+8+7+6+5+4+3+2+1) / 9 = 45 / 9 = 5.0$

**Output:** $[5.0, 5.0]$

---

## Why Use Global Average Pooling?

**1. Reduces spatial dimensions completely**

Converts any spatial size to a fixed-length vector. A 7x7 feature map and a 14x14 feature map both become a single vector per channel.

**2. No learnable parameters**

Unlike fully connected layers, GAP has no weights to train. This reduces overfitting and model size.

**3. Spatial invariance**

The output is the same regardless of where features appear in the image. Good for classification where position does not matter.

**4. Interpretability**

Each channel's average can be interpreted as "how much of this feature is present in the image."

---

## GAP vs. Flatten + Fully Connected

**Traditional approach (pre-2014):**

Conv layers -> Flatten -> FC layer -> Output

For input 7x7x512: Flatten gives 25,088 values. FC to 1000 classes needs 25 million parameters!

**Modern approach with GAP:**

Conv layers -> GAP -> FC layer -> Output

For input 7x7x512: GAP gives 512 values. FC to 1000 classes needs only 512,000 parameters.

**Reduction:** 50x fewer parameters in the classifier head.

---

## GAP in CNN Architectures

**GoogLeNet/Inception (2014):**
First major architecture to use GAP instead of flatten. Dramatically reduced parameters.

**ResNet (2015):**
Uses GAP before the final classification layer.

**Modern architectures:**
Nearly all CNNs now use GAP. It is the standard approach.

**Typical structure:**

Conv/ResBlocks -> Final Conv (channels = num_features) -> GAP -> FC (num_classes)

---

## Global Max Pooling

An alternative that takes the maximum instead of average:

$$
\text{GMP}(F)_c = \max_{i,j} F_{c,i,j}
$$

**GAP:** "How much of this feature is present overall?"

**GMP:** "Is this feature strongly present anywhere?"

GAP is more commonly used. GMP can be more sensitive to localized features.

---

## Handling Different Input Sizes

GAP naturally handles variable input sizes:

- Input 224x224 -> after convs: 7x7x512 -> GAP: 512
- Input 448x448 -> after convs: 14x14x512 -> GAP: 512

Same output size regardless of input resolution. This enables:
- Training on one size, testing on another
- Multi-scale testing
- Flexible deployment

---

## GAP for Feature Extraction

When using a pretrained CNN as a feature extractor:

1. Remove the final classification layer
2. Keep everything up to and including GAP
3. The GAP output is your feature vector

This gives a compact, fixed-size representation of any image.

**Example:** ResNet50 produces a 2048-dimensional feature vector via GAP.

---

## Spatial Information Loss

**Trade-off:** GAP discards all spatial information. The output only knows "what" features are present, not "where."

**When this is fine:**
- Image classification ("Is this a cat?")
- Feature extraction for similarity

**When this is a problem:**
- Object detection (need to know where)
- Segmentation (need pixel-level output)
- Pose estimation (need spatial structure)

For spatial tasks, avoid GAP or use it only for auxiliary branches.

---

## Class Activation Maps (CAM)

GAP enables a powerful visualization technique called Class Activation Maps.

If the architecture is: Conv -> GAP -> FC (weights $w$)

The class activation map for class $c$ is:

$$
\text{CAM}_c(x, y) = \sum_k w_{c,k} \cdot F_k(x, y)
$$

This shows which spatial regions contributed to classifying as class $c$. High values indicate important regions.

CAM works because GAP preserves the correspondence between spatial locations and the final prediction.

---

## Implementation

**For a single sample:**

Input: (C, H, W)

Output: mean over H and W dimensions -> (C,)

**For a batch:**

Input: (N, C, H, W)

Output: mean over H and W dimensions -> (N, C)

**Code logic:**
1. Sum all values in each channel
2. Divide by (H * W)

Or equivalently: reshape to (N, C, H*W), then mean over last dimension.

---

## Adaptive Average Pooling

A generalization where you specify the output size:

- AdaptiveAvgPool2d(1, 1): Global average pooling
- AdaptiveAvgPool2d(7, 7): Pool to 7x7 spatial size

The pooling kernel size is computed automatically based on input size.

Global Average Pooling is AdaptiveAvgPool2d with output size (1, 1).

---

## Gradient Through GAP

The gradient is simple: during backprop, the gradient flows equally to all spatial positions.

If $\frac{\partial L}{\partial y_c}$ is the gradient at the output:

$$
\frac{\partial L}{\partial F_{c,i,j}} = \frac{1}{H \times W} \frac{\partial L}{\partial y_c}
$$

Every spatial location receives the same gradient (scaled by 1/(H*W)).
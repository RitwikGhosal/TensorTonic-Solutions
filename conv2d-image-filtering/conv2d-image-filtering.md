## What Is Convolution?

Convolution is a mathematical operation that combines two signals to produce a third. In image processing, it slides a small matrix (the kernel) over an image, computing a weighted sum at each position.

The operation has two key components:
- **Image**: a 2D grid of pixel values
- **Kernel** (or filter): a small 2D matrix of weights (typically 3x3, 5x5, or 7x7)

At each position, the kernel is placed on top of the image, corresponding elements are multiplied, and all products are summed to produce one output value.

---

## The Convolution Formula

For an image $I$ and kernel $K$ of size $k_h \times k_w$:

$$
\text{output}[i][j] = \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} I[i+m][j+n] \cdot K[m][n]
$$

This is a simple dot product between the kernel and each image patch.

---

## Step-by-Step Example

**Image (4x4):**

1  2  3  4
5  6  7  8
9  10 11 12
13 14 15 16

**Kernel (2x2):**

1  0
0  1

**Computing output[0][0]:**
- Overlay kernel on top-left 2x2 patch of image
- Multiply elementwise: 1*1 + 2*0 + 5*0 + 6*1 = 1 + 0 + 0 + 6 = 7

**Computing output[0][1]:**
- Slide kernel one position right
- Multiply: 2*1 + 3*0 + 6*0 + 7*1 = 2 + 0 + 0 + 7 = 9

Continue sliding to fill the entire output.

---

## Padding

Without padding, the output is smaller than the input:
- Input: H x W
- Kernel: k x k
- Output: (H - k + 1) x (W - k + 1)

Padding adds zeros around the image border to control output size:

**No padding (valid):**
- Output shrinks
- Loses information at edges

**Same padding:**
- Add enough zeros so output equals input size
- For 3x3 kernel: pad = 1 on each side

**Full padding:**
- Add k-1 zeros on each side
- Output is larger than input

---

## Stride

Stride controls how many pixels the kernel moves between positions:

**Stride = 1:**
- Kernel moves one pixel at a time
- Maximum overlap between patches
- Largest output

**Stride = 2:**
- Kernel moves two pixels at a time
- Less overlap
- Output is roughly half the size in each dimension

**Output size formula:**

$$
H_{out} = \left\lfloor \frac{H + 2p - k_h}{s} \right\rfloor + 1
$$

$$
W_{out} = \left\lfloor \frac{W + 2p - k_w}{s} \right\rfloor + 1
$$

Where $p$ is padding and $s$ is stride.

---

## What Different Kernels Do

**Identity (no change):**

0 0 0
0 1 0
0 0 0

**Edge detection (horizontal):**

-1 -1 -1
 0  0  0
 1  1  1

**Edge detection (vertical):**

-1 0 1
-1 0 1
-1 0 1

**Blur (averaging):**

1/9 1/9 1/9
1/9 1/9 1/9
1/9 1/9 1/9

**Sharpen:**

 0 -1  0
-1  5 -1
 0 -1  0

Each kernel extracts different features from the image.

---

## Convolution in Neural Networks

In CNNs, convolution learns to detect features:

**Layer 1:** Learns simple patterns (edges, colors)
**Layer 2:** Combines layer 1 features (corners, textures)
**Layer 3+:** Learns complex patterns (objects, faces)

Key differences from classical image processing:
- Kernel values are **learned** from data, not hand-designed
- Multiple kernels per layer create multiple feature maps
- Nonlinear activation (ReLU) applied after convolution

---

## Multi-Channel Convolution

Real images have 3 channels (RGB). The kernel also has 3 channels:

**Input:** H x W x 3
**Kernel:** k x k x 3

The convolution sums across all channels:

$$
\text{output}[i][j] = \sum_{c=0}^{C-1} \sum_{m} \sum_{n} I[i+m][j+n][c] \cdot K[m][n][c]
$$

To produce multiple output channels, use multiple kernels. For N output channels, need N kernels each of size k x k x C.

---

## Why Convolution Works for Images

**Local connectivity:**
- Each output depends only on a small input region
- Captures local patterns like edges and textures
- Much fewer parameters than fully connected layers

**Weight sharing:**
- Same kernel applied at every position
- A pattern detector works everywhere in the image
- Translation equivariance: if input shifts, output shifts the same way

**Hierarchical features:**
- Stack multiple conv layers
- Early layers: local features
- Later layers: global features

---

## Common Convolution Configurations

**3x3 kernel, stride 1, padding 1:**
- Most common in modern architectures
- Output same size as input
- VGG, ResNet use this extensively

**1x1 kernel:**
- Does not look at spatial neighbors
- Mixes information across channels
- Used in Inception, ResNet bottlenecks

**Large kernels (7x7, 11x11):**
- Used in early layers of some architectures
- Captures larger receptive field
- More parameters, less common now
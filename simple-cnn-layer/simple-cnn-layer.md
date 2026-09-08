## What a Convolutional Layer Does

A convolutional layer applies learnable filters (kernels) to an input image or feature map. Each filter slides across the input, computing dot products at each position to produce an output feature map.

**Key idea:** Instead of connecting every input to every output (like a fully connected layer), convolution uses local connections with shared weights. This dramatically reduces parameters and captures spatial patterns.

---

## The Convolution Operation

For a 2D convolution with a single input channel:

$$
\text{output}[i, j] = \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} \text{input}[i+m, j+n] \times \text{kernel}[m, n] + \text{bias}
$$

The kernel slides across the input. At each position, we compute the element-wise product of the kernel and the input patch, sum them up, and add a bias.

---

## Components of a Conv Layer

**Input:** Feature map of shape $(C_{in}, H_{in}, W_{in})$
- $C_{in}$: number of input channels (e.g., 3 for RGB)
- $H_{in}$: input height
- $W_{in}$: input width

**Kernel/Filter:** Weight tensor of shape $(C_{out}, C_{in}, k_h, k_w)$
- $C_{out}$: number of output channels (number of filters)
- $C_{in}$: must match input channels
- $k_h, k_w$: kernel height and width

**Bias:** Vector of shape $(C_{out},)$, one bias per output channel

**Output:** Feature map of shape $(C_{out}, H_{out}, W_{out})$

---

## Output Size Calculation

For input size $H_{in}$, kernel size $k$, padding $p$, and stride $s$:

$$
H_{out} = \left\lfloor \frac{H_{in} + 2p - k}{s} \right\rfloor + 1
$$

Same formula applies for width.

**Common configurations:**

**"Same" padding (output = input size):** $p = (k-1)/2$ with $s = 1$

**"Valid" padding (no padding):** $p = 0$, output shrinks

---

## Worked Example: Single Channel

**Input:** 4x4 image

$$
\begin{bmatrix}
1 & 2 & 3 & 0 \\
4 & 5 & 6 & 1 \\
7 & 8 & 9 & 2 \\
0 & 1 & 2 & 3
\end{bmatrix}
$$

**Kernel:** 3x3

$$
\begin{bmatrix}
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1
\end{bmatrix}
$$

**Stride:** 1, **Padding:** 0

**Output size:** $(4 - 3)/1 + 1 = 2$, so 2x2 output

**Position (0,0):** Extract top-left 3x3 patch, dot product with kernel:

$(1 \times 1) + (2 \times 0) + (3 \times -1) + (4 \times 1) + (5 \times 0) + (6 \times -1) + (7 \times 1) + (8 \times 0) + (9 \times -1)$

$= 1 + 0 - 3 + 4 + 0 - 6 + 7 + 0 - 9 = -6$

**Position (0,1):** Shift kernel right by 1, repeat...

---

## Multiple Input Channels

For RGB images (3 channels), each filter has 3 sub-kernels, one per channel:

$$
\text{output}[i,j] = \sum_{c=0}^{C_{in}-1} \sum_{m,n} \text{input}[c, i+m, j+n] \times \text{kernel}[c, m, n] + \text{bias}
$$

The convolutions across all input channels are summed to produce one output value.

**Example:** 3x3 kernel on RGB input has shape (3, 3, 3) = 27 weights per filter.

---

## Multiple Output Channels

Each output channel comes from a different filter. With $C_{out}$ filters:

- Filter 1 produces output channel 1
- Filter 2 produces output channel 2
- ...
- Filter $C_{out}$ produces output channel $C_{out}$

**Total weights:** $C_{out} \times C_{in} \times k_h \times k_w$

**Total biases:** $C_{out}$

---

## Stride

Stride controls how far the kernel moves at each step.

**Stride = 1:** Move 1 pixel, maximum overlap

**Stride = 2:** Move 2 pixels, output size halved

Larger stride reduces output size and computation but loses spatial resolution.

---

## Padding

Padding adds zeros around the input border.

**No padding (valid):**
- Kernel must fit entirely inside input
- Output smaller than input

**Same padding:**
- Pad so output has same spatial size as input (for stride 1)
- For 3x3 kernel: pad = 1
- For 5x5 kernel: pad = 2

Padding preserves spatial dimensions and allows edges to be processed.

---

## Parameter Count

For a conv layer: $(C_{in}, H, W) \rightarrow (C_{out}, H', W')$ with kernel $k \times k$:

**Weights:** $C_{out} \times C_{in} \times k \times k$

**Biases:** $C_{out}$

**Total:** $C_{out} \times (C_{in} \times k \times k + 1)$

**Example:** 64 filters, 3 input channels, 3x3 kernel

Weights: $64 \times 3 \times 3 \times 3 = 1,728$

Biases: $64$

Total: $1,792$ parameters

Compare to fully connected: A 32x32x3 input to 64 outputs would need $32 \times 32 \times 3 \times 64 = 196,608$ parameters!

---

## What Filters Learn

Early layers learn low-level features:
- Edge detectors (horizontal, vertical, diagonal)
- Color blobs
- Texture patterns

Deeper layers learn higher-level features:
- Shapes (circles, corners)
- Object parts (eyes, wheels)
- Full objects (faces, cars)

The network automatically learns useful filters through backpropagation.

---

## Activation Function

Convolution is a linear operation. After convolution, apply a nonlinear activation:

$$
\text{output} = \text{ReLU}(\text{Conv}(\text{input}))
$$

Common pattern: Conv -> BatchNorm -> ReLU

The activation allows the network to learn nonlinear patterns.

---

## Implementation Considerations

**im2col:** Convert convolution to matrix multiplication for efficiency. Extract all patches into columns, multiply by reshaped kernel.

**Parallelization:** Each output position can be computed independently. Highly parallelizable on GPUs.

**Memory:** Feature maps can be large. Trade-off between storing activations (for backprop) and recomputing them.
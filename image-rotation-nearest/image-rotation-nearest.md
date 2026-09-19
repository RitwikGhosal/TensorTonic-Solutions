## Image Rotation Basics

Rotating an image by an arbitrary angle requires:
1. Computing where each output pixel maps to in the input
2. Sampling the input at that (usually non-integer) location
3. Filling the output with the sampled values

The rotation is performed around the image center.

---

## Rotation Mathematics

To rotate a point $(x, y)$ by angle $\theta$ around the origin:

$$
x' = x \cos\theta - y \sin\theta
$$

$$
y' = x \sin\theta + y \cos\theta
$$

For rotation around the image center $(c_x, c_y)$:
1. Translate to origin: $(x - c_x, y - c_y)$
2. Rotate
3. Translate back: add $(c_x, c_y)$

---

## Inverse Mapping

For each output pixel, we need to find which input pixel it comes from. This requires the **inverse** rotation (rotate by $-\theta$):

For output pixel $(i, j)$:
1. Compute offset from center: $dx = j - c_x$, $dy = i - c_y$
2. Apply inverse rotation:

$$
\text{src}_x = c_x + dx \cos\theta + dy \sin\theta
$$

$$
\text{src}_y = c_y - dx \sin\theta + dy \cos\theta
$$
3. Sample the input at $(\text{src}_y, \text{src}_x)$

---

## Nearest Neighbor Interpolation

The source coordinates are usually not integers. Nearest neighbor interpolation simply rounds to the closest pixel:

$$
i_{\text{src}} = \text{round}(\text{src}_y)
$$

$$
j_{\text{src}} = \text{round}(\text{src}_x)
$$

If the rounded coordinates are within bounds, copy that pixel. Otherwise, fill with 0 (or another background value).

---

## Step-by-Step Example

**Image (3x3):**

1 2 3
4 5 6
7 8 9

**Rotation: 90 degrees counterclockwise**

Center: (1, 1)

**For output pixel (0, 0):**
- dx = 0 - 1 = -1, dy = 0 - 1 = -1
- cos(90) = 0, sin(90) = 1
- src_x = 1 + (-1)*0 + (-1)*1 = 0
- src_y = 1 - (-1)*1 + (-1)*0 = 2
- Source: (2, 0) = 7

**For output pixel (0, 2):**
- dx = 2 - 1 = 1, dy = 0 - 1 = -1
- src_x = 1 + (1)*0 + (-1)*1 = 0
- src_y = 1 - (1)*1 + (-1)*0 = 0
- Source: (0, 0) = 1

**Result:**

3 6 9
2 5 8
1 4 7

(This is the expected 90-degree CCW rotation)

---

## Why Inverse Mapping?

**Forward mapping:**
- For each input pixel, compute where it goes in output
- Problem: multiple inputs may map to same output (conflicts)
- Problem: some outputs may have no input (holes)

**Inverse mapping:**
- For each output pixel, find which input it comes from
- Guarantees every output gets a value
- No holes, no conflicts

Inverse mapping is the standard approach for image transformations.

---

## Nearest Neighbor vs. Other Methods

**Nearest neighbor:**
- Fast (no interpolation computation)
- Produces blocky/aliased results
- Good for binary images or pixel art
- Preserves exact pixel values

**Bilinear interpolation:**
- Weighted average of 4 nearest pixels
- Smoother results
- Slight blurring
- Most common choice

**Bicubic interpolation:**
- Weighted average of 16 pixels
- Even smoother
- Better for high-quality resizing
- More computation

---

## Handling Out-of-Bounds

When the source coordinates fall outside the input image:

**Zero fill:**
- Output pixel is 0 (black)
- Creates black corners after rotation

**Border replication:**
- Use nearest valid pixel
- Extends edges

**Wrap around:**
- Treat image as tiling
- Rarely used for rotation

This problem uses zero fill.

---

## Rotation Angle Convention

**Counterclockwise (positive angle):**
- Standard mathematical convention
- 90 degrees: right side moves up
- Used in most math contexts

**Clockwise (positive angle):**
- Used in some graphics systems
- 90 degrees: right side moves down

This problem uses counterclockwise rotation.

---

## Image Center

The center of rotation affects the result:

**Pixel center convention:**

$$
c_x = \frac{W - 1}{2}, \quad c_y = \frac{H - 1}{2}
$$

For a 5x5 image: center is (2, 2)
For a 4x4 image: center is (1.5, 1.5)

This places the center at the middle pixel (or between pixels for even sizes).

---

## Aliasing Artifacts

Nearest neighbor rotation produces visible artifacts:
- Jagged edges (staircase effect)
- "Dancing" pixels in animations
- Loss of fine detail

These artifacts are worse for:
- Small rotation angles
- High-frequency content (text, sharp edges)
- Repeated rotations

For better quality, use bilinear or bicubic interpolation.
## What Is Rotation Around the Z-Axis?

Rotation around the Z-axis spins a point or vector in the XY-plane while keeping the Z-coordinate unchanged. Looking down the positive Z-axis (toward negative Z), a positive angle rotates counterclockwise.

This is one of the three fundamental rotations in 3D space, along with rotations around X and Y axes.

---

## The Rotation Matrix

The 3x3 rotation matrix for angle $\theta$ around the Z-axis is:

$$
R_z(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{pmatrix}
$$

To rotate a point $\mathbf{p} = (x, y, z)^T$:

$$
\mathbf{p}' = R_z(\theta) \mathbf{p}
$$

---

## Deriving the Formula

Consider a point $(x, y)$ in the XY-plane. In polar coordinates:

$$
x = r\cos\phi, \quad y = r\sin\phi
$$

where $r$ is the distance from origin and $\phi$ is the current angle.

After rotating by $\theta$, the new angle is $\phi + \theta$:

$$
x' = r\cos(\phi + \theta) = r(\cos\phi\cos\theta - \sin\phi\sin\theta) = x\cos\theta - y\sin\theta
$$

$$
y' = r\sin(\phi + \theta) = r(\sin\phi\cos\theta + \cos\phi\sin\theta) = x\sin\theta + y\cos\theta
$$

The Z-coordinate is unchanged: $z' = z$

---

## The Expanded Equations

For a point $(x, y, z)$ rotated by angle $\theta$ around Z:

$$
x' = x\cos\theta - y\sin\theta
$$

$$
y' = x\sin\theta + y\cos\theta
$$

$$
z' = z
$$

Or in matrix form:

$$
\begin{pmatrix} x' \\ y' \\ z' \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} x \\ y \\ z \end{pmatrix}
$$

---

## Worked Example: 90 Degree Rotation

**Point:** $(1, 0, 5)$

**Angle:** $\theta = 90° = \frac{\pi}{2}$

**Trigonometric values:**
- $\cos(90°) = 0$
- $\sin(90°) = 1$

**Calculation:**

$$
x' = 1 \cdot 0 - 0 \cdot 1 = 0
$$

$$
y' = 1 \cdot 1 + 0 \cdot 0 = 1
$$

$$
z' = 5
$$

**Result:** $(0, 1, 5)$

The point moved from the positive X-axis to the positive Y-axis.

---

## Worked Example: 45 Degree Rotation

**Point:** $(2, 0, 3)$

**Angle:** $\theta = 45° = \frac{\pi}{4}$

**Trigonometric values:**
- $\cos(45°) = \frac{\sqrt{2}}{2} \approx 0.707$
- $\sin(45°) = \frac{\sqrt{2}}{2} \approx 0.707$

**Calculation:**

$$
x' = 2 \cdot 0.707 - 0 \cdot 0.707 = 1.414
$$

$$
y' = 2 \cdot 0.707 + 0 \cdot 0.707 = 1.414
$$

$$
z' = 3
$$

**Result:** $(1.414, 1.414, 3)$

---

## Worked Example: General Point

**Point:** $(3, 4, 2)$

**Angle:** $\theta = 30° = \frac{\pi}{6}$

**Trigonometric values:**
- $\cos(30°) = \frac{\sqrt{3}}{2} \approx 0.866$
- $\sin(30°) = 0.5$

**Calculation:**

$$
x' = 3 \cdot 0.866 - 4 \cdot 0.5 = 2.598 - 2 = 0.598
$$

$$
y' = 3 \cdot 0.5 + 4 \cdot 0.866 = 1.5 + 3.464 = 4.964
$$

$$
z' = 2
$$

**Result:** $(0.598, 4.964, 2)$

---

## Sign Convention

**Positive angle (counterclockwise):**

Looking down the Z-axis (from positive Z toward origin):
- Positive X rotates toward positive Y
- Positive Y rotates toward negative X

**Negative angle (clockwise):**

The opposite direction. Equivalent to rotating by $-\theta$.

---

## Special Angles

**$\theta = 0°$:** No rotation, identity matrix

**$\theta = 90°$:** $(x, y, z) \to (-y, x, z)$

**$\theta = 180°$:** $(x, y, z) \to (-x, -y, z)$

**$\theta = 270°$ or $-90°$:** $(x, y, z) \to (y, -x, z)$

**$\theta = 360°$:** Back to original, identity

---

## Properties of Rotation Matrices

**Orthogonal:**

$$
R_z^T R_z = I
$$

**Determinant equals 1:**

$$
\det(R_z) = \cos^2\theta + \sin^2\theta = 1
$$

**Inverse is transpose:**

$$
R_z^{-1}(\theta) = R_z^T(\theta) = R_z(-\theta)
$$

**Preserves length:**

$$
||R_z \mathbf{v}|| = ||\mathbf{v}||
$$

---

## Composing Rotations

Rotating by $\theta_1$ then $\theta_2$ around Z:

$$
R_z(\theta_1 + \theta_2) = R_z(\theta_2) R_z(\theta_1)
$$

Note: Matrix multiplication is right to left, but angles add.

For rotation around Z, order does not matter since all rotations share the same axis.

---

## Rotation Around Other Axes

**Around X-axis:**

$$
R_x(\theta) = \begin{pmatrix} 1 & 0 & 0 \\ 0 & \cos\theta & -\sin\theta \\ 0 & \sin\theta & \cos\theta \end{pmatrix}
$$

**Around Y-axis:**

$$
R_y(\theta) = \begin{pmatrix} \cos\theta & 0 & \sin\theta \\ 0 & 1 & 0 \\ -\sin\theta & 0 & \cos\theta \end{pmatrix}
$$

Note: $R_y$ has opposite sign pattern due to the cyclic nature of cross products.

---

## Euler Angles

Any 3D rotation can be decomposed into three rotations around coordinate axes.

**ZYX (aerospace):**

$$
R = R_z(\psi) R_y(\theta) R_x(\phi)
$$

**XYZ (common in graphics):**

$$
R = R_x(\phi) R_y(\theta) R_z(\psi)
$$

The order matters! Different orders give different rotations.

---

## Axis-Angle to Matrix

For rotation of angle $\theta$ around Z-axis specifically, we have the matrix above.

For arbitrary axis $\hat{\mathbf{k}}$, use Rodrigues' rotation formula:

$$
R = I + (\sin\theta)K + (1 - \cos\theta)K^2
$$

where $K$ is the skew-symmetric matrix of $\hat{\mathbf{k}}$.

---

## Rotating Multiple Points

To rotate many points efficiently:

1. Precompute $\cos\theta$ and $\sin\theta$ once
2. Apply the formulas to each point
3. Or use matrix-vector multiplication with batched operations

For $n$ points, this is $O(n)$ after constant-time setup.

---

## Homogeneous Coordinates

In 4x4 homogeneous form (for combining with translations):

$$
R_z^{4\times4}(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta & 0 & 0 \\ \sin\theta & \cos\theta & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}
$$

This can be composed with translation matrices.

---

## Numerical Stability

**Near-zero angles:**

When $\theta \approx 0$:
- $\cos\theta \approx 1$
- $\sin\theta \approx \theta$

Use Taylor series for better precision if needed.

**Accumulated rotations:**

After many small rotations, the matrix may drift from being perfectly orthogonal. Re-orthogonalize periodically.

---

## Applications

**3D Graphics:**
- Camera rotation
- Object transformation
- Animation systems

**Robotics:**
- Joint rotations
- End-effector orientation
- Path planning

**Computer Vision:**
- Image rotation
- Camera calibration
- 3D reconstruction

**Physics Simulation:**
- Rigid body dynamics
- Angular momentum

---

## Quaternion Alternative

Rotations can also be represented as quaternions:

$$
q = \cos\frac{\theta}{2} + \sin\frac{\theta}{2}(0i + 0j + 1k)
$$

For Z-axis rotation, only the scalar and $k$ components are non-zero.

Quaternions avoid gimbal lock and interpolate smoothly.
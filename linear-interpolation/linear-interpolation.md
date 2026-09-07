## What is Linear Interpolation?

Linear interpolation is a method for estimating unknown values that fall between two known data points. It assumes the value changes linearly (in a straight line) between the known points. Given two points $(x_0, y_0)$ and $(x_1, y_1)$, linear interpolation finds the $y$ value for any $x$ between $x_0$ and $x_1$.

---

## Why Linear Interpolation Matters

**Filling missing data**: Time series with gaps can be filled by interpolating between adjacent observations.

**Resampling**: Converting data from one sampling rate to another (e.g., hourly to every 15 minutes).

**Function approximation**: When the true function is unknown, linear interpolation provides a simple estimate.

**Computer graphics**: Smoothly transitioning colors, positions, or other properties between keyframes.

**Numerical methods**: Building blocks for more complex interpolation schemes.

---

## The Linear Interpolation Formula

Given two known points $(x_0, y_0)$ and $(x_1, y_1)$, the interpolated value at position $x$ is:

$$
y = y_0 + (y_1 - y_0) \cdot \frac{x - x_0}{x_1 - x_0}
$$

This can be rewritten in several equivalent forms:

**Slope-intercept form**:

$$
y = y_0 + m \cdot (x - x_0) \quad \text{where } m = \frac{y_1 - y_0}{x_1 - x_0}
$$

**Weighted average form**:

$$
y = (1 - t) \cdot y_0 + t \cdot y_1 \quad \text{where } t = \frac{x - x_0}{x_1 - x_0}
$$

The parameter $t$ represents how far along the interval we are:
- $t = 0$ gives $y = y_0$ (at the start)
- $t = 1$ gives $y = y_1$ (at the end)
- $t = 0.5$ gives the midpoint

---

## Geometric Interpretation

Linear interpolation connects the two known points with a straight line segment. The interpolated value is simply the $y$-coordinate on this line at the desired $x$ position.

The slope of this line is:

$$
m = \frac{\Delta y}{\Delta x} = \frac{y_1 - y_0}{x_1 - x_0}
$$

This represents the rate of change between the two points.

---

## Worked Example

**Known points**: $(2, 10)$ and $(8, 40)$

**Find**: The interpolated value at $x = 5$

**Step 1 - Calculate the interpolation parameter**:

$$
t = \frac{x - x_0}{x_1 - x_0} = \frac{5 - 2}{8 - 2} = \frac{3}{6} = 0.5
$$

**Step 2 - Apply the formula**:

$$
y = (1 - 0.5) \cdot 10 + 0.5 \cdot 40 = 5 + 20 = 25
$$

**Verification**: At $x = 5$ (halfway between 2 and 8), the value is 25 (halfway between 10 and 40). This confirms the linear relationship.

---

## Interpolation vs Extrapolation

**Interpolation**: Estimating values within the range of known data (between $x_0$ and $x_1$). Generally reliable since we have data on both sides.

**Extrapolation**: Estimating values outside the range of known data ($x < x_0$ or $x > x_1$). Risky because we are projecting trends beyond observed data.

The same formula works for extrapolation, but the results become increasingly unreliable further from the known points.

---

## Piecewise Linear Interpolation

When you have multiple known points $(x_0, y_0), (x_1, y_1), ..., (x_n, y_n)$:

**Step 1**: Find which interval contains the query point $x$

**Step 2**: Apply linear interpolation using only the two endpoints of that interval

This creates a continuous function that passes through all data points, with "corners" at each known point where the slope changes.

**Example**: Points at x = [0, 2, 5, 10] with y = [0, 3, 2, 8]

To interpolate at x = 4:
- Find interval: 4 is between 2 and 5
- Use points (2, 3) and (5, 2)
- Calculate: $y = 3 + (2-3) \cdot \frac{4-2}{5-2} = 3 - \frac{2}{3} = 2.33$

---

## Properties of Linear Interpolation

**Exact at data points**: The interpolant passes through all known points exactly.

**Continuity**: The resulting function is continuous (no jumps) but not smooth (has corners at data points).

**Local**: Changing one data point only affects the interpolant in adjacent intervals.

**First-order accuracy**: Error is proportional to the square of the interval width and the second derivative of the true function.

$$
\text{Error} \approx \frac{(x_1 - x_0)^2}{8} \cdot |f'(\xi)|
$$

Where $\xi$ is some point in the interval.

---

## Handling Edge Cases

**Query outside data range**: Options include:
- Return the nearest endpoint value (constant extrapolation)
- Apply the formula (linear extrapolation)
- Return NaN or raise an error

**Single data point**: Interpolation is undefined with only one point. Return that value as constant or raise an error.

**Duplicate x-values**: If two points have the same $x$ but different $y$, the function is not well-defined. Handle by averaging, taking the first/last value, or raising an error.

---

## Comparison with Other Interpolation Methods

**Nearest neighbor**: Returns the value of the closest known point. Discontinuous but simple.

**Polynomial interpolation**: Fits a single polynomial through all points. Smooth but can oscillate wildly (Runge's phenomenon).

**Spline interpolation**: Fits piecewise polynomials with smooth connections. Better than linear for smooth data but more complex.

**Cubic interpolation**: Uses cubic polynomials for smoother curves. Requires more computation but avoids the corners of linear interpolation.

---

## Where Linear Interpolation Shows Up

- **Time Series Analysis**: Filling missing timestamps in sensor data, stock prices, or weather measurements

- **Image Processing**: Resizing images, texture mapping, bilinear interpolation is 2D linear interpolation

- **Audio Processing**: Resampling audio signals to different sample rates

- **Animation**: Interpolating between keyframes for position, rotation, color

- **Scientific Computing**: Lookup tables with linear interpolation for fast function evaluation

- **Geographic Information Systems**: Estimating elevation or values between grid points

- **Machine Learning**: Linear interpolation in latent spaces for generative models

- **Finance**: Yield curve interpolation, estimating interest rates at non-standard maturities

- **Control Systems**: Signal reconstruction from sampled data

- **Game Development**: Smooth movement, camera transitions, color gradients

## The Optimization Problem

In machine learning, we minimize a loss function $f(x)$ where $x$ is a vector of parameters. Gradient descent updates parameters using:

$$
x_{k+1} = x_k - \alpha \nabla f(x_k)
$$

This only uses first-order information (the gradient). **Newton's method** uses second-order information (the Hessian matrix $H$):

$$
x_{k+1} = x_k - H^{-1} \nabla f(x_k)
$$

Newton's method converges faster but requires computing and inverting the Hessian, which is $O(n^2)$ storage and $O(n^3)$ computation for $n$ parameters.

---

## What L-BFGS Does

L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) approximates Newton's method without storing the full Hessian. It:

1. Stores only the last $m$ gradient differences and parameter differences (typically $m = 10$)
2. Implicitly represents the inverse Hessian approximation
3. Computes the search direction in $O(mn)$ time and $O(mn)$ space

This makes quasi-Newton optimization practical for problems with millions of parameters.

---

## The Key Quantities

At each iteration $k$, L-BFGS stores:

**$s_k$:** Parameter difference

$$
s_k = x_{k+1} - x_k
$$

**$y_k$:** Gradient difference

$$
y_k = \nabla f(x_{k+1}) - \nabla f(x_k)
$$

These pairs $(s_i, y_i)$ for $i = k-m+1, ..., k$ encode curvature information about the loss surface.

---

## The Two-Loop Recursion

The two-loop algorithm computes the search direction $r = H_k \nabla f(x_k)$ without explicitly forming $H_k$.

**Input:**
- Current gradient $q = \nabla f(x_k)$
- History of $m$ pairs: $(s_0, y_0), (s_1, y_1), ..., (s_{m-1}, y_{m-1})$
- Initial Hessian approximation $H_0$ (often $H_0 = \gamma I$ for scalar $\gamma$)

**Loop 1 (backward, from newest to oldest):**

For $i = m-1, m-2, ..., 0$:

$$
\rho_i = \frac{1}{y_i^T s_i}
$$

$$
\alpha_i = \rho_i s_i^T q
$$

$$
q = q - \alpha_i y_i
$$

**Middle step:**

$$
r = H_0 q
$$

Often $H_0 = \gamma I$ where $\gamma = \frac{s_{m-1}^T y_{m-1}}{y_{m-1}^T y_{m-1}}$

**Loop 2 (forward, from oldest to newest):**

For $i = 0, 1, ..., m-1$:

$$
\beta_i = \rho_i y_i^T r
$$

$$
r = r + s_i (\alpha_i - \beta_i)
$$

**Output:** $r$ is the search direction (approximation of $H_k \nabla f$)

---

## Worked Example

**Setup:** 2D problem, $m = 2$ history pairs

$q = [4, 6]$ (current gradient)

History:
- $(s_0, y_0) = ([1, 0], [2, 1])$
- $(s_1, y_1) = ([0.5, 0.5], [1, 2])$

**Precompute $\rho$ values:**

$\rho_0 = \frac{1}{y_0^T s_0} = \frac{1}{[2,1] \cdot [1,0]} = \frac{1}{2} = 0.5$

$\rho_1 = \frac{1}{y_1^T s_1} = \frac{1}{[1,2] \cdot [0.5,0.5]} = \frac{1}{1.5} \approx 0.667$

**Loop 1 (backward):**

$i = 1$:
$\alpha_1 = \rho_1 (s_1^T q) = 0.667 \times ([0.5, 0.5] \cdot [4, 6]) = 0.667 \times 5 = 3.33$

$q = q - \alpha_1 y_1 = [4, 6] - 3.33 \times [1, 2] = [4-3.33, 6-6.67] = [0.67, -0.67]$

$i = 0$:
$\alpha_0 = \rho_0 (s_0^T q) = 0.5 \times ([1, 0] \cdot [0.67, -0.67]) = 0.5 \times 0.67 = 0.33$

$q = q - \alpha_0 y_0 = [0.67, -0.67] - 0.33 \times [2, 1] = [0.67-0.67, -0.67-0.33] = [0, -1]$

**Middle step:**

Compute $\gamma = \frac{s_1^T y_1}{y_1^T y_1} = \frac{1.5}{5} = 0.3$

$r = \gamma q = 0.3 \times [0, -1] = [0, -0.3]$

**Loop 2 (forward):**

$i = 0$:
$\beta_0 = \rho_0 (y_0^T r) = 0.5 \times ([2, 1] \cdot [0, -0.3]) = 0.5 \times (-0.3) = -0.15$

$r = r + s_0 (\alpha_0 - \beta_0) = [0, -0.3] + [1, 0] \times (0.33 - (-0.15))$
$= [0, -0.3] + [0.48, 0] = [0.48, -0.3]$

$i = 1$:
$\beta_1 = \rho_1 (y_1^T r) = 0.667 \times ([1, 2] \cdot [0.48, -0.3]) = 0.667 \times (-0.12) = -0.08$

$r = r + s_1 (\alpha_1 - \beta_1) = [0.48, -0.3] + [0.5, 0.5] \times (3.33 - (-0.08))$
$= [0.48, -0.3] + [1.71, 1.71] = [2.19, 1.41]$

**Result:** Search direction $r = [2.19, 1.41]$

---

## Why Two Loops?

The algorithm exploits the recursive structure of the BFGS update formula. Instead of storing and multiplying large matrices, it:

1. **First loop:** Strips away the "outer" updates from newest to oldest
2. **Middle:** Applies the base inverse Hessian
3. **Second loop:** Rebuilds the "outer" updates from oldest to newest

This is mathematically equivalent to $H_k g$ but computed in $O(mn)$ instead of $O(n^2)$.

---

## The Initial Hessian $H_0$

The choice of $H_0$ affects convergence. Common choices:

**Identity:** $H_0 = I$ (simple but may not scale well)

**Scaled identity:** $H_0 = \gamma I$ where:

$$
\gamma = \frac{s_{k-1}^T y_{k-1}}{y_{k-1}^T y_{k-1}}
$$

This scaling adapts to the local curvature and is the standard choice.

---

## Memory Management

L-BFGS stores a **circular buffer** of $m$ pairs:

- When a new pair $(s_k, y_k)$ arrives and buffer is full
- Discard the oldest pair
- Add the new pair

Typical values: $m = 3$ to $20$. Larger $m$ gives better Hessian approximation but more computation and memory.

---

## When to Use L-BFGS

**Good for:**
- Convex optimization (logistic regression, linear SVM)
- Problems where function/gradient evaluation is expensive
- Medium to large problems where full Hessian is impractical

**Less suitable for:**
- Non-smooth functions
- Stochastic optimization (mini-batch gradients are noisy)
- Very large neural networks (SGD variants often work better)

---

## Comparison to Other Methods

**Gradient descent:** Simple but slow, linear convergence

**Newton:** Fast (quadratic convergence) but $O(n^3)$ per iteration

**BFGS:** Superlinear convergence, $O(n^2)$ storage

**L-BFGS:** Near-superlinear convergence, $O(mn)$ storage

L-BFGS is the go-to method for large-scale convex optimization.

---

## Numerical Considerations

**Curvature condition:** For the update to be valid, we need $y_k^T s_k > 0$. If violated, skip the update or restart.

**Numerical stability:** The $\rho_i$ values can be very large if $y_i^T s_i$ is small. Some implementations add safeguards.

**Line search:** L-BFGS is typically paired with a line search (Wolfe conditions) to determine step size.
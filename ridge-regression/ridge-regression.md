## What Is Ridge Regression?

Ridge regression is **linear regression with L2 regularization**. It adds a penalty term to the loss function that discourages large coefficient values, helping prevent overfitting.

$$
L(\beta) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} \beta_j^2
$$

where:
- $y_i$ is the target value for sample $i$
- $\hat{y}_i = \beta_0 + \sum_j \beta_j x_{ij}$ is the prediction
- $\lambda$ is the regularization strength
- $\beta_j$ are the coefficients (excluding the intercept)

---

## Why Regularize?

**Problem with ordinary least squares (OLS):**

When features are correlated (multicollinearity) or when $p > n$ (more features than samples), OLS:
- Has high variance in coefficient estimates
- May produce arbitrarily large coefficients
- Overfits the training data

**Ridge solution:**

The penalty term $\lambda \sum_j \beta_j^2$ shrinks coefficients toward zero, reducing variance at the cost of introducing some bias.

---

## The Ridge Objective

In matrix notation:

$$
L(\beta) = ||y - X\beta||^2 + \lambda ||\beta||^2
$$

where:
- $y$ is the target vector of shape $(n,)$
- $X$ is the design matrix of shape $(n, p)$
- $\beta$ is the coefficient vector of shape $(p,)$
- $||\cdot||^2$ is the squared L2 norm

---

## The Closed-Form Solution

Taking the derivative and setting to zero:

$$
\frac{\partial L}{\partial \beta} = -2X^T(y - X\beta) + 2\lambda\beta = 0
$$

$$
X^T X \beta + \lambda\beta = X^T y
$$

$$
(X^T X + \lambda I) \beta = X^T y
$$

**Ridge solution:**

$$
\hat{\beta} = (X^T X + \lambda I)^{-1} X^T y
$$

The addition of $\lambda I$ ensures the matrix is invertible, even when $X^T X$ is singular.

---

## Comparing to OLS

**Ordinary Least Squares:**

$$
\hat{\beta}_{OLS} = (X^T X)^{-1} X^T y
$$

**Ridge Regression:**

$$
\hat{\beta}_{Ridge} = (X^T X + \lambda I)^{-1} X^T y
$$

The difference is adding $\lambda I$ to the diagonal of $X^T X$.

- When $\lambda = 0$: Ridge equals OLS
- When $\lambda \to \infty$: Coefficients shrink to zero

---

## The Role of $\lambda$

**Small $\lambda$ (weak regularization):**
- Coefficients close to OLS solution
- Higher variance, lower bias
- Risk of overfitting

**Large $\lambda$ (strong regularization):**
- Coefficients shrink toward zero
- Lower variance, higher bias
- Risk of underfitting

**Optimal $\lambda$:**
- Found via cross-validation
- Balances bias and variance

---

## Worked Example

**Data:** 3 samples, 2 features

$$
X = \begin{bmatrix} 1 & 2 \\ 2 & 3 \\ 3 & 5 \end{bmatrix}, \quad y = \begin{bmatrix} 5 \\ 8 \\ 12 \end{bmatrix}
$$

**Step 1: Compute $X^T X$**

$$
X^T X = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 3 & 5 \end{bmatrix} \begin{bmatrix} 1 & 2 \\ 2 & 3 \\ 3 & 5 \end{bmatrix} = \begin{bmatrix} 14 & 23 \\ 23 & 38 \end{bmatrix}
$$

**Step 2: Add $\lambda I$ (let $\lambda = 1$)**

$$
X^T X + \lambda I = \begin{bmatrix} 14 & 23 \\ 23 & 38 \end{bmatrix} + \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 15 & 23 \\ 23 & 39 \end{bmatrix}
$$

**Step 3: Compute $X^T y$**

$$
X^T y = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 3 & 5 \end{bmatrix} \begin{bmatrix} 5 \\ 8 \\ 12 \end{bmatrix} = \begin{bmatrix} 57 \\ 94 \end{bmatrix}
$$

**Step 4: Solve $(X^T X + \lambda I)\beta = X^T y$**

$$
\begin{bmatrix} 15 & 23 \\ 23 & 39 \end{bmatrix} \beta = \begin{bmatrix} 57 \\ 94 \end{bmatrix}
$$

Inverse: $\det = 15 \times 39 - 23 \times 23 = 585 - 529 = 56$

$$
(X^T X + \lambda I)^{-1} = \frac{1}{56} \begin{bmatrix} 39 & -23 \\ -23 & 15 \end{bmatrix}
$$

$$
\hat{\beta} = \frac{1}{56} \begin{bmatrix} 39 & -23 \\ -23 & 15 \end{bmatrix} \begin{bmatrix} 57 \\ 94 \end{bmatrix}
$$

$$
= \frac{1}{56} \begin{bmatrix} 39 \times 57 - 23 \times 94 \\ -23 \times 57 + 15 \times 94 \end{bmatrix} = \frac{1}{56} \begin{bmatrix} 2223 - 2162 \\ -1311 + 1410 \end{bmatrix} = \frac{1}{56} \begin{bmatrix} 61 \\ 99 \end{bmatrix}
$$

$$
\hat{\beta} \approx \begin{bmatrix} 1.09 \\ 1.77 \end{bmatrix}
$$

---

## Geometric Interpretation

**OLS:** Find the point in the column space of $X$ closest to $y$.

**Ridge:** Find the point that balances closeness to $y$ (fitting the data) with staying close to the origin (small coefficients).

The ridge constraint $||\beta||^2 \leq t$ defines a **sphere** in coefficient space. Ridge finds the point on or inside this sphere that minimizes the residual sum of squares.

---

## Bias-Variance Tradeoff

**OLS:**
- Unbiased: $E[\hat{\beta}_{OLS}] = \beta_{true}$
- High variance when features are correlated

**Ridge:**
- Biased: $E[\hat{\beta}_{Ridge}] \neq \beta_{true}$ (shrunk toward zero)
- Lower variance

**Total error = Bias$^2$ + Variance**

Ridge can achieve lower total error by trading off increased bias for reduced variance.

---

## Ridge vs Lasso

**Ridge (L2 regularization):**
- Penalty: $\lambda \sum_j \beta_j^2$
- Shrinks all coefficients proportionally
- Does NOT set coefficients exactly to zero
- Handles correlated features well

**Lasso (L1 regularization):**
- Penalty: $\lambda \sum_j |\beta_j|$
- Can set coefficients exactly to zero (feature selection)
- Tends to select one from a group of correlated features

**Elastic Net:**
- Combines L1 and L2 penalties
- Balances properties of both

---

## Standardization

**Important:** Ridge penalizes coefficients, so features must be on the same scale.

**Before fitting:**
1. Center features: subtract mean
2. Scale features: divide by standard deviation

Without standardization, features with large values will have artificially small coefficients, receiving less penalty.

**The intercept** is typically not penalized and is computed separately after centering.

---

## Choosing $\lambda$ via Cross-Validation

**Procedure:**

1. Choose a grid of $\lambda$ values (e.g., $10^{-4}$ to $10^4$, logarithmically spaced)
2. For each $\lambda$:
   - Perform k-fold cross-validation
   - Compute average validation error
3. Select $\lambda$ with lowest validation error

**Common choices:**
- $\lambda$ with minimum CV error
- Largest $\lambda$ within 1 standard error of minimum (more regularization)

---

## SVD Solution

Using singular value decomposition $X = U\Sigma V^T$:

$$
\hat{\beta} = V (\Sigma^2 + \lambda I)^{-1} \Sigma U^T y
$$

This shows that ridge regression shrinks the contribution of small singular values more than large ones, stabilizing the solution.

**Effective degrees of freedom:**

$$
df(\lambda) = \sum_{j=1}^{p} \frac{\sigma_j^2}{\sigma_j^2 + \lambda}
$$

where $\sigma_j$ are the singular values. As $\lambda$ increases, effective degrees of freedom decreases.

---

## Kernel Ridge Regression

For non-linear relationships, use the kernel trick:

$$
\hat{y} = K(K + \lambda I)^{-1} y
$$

where $K$ is the kernel matrix: $K_{ij} = k(x_i, x_j)$.

This allows ridge regression in infinite-dimensional feature spaces without explicitly computing the features.

---

## Computational Complexity

**Direct solution:**
- Form $X^T X$: $O(np^2)$
- Add $\lambda I$ and invert: $O(p^3)$
- Multiply by $X^T y$: $O(p^2)$
- Total: $O(np^2 + p^3)$

**When $n < p$:**
- Use the dual formulation: $(XX^T + \lambda I)^{-1}$
- Complexity: $O(n^2 p + n^3)$

**Iterative methods:**
- Gradient descent, conjugate gradient
- Useful for very large problems

---

## Ridge Regression for Classification

Ridge can be adapted for classification:

**Approach 1:** Use ridge regression directly with binary targets (0/1 or -1/+1). Not ideal but sometimes works.

**Approach 2:** Logistic regression with L2 penalty (ridge logistic regression):

$$
L = -\sum_i [y_i \log p_i + (1-y_i) \log(1-p_i)] + \lambda ||\beta||^2
$$

This is more principled for classification tasks.
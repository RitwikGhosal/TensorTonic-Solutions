## What Are Eigenvalues?

An **eigenvalue** of a square matrix $A$ is a scalar $\lambda$ such that there exists a nonzero vector $v$ satisfying:

$$
Av = \lambda v
$$

In words: when you multiply the matrix $A$ by the vector $v$, the result is just $v$ scaled by $\lambda$. The direction of $v$ is unchanged (or reversed if $\lambda < 0$).

The vector $v$ is called an **eigenvector** corresponding to eigenvalue $\lambda$.

---

## Geometric Interpretation

Normally, multiplying a vector by a matrix rotates and stretches it. But eigenvectors are special: they only get stretched (scaled), not rotated.

For matrix $A$ and eigenvector $v$:
- $\lambda > 1$: $v$ is stretched (gets longer)
- $0 < \lambda < 1$: $v$ is compressed (gets shorter)
- $\lambda = 1$: $v$ is unchanged
- $\lambda < 0$: $v$ is flipped and scaled
- $\lambda = 0$: $v$ is collapsed to zero

---

## The Characteristic Equation

To find eigenvalues, we solve:

$$
Av = \lambda v
$$

Rearranging:

$$
Av - \lambda v = 0
$$

$$
(A - \lambda I)v = 0
$$

For a nonzero solution $v$ to exist, the matrix $(A - \lambda I)$ must be singular (non-invertible). This happens when:

$$
\det(A - \lambda I) = 0
$$

This is the **characteristic equation**. It is a polynomial of degree $n$ in $\lambda$, where $n$ is the size of the matrix. The roots of this polynomial are the eigenvalues.

---

## Computing Eigenvalues: 2x2 Example

For a 2x2 matrix:

$$
A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}
$$

The characteristic equation is:

$$
\det\begin{bmatrix} a - \lambda & b \\ c & d - \lambda \end{bmatrix} = 0
$$

$$
(a - \lambda)(d - \lambda) - bc = 0
$$

$$
\lambda^2 - (a + d)\lambda + (ad - bc) = 0
$$

$$
\lambda^2 - \text{tr}(A)\lambda + \det(A) = 0
$$

**Concrete example:**

$$
A = \begin{bmatrix} 4 & 2 \\ 1 & 3 \end{bmatrix}
$$

$\text{tr}(A) = 4 + 3 = 7$

$\det(A) = 4 \cdot 3 - 2 \cdot 1 = 10$

Characteristic equation: $\lambda^2 - 7\lambda + 10 = 0$

Factoring: $(\lambda - 5)(\lambda - 2) = 0$

Eigenvalues: $\lambda_1 = 5$, $\lambda_2 = 2$

---

## Properties of Eigenvalues

**Sum equals trace:**

$$
\sum_{i=1}^{n} \lambda_i = \text{tr}(A)
$$

**Product equals determinant:**

$$
\prod_{i=1}^{n} \lambda_i = \det(A)
$$

**Number of eigenvalues:**

An $n \times n$ matrix has exactly $n$ eigenvalues (counted with multiplicity), though some may be complex.

---

## Real vs. Complex Eigenvalues

Eigenvalues can be complex even for real matrices.

**Example:**

$$
A = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}
$$

This is a 90-degree rotation matrix. No vector stays in the same direction after rotation.

Characteristic equation: $\lambda^2 + 1 = 0$

Solutions: $\lambda = \pm i$ (imaginary)

Complex eigenvalues come in conjugate pairs for real matrices, and each pair has the same algebraic multiplicity.

---

## Eigenvalues of Special Matrices

**Identity matrix $I$:**

All eigenvalues are 1.

$Iv = v = 1 \cdot v$

**Diagonal matrix:**

Eigenvalues are the diagonal entries.

$$
D = \begin{bmatrix} 3 & 0 \\ 0 & 7 \end{bmatrix}
$$

Eigenvalues: 3 and 7.

**Triangular matrix (upper or lower):**

Eigenvalues are the diagonal entries.

**Singular matrix:**

At least one eigenvalue is 0 (since $\det(A) = 0$).

**Orthogonal matrix:**

All eigenvalues have absolute value 1 ($|\lambda| = 1$).

**Symmetric matrix (Spectral Theorem):**

Real symmetric matrices always have real eigenvalues and orthogonal eigenvectors. The matrix can be orthogonally diagonalized: $A = Q\Lambda Q^T$. This is the foundation of PCA.

**Positive definite matrix:**

All eigenvalues are strictly positive. Equivalently, $x^T A x > 0$ for all nonzero $x$.

---

## Eigenvalue Decomposition

If a matrix has $n$ linearly independent eigenvectors, it can be decomposed as:

$$
A = V \Lambda V^{-1}
$$

where:
- $V$ is a matrix whose columns are eigenvectors
- $\Lambda$ is a diagonal matrix with eigenvalues on the diagonal

For symmetric matrices:

$$
A = Q \Lambda Q^T
$$

where $Q$ is orthogonal ($Q^{-1} = Q^T$).

---

## Applications in Machine Learning

**Principal Component Analysis (PCA):**

Eigenvalues of the covariance matrix represent the variance captured by each principal component. Eigenvectors are the principal directions.

**Spectral clustering:**

Eigenvalues and eigenvectors of the graph Laplacian reveal cluster structure.

**PageRank:**

The PageRank vector is the eigenvector corresponding to eigenvalue 1 of the transition matrix.

**Stability analysis:**

In dynamical systems, eigenvalues determine stability. Systems are stable if all eigenvalues have absolute value less than 1.

**Matrix powers:**

$A^k = V \Lambda^k V^{-1}$

Eigenvalues make computing matrix powers efficient.

**Condition number:**

The ratio of largest to smallest eigenvalue (for symmetric positive definite matrices) indicates numerical sensitivity.

---

## Numerical Computation

Finding eigenvalues analytically requires solving a polynomial equation. For large matrices, this is impractical.

Numerical methods include:
- **Power iteration:** finds the largest eigenvalue
- **QR algorithm:** finds all eigenvalues iteratively
- **Lanczos algorithm:** for sparse matrices

Time complexity is typically $O(n^3)$ for dense matrices.

---

## Eigenvalues and Matrix Rank

The number of nonzero eigenvalues equals the rank of the matrix.

A rank-deficient matrix has at least one zero eigenvalue.

The **nullity** (dimension of null space) equals the multiplicity of the zero eigenvalue.

---

## The Spectral Theorem

For real symmetric matrices:
1. All eigenvalues are real
2. Eigenvectors corresponding to different eigenvalues are orthogonal
3. The matrix can be orthogonally diagonalized: $A = Q\Lambda Q^T$

This is fundamental to PCA, where we decompose a covariance matrix (symmetric) into orthogonal principal components.
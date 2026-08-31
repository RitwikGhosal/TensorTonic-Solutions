## What Is PCA?

Principal Component Analysis (PCA) is a technique for reducing the dimensionality of data while preserving as much variance as possible. It finds new axes (principal components) along which the data varies most.

If you have 100 features but only 10 directions capture most of the variation, PCA lets you project onto those 10 directions and discard the rest.

---

## The Core Idea

Imagine a cloud of data points in 3D space that is shaped like a flat pancake. Most of the spread is in two directions (the pancake's surface), with little variation in the third (the pancake's thickness).

PCA identifies:
1. The direction of maximum variance (first principal component)
2. The direction of second-most variance, perpendicular to the first (second PC)
3. And so on...

By projecting onto the top-k components, you keep the most important variation and discard the noise.

---

## The Algorithm

**Step 1: Center the data**

Subtract the mean of each feature so the data is centered at the origin:

$$
X_c = X - \bar{X}
$$

Where $\bar{X}$ is the row vector of column means.

**Step 2: Compute the covariance matrix**

$$
C = \frac{1}{n-1} X_c^T X_c
$$

This $d \times d$ matrix captures how features vary together.

**Step 3: Find eigenvalues and eigenvectors**

Solve $Cv = \lambda v$ to find:
- Eigenvalues $\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_d$
- Corresponding eigenvectors $v_1, v_2, \ldots, v_d$

The eigenvectors are the principal component directions.

**Step 4: Project the data**

Take the top-k eigenvectors as columns of $W$ (shape $d \times k$):

$$
X_{proj} = X_c W
$$

The result has shape $n \times k$: the data projected onto k dimensions.

---

## A Concrete Example

**Data:** 4 samples, 2 features

$$
X = \begin{bmatrix} 2 & 1 \\ 3 & 2 \\ 3 & 3 \\ 5 & 4 \end{bmatrix}
$$

**Step 1: Center**

Means: $\bar{x}_1 = 3.25$, $\bar{x}_2 = 2.5$

$$
X_c = \begin{bmatrix} -1.25 & -1.5 \\ -0.25 & -0.5 \\ -0.25 & 0.5 \\ 1.75 & 1.5 \end{bmatrix}
$$

**Step 2: Covariance matrix**

$$
C = \frac{1}{3} X_c^T X_c = \begin{bmatrix} 1.5625 & 1.4167 \\ 1.4167 & 1.5833 \end{bmatrix}
$$

**Step 3: Eigendecomposition**

Solving gives eigenvalues approximately 2.98 and 0.17.

The first eigenvector (direction of maximum variance) points roughly along the diagonal, which makes sense since both features increase together.

**Step 4: Project to 1D**

Using the first eigenvector, each data point becomes a single number representing its position along the principal direction.

---

## Variance Explained

Each eigenvalue represents the variance along that principal component:

$$
\text{Variance explained by PC}_i = \frac{\lambda_i}{\sum_{j} \lambda_j}
$$

If the top 3 eigenvalues are [10, 3, 1] out of a total of 15:
- PC1 explains 10/15 = 67% of variance
- PC2 explains 3/15 = 20% of variance
- PC3 explains 1/15 = 7% of variance

Together, the top 2 components explain 87% of the variance. You might keep just those 2.

---

## Choosing the Number of Components

**Common approaches:**

**Variance threshold:**
Keep enough components to explain (e.g.) 95% of total variance.

**Elbow method:**
Plot variance explained vs. number of components. Look for an "elbow" where adding more components gives diminishing returns.

**Kaiser criterion:**
Keep components with eigenvalue > 1 (for standardized data).

**Application-specific:**
If you need exactly 2D for visualization, use k=2.

---

## Properties of Principal Components

**Orthogonality:**
The principal components are mutually orthogonal (perpendicular). This means the projected features are uncorrelated.

**Ordered by importance:**
PC1 captures more variance than PC2, which captures more than PC3, etc.

**Linear combinations:**
Each PC is a linear combination of the original features. If the first eigenvector is $[0.7, 0.7]$, then PC1 = 0.7 * feature1 + 0.7 * feature2.

---

## PCA vs. Other Dimensionality Reduction

**PCA:**
- Linear method
- Fast and scalable
- Global structure preserved
- May miss non-linear patterns

**t-SNE, UMAP:**
- Non-linear methods
- Better for visualization
- Preserves local structure
- More computationally expensive

**Autoencoders:**
- Neural network approach
- Can learn non-linear mappings
- Requires training data
- More flexible but complex

---

## When PCA Helps

**Reducing computation:**
Train models faster with fewer features.

**Removing noise:**
Low-variance components often capture noise. Discarding them can improve model performance.

**Visualization:**
Project high-dimensional data to 2D or 3D for plotting.

**Decorrelating features:**
Some algorithms work better with uncorrelated inputs.

---

## When PCA Struggles

**Non-linear relationships:**
PCA finds linear directions. If the data lies on a curved manifold, PCA may miss the structure.

**Different scales:**
Features with larger scales dominate the variance. Standardize features first if scales differ.

**Interpretability:**
Principal components are mixtures of original features. "PC1 is 0.3 * age + 0.2 * income + ..." is hard to interpret.

---

## Implementation Notes

**SVD alternative:**

Instead of eigendecomposition of $X^T X$, you can use SVD of $X$ directly:

$$
X = U \Sigma V^T
$$

The columns of $V$ are the principal component directions. This is often more numerically stable.

**Centering is essential:**

If you skip centering, the first "principal component" will point toward the data centroid, not the direction of maximum variance. Always center before PCA.
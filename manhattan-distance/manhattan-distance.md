## What Is Manhattan Distance?

Manhattan distance measures the distance between two points by summing the absolute differences along each dimension. It is called "Manhattan" distance because it measures distance as if you were walking on a grid of city blocks, where you can only move horizontally or vertically, never diagonally.

Also known as:
- L1 distance
- Taxicab distance
- City block distance
- Rectilinear distance

---

## The Formula

For two points $A = (a_1, a_2, ..., a_n)$ and $B = (b_1, b_2, ..., b_n)$:

$$
d_{\text{Manhattan}}(A, B) = \sum_{i=1}^{n} |a_i - b_i|
$$

Sum the absolute differences in each coordinate.

---

## Step-by-Step Computation

**Example 1: 2D points**

$A = (2, 3)$ and $B = (5, 7)$

$d = |2 - 5| + |3 - 7| = |-3| + |-4| = 3 + 4 = 7$

You would walk 3 blocks east and 4 blocks north.

**Example 2: 3D points**

$A = (1, 2, 3)$ and $B = (4, 0, 6)$

$d = |1 - 4| + |2 - 0| + |3 - 6| = 3 + 2 + 3 = 8$

**Example 3: High-dimensional vectors**

$A = [1, 0, 3, 2]$ and $B = [0, 2, 1, 5]$

$d = |1-0| + |0-2| + |3-1| + |2-5| = 1 + 2 + 2 + 3 = 8$

---

## Geometric Interpretation

In 2D, Manhattan distance traces a path along the grid:

From $(0, 0)$ to $(3, 4)$:
- Euclidean distance: $\sqrt{3^2 + 4^2} = 5$ (straight line)
- Manhattan distance: $3 + 4 = 7$ (grid path)

There are many paths with the same Manhattan distance. Any path that goes 3 units right and 4 units up (in any order) has distance 7.

The Manhattan "circle" of radius $r$ centered at the origin is a diamond shape with vertices at $(r, 0)$, $(0, r)$, $(-r, 0)$, $(0, -r)$.

---

## Manhattan vs. Euclidean Distance

**Euclidean distance (L2):**

$$
d_{\text{Euclidean}} = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2}
$$

**Manhattan distance (L1):**

$$
d_{\text{Manhattan}} = \sum_{i=1}^{n} |a_i - b_i|
$$

**Key differences:**

Euclidean squares the differences, then takes the square root. Large differences are penalized more heavily.

Manhattan just sums absolute differences. All differences are weighted equally.

**Comparison for $A = (0, 0)$ and $B = (3, 4)$:**

- Euclidean: $\sqrt{9 + 16} = 5$
- Manhattan: $3 + 4 = 7$

**Comparison for $A = (0, 0)$ and $B = (1, 1)$:**

- Euclidean: $\sqrt{1 + 1} = 1.414$
- Manhattan: $1 + 1 = 2$

Manhattan distance is always greater than or equal to Euclidean distance.

---

## Properties of Manhattan Distance

**Non-negativity:**

$d(A, B) \geq 0$, and $d(A, B) = 0$ if and only if $A = B$.

**Symmetry:**

$d(A, B) = d(B, A)$

**Triangle inequality:**

$d(A, C) \leq d(A, B) + d(B, C)$

These three properties make Manhattan distance a valid metric.

**Relationship to Euclidean:**

$$
d_{\text{Euclidean}} \leq d_{\text{Manhattan}} \leq \sqrt{n} \cdot d_{\text{Euclidean}}
$$

where $n$ is the number of dimensions.

---

## When to Use Manhattan Distance

**Grid-based problems:**

- Pathfinding on a grid (games, robotics)
- Warehouse optimization (aisles are grid-like)
- Circuit design (wires run horizontally/vertically)

**Sparse, high-dimensional data:**

- Manhattan distance is more robust to outliers than Euclidean
- In high dimensions, Euclidean distances become less meaningful (curse of dimensionality)
- L1 promotes sparsity (related to L1 regularization)

**Integer or categorical differences:**

- When each dimension is an independent attribute
- Hamming distance (for binary vectors) is a special case

**Robust regression:**

- L1 loss (MAE) uses Manhattan distance from predictions to targets
- Less sensitive to outliers than L2 loss (MSE)

---

## Manhattan Distance in Machine Learning

**K-Nearest Neighbors (KNN):**

KNN can use any distance metric. Manhattan distance is common for:
- High-dimensional data
- Data with outliers
- Features on different scales (after normalization)

**Clustering:**

K-medians clustering minimizes Manhattan distance (vs. K-means which minimizes Euclidean).

**Recommendation systems:**

User similarity can be measured with Manhattan distance on rating vectors.

**Feature selection:**

L1 regularization (LASSO) adds a penalty proportional to Manhattan distance from zero.

---

## Computational Considerations

Manhattan distance requires:
- $n$ subtractions
- $n$ absolute values
- $n - 1$ additions
- Time complexity: $O(n)$

No square root needed, unlike Euclidean distance. This makes Manhattan distance slightly faster to compute.

**Vectorized computation:**

For arrays $A$ and $B$:

$d = \text{sum}(\text{abs}(A - B))$

This is highly efficient with optimized array libraries.

---

## Handling Numerical Issues

Manhattan distance has no numerical stability issues:
- No division
- No square root
- Just subtraction and absolute value

Even for very large or very small numbers, the computation is straightforward.

---

## Extension: Weighted Manhattan Distance

You can weight each dimension differently:

$$
d_{\text{weighted}} = \sum_{i=1}^{n} w_i |a_i - b_i|
$$

This is useful when some features are more important than others.
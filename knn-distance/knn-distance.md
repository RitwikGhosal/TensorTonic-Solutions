## The Core Idea of KNN

K-Nearest Neighbors (KNN) is a simple yet powerful algorithm based on a single assumption: **similar inputs have similar outputs**.

To classify a new point:
1. Find the $k$ closest points in the training data
2. Let them vote on the class
3. Return the majority class

The key step is computing **distances** between the query point and all training points.

---

## Distance Metrics

The choice of distance metric determines what "similar" means. Common metrics:

**Euclidean distance (L2):**

$$
d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
$$

**Manhattan distance (L1):**

$$
d(x, y) = \sum_{i=1}^{n} |x_i - y_i|
$$

**Minkowski distance (Lp):**

$$
d(x, y) = \left(\sum_{i=1}^{n} |x_i - y_i|^p\right)^{1/p}
$$

Euclidean is $p=2$, Manhattan is $p=1$.

---

## Euclidean Distance in Detail

The most common distance metric. Measures straight-line distance in n-dimensional space.

**For 2D points** $x = (x_1, x_2)$ and $y = (y_1, y_2)$:

$$
d = \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2}
$$

**Example:**

$x = (1, 2)$, $y = (4, 6)$

$d = \sqrt{(1-4)^2 + (2-6)^2} = \sqrt{9 + 16} = \sqrt{25} = 5$

---

## Computing Distances: One Query vs. All Training Points

**Query point:** $q$ of shape $(d,)$ where $d$ is the number of features

**Training points:** $X$ of shape $(n, d)$ where $n$ is the number of samples

**Goal:** Compute $n$ distances, one from $q$ to each row of $X$

**Naive approach:**

For each training point $x_i$:

$$
d_i = \sqrt{\sum_{j=1}^{d} (q_j - x_{i,j})^2}
$$

---

## Efficient Computation Using Broadcasting

Instead of looping, use vectorized operations:

**Step 1:** Compute differences

$\text{diff} = X - q$, shape $(n, d)$

Each row is $x_i - q$.

**Step 2:** Square the differences

$\text{diff}^2$, element-wise squaring

**Step 3:** Sum across features

$\text{sum}(\text{diff}^2, \text{axis}=1)$, shape $(n,)$

**Step 4:** Square root

$\sqrt{\text{sums}}$, shape $(n,)$

---

## Worked Example

**Training data (3 samples, 2 features):**

$$
X = \begin{bmatrix}
1 & 2 \\
4 & 5 \\
7 & 8
\end{bmatrix}
$$

**Query point:**

$q = [2, 3]$

**Step 1: Differences**

$$
X - q = \begin{bmatrix}
1-2 & 2-3 \\
4-2 & 5-3 \\
7-2 & 8-3
\end{bmatrix} = \begin{bmatrix}
-1 & -1 \\
2 & 2 \\
5 & 5
\end{bmatrix}
$$

**Step 2: Square**

$$
\begin{bmatrix}
1 & 1 \\
4 & 4 \\
25 & 25
\end{bmatrix}
$$

**Step 3: Sum across features**

$[1+1, 4+4, 25+25] = [2, 8, 50]$

**Step 4: Square root**

$[\sqrt{2}, \sqrt{8}, \sqrt{50}] = [1.41, 2.83, 7.07]$

**Result:** Distances are $[1.41, 2.83, 7.07]$

The nearest neighbor is sample 0 (distance 1.41).

---

## Finding the K Nearest Neighbors

After computing all distances:

1. Sort the distances (or use partial sort for efficiency)
2. Take the indices of the $k$ smallest distances
3. These are the $k$ nearest neighbors

**Example with $k = 2$:**

Distances: $[1.41, 2.83, 7.07]$

Sorted indices: $[0, 1, 2]$ (already sorted)

$k = 2$ nearest: indices $[0, 1]$, distances $[1.41, 2.83]$

---

## Distance Matrix for Multiple Queries

When you have multiple query points, compute a **distance matrix**:

**Queries:** $Q$ of shape $(m, d)$

**Training:** $X$ of shape $(n, d)$

**Distance matrix:** $D$ of shape $(m, n)$ where $D_{ij}$ is the distance from query $i$ to training point $j$

---

## Efficient Distance Matrix Computation

Using the identity:

$$
||q - x||^2 = ||q||^2 + ||x||^2 - 2 q \cdot x
$$

**Step 1:** Compute squared norms

$||Q||^2$: shape $(m, 1)$

$||X||^2$: shape $(1, n)$

**Step 2:** Compute dot products

$Q \cdot X^T$: shape $(m, n)$

**Step 3:** Combine

$D^2 = ||Q||^2 + ||X||^2 - 2 (Q \cdot X^T)$

**Step 4:** Square root

$D = \sqrt{D^2}$

This avoids explicit loops and leverages optimized matrix multiplication.

---

## Choosing the Distance Metric

**Euclidean (L2):**
- Default choice
- Sensitive to scale (normalize features first)
- Sensitive to outliers

**Manhattan (L1):**
- More robust to outliers
- Better for high-dimensional sparse data
- Treats all dimensions equally

**Cosine distance:**
- $d = 1 - \text{cosine\_similarity}$
- Good for text (TF-IDF vectors)
- Ignores magnitude, focuses on direction

**Minkowski (general Lp):**
- $p = 2$: Euclidean
- $p = 1$: Manhattan
- $p \to \infty$: Chebyshev (max absolute difference)

---

## Feature Normalization

**Problem:** Features on different scales dominate the distance.

Example: Age (0-100) vs. Income (0-1,000,000)

A difference of 10,000 in income dominates any difference in age.

**Solution:** Normalize features before computing distances:

**Z-score normalization:**

$$
x' = \frac{x - \mu}{\sigma}
$$

**Min-max normalization:**

$$
x' = \frac{x - \min}{\max - \min}
$$

---

## Handling Ties

When multiple training points have the same distance:

**Options:**
- Include all tied points (may exceed k)
- Random selection among tied points
- Use secondary criterion (e.g., earlier index)

For classification, ties in voting are also possible. Common resolution: random choice, or favor smaller class label.

---

## Weighted KNN

Instead of equal votes, weight neighbors by inverse distance:

$$
w_i = \frac{1}{d_i}
$$

Closer neighbors have more influence. Helps when neighbors vary significantly in distance.

**Variant:** $w_i = \frac{1}{d_i^2}$ for even stronger locality.

---

## Computational Complexity

**Naive distance computation:**
- Per query: $O(n \cdot d)$ where $n$ = training samples, $d$ = features
- For $m$ queries: $O(m \cdot n \cdot d)$

**Finding k nearest:**
- Full sort: $O(n \log n)$ per query
- Partial sort (heap): $O(n \log k)$ per query

**Space:** $O(n \cdot d)$ to store training data, $O(n)$ or $O(m \cdot n)$ for distances

---

## Speeding Up KNN

**KD-trees:** Partition space for faster nearest neighbor queries. Works well in low dimensions.

**Ball trees:** Better for high dimensions than KD-trees.

**Approximate methods:** Locality-Sensitive Hashing (LSH) for approximate nearest neighbors.

**Reducing training data:** Prototype selection, condensing.

For small to medium datasets, brute-force distance computation is often fast enough.
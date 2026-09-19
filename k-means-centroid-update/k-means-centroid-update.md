## What Is the Centroid Update Step?

The centroid update step is the second half of the K-Means iteration. After assigning each point to a cluster, we recompute each centroid as the **mean** of all points assigned to that cluster.

$$
\mu_j = \frac{1}{|C_j|} \sum_{i \in C_j} x_i
$$

where $C_j$ is the set of points assigned to cluster $j$ and $|C_j|$ is the number of points in cluster $j$.

---

## Why the Mean?

The mean minimizes the sum of squared distances to all points:

$$
\mu_j^* = \arg\min_\mu \sum_{i \in C_j} ||x_i - \mu||^2
$$

Taking the derivative with respect to $\mu$ and setting to zero:

$$
\frac{\partial}{\partial \mu} \sum_{i \in C_j} ||x_i - \mu||^2 = -2 \sum_{i \in C_j} (x_i - \mu) = 0
$$

Solving: $\sum_{i \in C_j} x_i = |C_j| \cdot \mu$

$$
\mu = \frac{1}{|C_j|} \sum_{i \in C_j} x_i
$$

The mean is the optimal centroid for minimizing squared distances.

---

## Step-by-Step Example

**Data points (2D):**
- $x_1 = (1, 2)$
- $x_2 = (2, 1)$
- $x_3 = (4, 5)$
- $x_4 = (5, 4)$
- $x_5 = (3, 3)$

**Current cluster assignments:**
- Cluster 1: $\{x_1, x_2, x_5\}$
- Cluster 2: $\{x_3, x_4\}$

---

**Update centroid 1:**

Points in cluster 1: $(1, 2)$, $(2, 1)$, $(3, 3)$

$$
\mu_1 = \frac{1}{3} [(1, 2) + (2, 1) + (3, 3)]
$$

$$
= \frac{1}{3} (1+2+3, 2+1+3) = \frac{1}{3} (6, 6) = (2, 2)
$$

---

**Update centroid 2:**

Points in cluster 2: $(4, 5)$, $(5, 4)$

$$
\mu_2 = \frac{1}{2} [(4, 5) + (5, 4)]
$$

$$
= \frac{1}{2} (4+5, 5+4) = \frac{1}{2} (9, 9) = (4.5, 4.5)
$$

---

**Updated centroids:**
- $\mu_1 = (2, 2)$
- $\mu_2 = (4.5, 4.5)$

---

## Multi-Dimensional Computation

For $d$-dimensional data, compute the mean for each dimension separately:

$$
\mu_j = (\mu_{j1}, \mu_{j2}, ..., \mu_{jd})
$$

where:

$$
\mu_{jk} = \frac{1}{|C_j|} \sum_{i \in C_j} x_{ik}
$$

Each coordinate of the centroid is the average of that coordinate across all cluster points.

---

## Vectorized Computation

For efficiency, use matrix operations:

**Input:**
- $X$: data matrix of shape $(n, d)$
- $c$: cluster assignments, array of length $n$ with values in $\{1, ..., k\}$

**For each cluster $j$:**

1. Create mask: $m_j = (c == j)$, boolean array of length $n$
2. Select points: $X_j = X[m_j]$, shape $(|C_j|, d)$
3. Compute mean: $\mu_j = \text{mean}(X_j, \text{axis}=0)$, shape $(d,)$

Alternatively, use scatter-add operations for parallel computation.

---

## Empty Clusters

If a cluster has no assigned points, the mean is undefined (division by zero).

**Problem:** $|C_j| = 0 \Rightarrow \mu_j = ?$

**Solutions:**

**Keep old centroid:** Do not update the centroid if the cluster is empty.

**Reinitialize randomly:** Place the centroid at a random data point.

**Farthest point:** Place the centroid at the point farthest from all other centroids (encourages coverage).

**Split large cluster:** Take the largest cluster and divide it, placing one centroid at each half's mean.

---

## The K-Means Objective

The centroid update step decreases (or maintains) the within-cluster sum of squares:

$$
J = \sum_{j=1}^{k} \sum_{i \in C_j} ||x_i - \mu_j||^2
$$

**Proof:**

Given fixed assignments $C_1, ..., C_k$, the objective $J$ is minimized when each $\mu_j$ is the mean of $C_j$.

Since we set $\mu_j$ to exactly this mean, we achieve the minimum for the current assignments.

Thus, $J$ cannot increase after the centroid update.

---

## Convergence Guarantee

Each K-Means iteration (assignment + update) either:
- Decreases $J$, or
- Keeps $J$ the same (converged)

Since:
1. $J$ is bounded below (by 0)
2. There are finite possible assignments
3. $J$ decreases at each step

The algorithm must converge in finite iterations.

---

## Online Centroid Update

For streaming data or mini-batch K-Means, update centroids incrementally:

**Running mean formula:**

When adding a new point $x_{\text{new}}$ to cluster $j$:

$$
\mu_j^{\text{new}} = \mu_j^{\text{old}} + \frac{1}{n_j + 1} (x_{\text{new}} - \mu_j^{\text{old}})
$$

where $n_j$ is the current count of points in cluster $j$.

This avoids recomputing the sum over all points.

---

## Mini-Batch Update

For mini-batch K-Means:

1. For each point in the mini-batch, find nearest centroid
2. Update centroid with learning rate $\eta$:

$$
\mu_j \leftarrow (1 - \eta) \mu_j + \eta \cdot \frac{1}{|B_j|} \sum_{i \in B_j} x_i
$$

where $B_j$ is the set of mini-batch points assigned to cluster $j$.

The learning rate $\eta$ decays over time for convergence.

---

## K-Medians vs K-Means

**K-Means:** Centroid is the mean, minimizes squared distances.

**K-Medians:** Centroid is the median (coordinate-wise), minimizes absolute distances (L1).

$$
\mu_{jk}^{\text{median}} = \text{median}(\{x_{ik} : i \in C_j\})
$$

K-Medians is more robust to outliers.

---

## K-Medoids

Instead of computing the mean, the centroid must be an actual data point:

$$
\mu_j = \arg\min_{x \in C_j} \sum_{i \in C_j} ||x_i - x||^2
$$

The medoid is the point that minimizes total distance to all other cluster points.

**Advantages:**
- Works with any distance metric
- Robust to outliers
- Centroid is always a real data point

**Disadvantage:**
- More expensive to compute: $O(|C_j|^2)$ per cluster

---

## Computational Complexity

**For each cluster:**
- Sum $|C_j|$ vectors of dimension $d$: $O(|C_j| \cdot d)$
- Divide by count: $O(d)$

**Total for all clusters:**
- $\sum_j O(|C_j| \cdot d) = O(n \cdot d)$

Very efficient, linear in the number of points.

---

## Numerical Precision

For large datasets, summing many numbers can cause precision loss.

**Solutions:**

**Kahan summation:** Use compensated summation algorithm.

**Two-pass:** First compute the mean approximately, then compute deviations.

**Higher precision:** Use 64-bit floats even if data is 32-bit.

For most applications, standard floating-point arithmetic is sufficient.

---

## Weighted K-Means

If points have weights $w_i$, the weighted centroid is:

$$
\mu_j = \frac{\sum_{i \in C_j} w_i x_i}{\sum_{i \in C_j} w_i}
$$

This generalizes the unweighted case where all $w_i = 1$.

Useful when some data points are more important or represent multiple observations.
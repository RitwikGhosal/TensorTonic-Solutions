## What Is K-Means Clustering?

K-Means is an unsupervised learning algorithm that partitions $n$ data points into $k$ clusters. Each point belongs to the cluster with the nearest centroid (cluster center).

The algorithm iterates between two steps:
1. **Assignment:** Assign each point to the nearest centroid
2. **Update:** Recompute centroids as the mean of assigned points

This repeats until convergence.

---

## The Assignment Step

Given $k$ centroids $\mu_1, \mu_2, ..., \mu_k$, the assignment step assigns each data point $x_i$ to the nearest centroid:

$$
c_i = \arg\min_j ||x_i - \mu_j||^2
$$

where $c_i$ is the cluster assignment for point $i$, and $||\cdot||^2$ is the squared Euclidean distance.

Each point is assigned to exactly one cluster.

---

## Computing Distances

For a point $x = (x_1, x_2, ..., x_d)$ and centroid $\mu = (\mu_1, \mu_2, ..., \mu_d)$:

**Squared Euclidean distance:**

$$
||x - \mu||^2 = \sum_{j=1}^{d} (x_j - \mu_j)^2
$$

We use squared distance because:
- Avoids computing square root (faster)
- Minimizing squared distance is equivalent to minimizing distance
- Mathematically convenient for the objective function

---

## Step-by-Step Assignment Example

**Data points (2D):**
- $x_1 = (1, 2)$
- $x_2 = (2, 1)$
- $x_3 = (4, 5)$
- $x_4 = (5, 4)$

**Current centroids:**
- $\mu_1 = (1, 1)$
- $\mu_2 = (5, 5)$

---

**Assign $x_1 = (1, 2)$:**

Distance to $\mu_1$: $(1-1)^2 + (2-1)^2 = 0 + 1 = 1$

Distance to $\mu_2$: $(1-5)^2 + (2-5)^2 = 16 + 9 = 25$

$x_1$ assigned to cluster 1 (distance 1 < 25)

---

**Assign $x_2 = (2, 1)$:**

Distance to $\mu_1$: $(2-1)^2 + (1-1)^2 = 1 + 0 = 1$

Distance to $\mu_2$: $(2-5)^2 + (1-5)^2 = 9 + 16 = 25$

$x_2$ assigned to cluster 1 (distance 1 < 25)

---

**Assign $x_3 = (4, 5)$:**

Distance to $\mu_1$: $(4-1)^2 + (5-1)^2 = 9 + 16 = 25$

Distance to $\mu_2$: $(4-5)^2 + (5-5)^2 = 1 + 0 = 1$

$x_3$ assigned to cluster 2 (distance 1 < 25)

---

**Assign $x_4 = (5, 4)$:**

Distance to $\mu_1$: $(5-1)^2 + (4-1)^2 = 16 + 9 = 25$

Distance to $\mu_2$: $(5-5)^2 + (4-5)^2 = 0 + 1 = 1$

$x_4$ assigned to cluster 2 (distance 1 < 25)

---

**Final assignments:**
- Cluster 1: $\{x_1, x_2\}$
- Cluster 2: $\{x_3, x_4\}$

---

## The K-Means Objective

K-Means minimizes the **within-cluster sum of squares (WCSS)**:

$$
J = \sum_{i=1}^{n} ||x_i - \mu_{c_i}||^2
$$

where $c_i$ is the cluster assignment for point $i$ and $\mu_{c_i}$ is the centroid of that cluster.

The assignment step minimizes $J$ with respect to cluster assignments (holding centroids fixed).

---

## Handling Ties

When a point is equidistant from multiple centroids:

**Option 1:** Assign to the first centroid in the list

**Option 2:** Assign randomly among tied centroids

**Option 3:** Assign to the smaller cluster (for balance)

In practice, exact ties are rare with continuous data.

---

## Vectorized Assignment

For efficiency, compute all distances at once:

**Distance matrix:** $D$ of shape $(n, k)$ where $D_{ij} = ||x_i - \mu_j||^2$

Using the identity:

$$
||x - \mu||^2 = ||x||^2 + ||\mu||^2 - 2x^T\mu
$$

**Compute:**
1. $||X||^2$: squared norms of all points, shape $(n, 1)$
2. $||M||^2$: squared norms of all centroids, shape $(1, k)$
3. $XM^T$: dot products, shape $(n, k)$
4. $D = ||X||^2 + ||M||^2 - 2XM^T$

**Assignments:** $c = \arg\min_j D_{:,j}$ (column-wise argmin)

---

## Distance Metrics

While K-Means typically uses Euclidean distance, variants exist:

**Euclidean (L2):**

$$
d(x, \mu) = \sqrt{\sum_j (x_j - \mu_j)^2}
$$
Standard K-Means

**Squared Euclidean:**

$$
d(x, \mu) = \sum_j (x_j - \mu_j)^2
$$
Equivalent for assignment (no square root)

**Manhattan (L1):**

$$
d(x, \mu) = \sum_j |x_j - \mu_j|
$$
K-Medians algorithm (centroid becomes median)

**Cosine distance:**

$$
d(x, \mu) = 1 - \frac{x^T\mu}{||x|| \cdot ||\mu||}
$$
Spherical K-Means (normalize vectors to unit sphere)

---

## Empty Cluster Problem

Sometimes, after assignment, a cluster may have no points assigned. This causes problems when computing the centroid.

**Solutions:**

**Reinitialize:** Replace the empty centroid with a random point or the point farthest from all centroids.

**Split a cluster:** Take the largest cluster and split it in two.

**Remove the cluster:** Reduce $k$ by one (may not be desirable).

**Prevention:** Use K-Means++ initialization to get well-spread centroids.

---

## Assignment Complexity

**Per point:**
- Compute distance to each centroid: $O(k \cdot d)$
- Find minimum: $O(k)$
- Total per point: $O(k \cdot d)$

**All points:**
- $O(n \cdot k \cdot d)$

With matrix operations, this is very efficient on modern hardware.

---

## Hard vs Soft Assignment

**Hard assignment (standard K-Means):**

Each point belongs to exactly one cluster.

$$
c_i = \arg\min_j ||x_i - \mu_j||^2
$$

**Soft assignment (fuzzy K-Means / EM):**

Each point has a probability of belonging to each cluster.

$$
P(c_i = j) \propto \exp\left(-\frac{||x_i - \mu_j||^2}{2\sigma^2}\right)
$$

Soft assignment is used in Gaussian Mixture Models (GMM).

---

## Convergence

The assignment step (along with the update step) guarantees that:

1. Each step decreases or maintains the objective $J$
2. There are finite possible assignments
3. Algorithm must converge in finite iterations

However, K-Means only finds a **local minimum**, not necessarily the global minimum.

---

## Mini-Batch K-Means

For large datasets, compute assignments on a random subset (mini-batch) at each iteration:

1. Sample a mini-batch of $b$ points
2. Assign mini-batch points to nearest centroids
3. Update centroids using mini-batch assignments

This approximates the full K-Means but is much faster for large $n$.

---

## Practical Considerations

**Feature scaling:** Normalize features to have similar scales, otherwise features with large values dominate the distance.

**Initialization:** Use K-Means++ for better initial centroids.

**Multiple runs:** Run K-Means several times with different initializations and keep the best result (lowest $J$).

**Choosing $k$:** Use elbow method, silhouette score, or domain knowledge.
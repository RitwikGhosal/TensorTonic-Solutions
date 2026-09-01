## From Pythagoras to N Dimensions

The Euclidean distance is the straight-line distance between two points. It comes directly from the Pythagorean theorem. In 2D, if you have two points $(x_1, y_1)$ and $(x_2, y_2)$, the horizontal difference is $\Delta x = x_2 - x_1$ and the vertical difference is $\Delta y = y_2 - y_1$. These two differences form the legs of a right triangle, and the distance is the hypotenuse:

$$
d = \sqrt{(\Delta x)^2 + (\Delta y)^2}
$$

The classic example is the point $(3, 4)$ and the origin $(0, 0)$: $d = \sqrt{9 + 16} = \sqrt{25} = 5$. This is the 3-4-5 right triangle that the ancient Greeks knew well.

The beautiful thing about this formula is that it generalizes effortlessly. In 3D, you simply add a third squared difference: $d = \sqrt{(\Delta x)^2 + (\Delta y)^2 + (\Delta z)^2}$. In $n$ dimensions, with vectors $x = (x_1, x_2, \ldots, x_n)$ and $y = (y_1, y_2, \ldots, y_n)$:

$$
d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
$$

Each dimension contributes a squared difference. The sum of all squared differences gives the squared distance, and the square root brings it back to the same units as the original coordinates. This is the Euclidean distance, also called the L2 distance.

---

## What Makes It a "True" Distance

Not every function that measures similarity qualifies as a distance in the mathematical sense. A proper **metric** must satisfy four properties, and the Euclidean distance satisfies all of them:

**Non-negativity**: $d(x, y) \ge 0$ for all $x, y$. Distance is never negative. This follows because we are summing squares (which are non-negative) and then taking a square root.

**Identity of indiscernibles**: $d(x, y) = 0$ if and only if $x = y$. Two points are at distance zero only when they are the same point. A sum of squares is zero only if every term is zero, which means $x_i = y_i$ for all $i$.

**Symmetry**: $d(x, y) = d(y, x)$. The distance from A to B is the same as from B to A. This is obvious from the formula since $(x_i - y_i)^2 = (y_i - x_i)^2$.

**Triangle inequality**: $d(x, z) \le d(x, y) + d(y, z)$. Going directly from $x$ to $z$ is never longer than going from $x$ to $y$ and then from $y$ to $z$. The straight line is always the shortest path.

These four properties are what make the Euclidean distance a valid metric. Algorithms that rely on metric properties (like KD-trees for nearest neighbor search, or the triangle inequality for pruning distance computations) depend on all four holding.

---

## The L2 Norm and the Family of Lp Distances

The Euclidean distance is a special case of a broader family called **Lp distances** (or Minkowski distances). The general formula is:

$$
d_p(x, y) = \left( \sum_{i=1}^{n} |x_i - y_i|^p \right)^{1/p}
$$

Different values of $p$ give different distance metrics:

The **L1 distance** ($p = 1$) is the Manhattan distance, also called the taxicab or city-block distance. It sums the absolute differences: $d_1 = \sum |x_i - y_i|$. Imagine navigating a grid of city blocks where you can only move along the axes. In 2D, the L1 distance from $(0,0)$ to $(3,4)$ is $3 + 4 = 7$, not 5.

The **L2 distance** ($p = 2$) is the Euclidean distance. It is the most natural and widely used because it corresponds to the physical straight-line distance and arises naturally from the Pythagorean theorem and the dot product.

The **L-infinity distance** ($p \to \infty$) is the Chebyshev distance. It equals the maximum absolute difference across all dimensions: $d_\infty = \max_i |x_i - y_i|$. In the chess world, this is the number of king moves needed to go between two squares (since a king can move diagonally).

Each of these defines a different notion of "closeness." The L1 unit ball (all points at distance 1 from the origin) is a diamond shape. The L2 unit ball is a circle (or hypersphere). The L-infinity unit ball is a square (or hypercube). These shapes reflect the geometric character of each distance.

The Euclidean distance can also be expressed using the **L2 norm** of the difference vector. The L2 norm of a vector $v$ is its length:

$$
\|v\|_2 = \sqrt{\sum_{i=1}^{n} v_i^2}
$$

So the Euclidean distance between $x$ and $y$ is simply $d(x, y) = \|x - y\|_2$. Compute the difference vector, then take its length.

---

## Connection to the Dot Product

There is an important algebraic relationship between the Euclidean distance, the dot product, and the L2 norm. The squared Euclidean distance can be expanded as:

$$
d(x, y)^2 = \|x - y\|_2^2 = \|x\|_2^2 + \|y\|_2^2 - 2 \, x \cdot y
$$

where $x \cdot y = \sum_i x_i y_i$ is the dot product. This means that if you already know the norms of $x$ and $y$ and their dot product, you can compute the distance without going back to element-wise differences. This identity is used extensively in optimized distance computations, for example when computing all pairwise distances in a dataset using matrix operations.

This also connects Euclidean distance to **cosine similarity**. Cosine similarity measures the angle between two vectors: $\cos \theta = \frac{x \cdot y}{\|x\| \|y\|}$. If the vectors are normalized to unit length ($\|x\| = \|y\| = 1$), then $d^2 = 2 - 2 \cos \theta$. So for unit vectors, small Euclidean distance corresponds to high cosine similarity and vice versa. This is why Euclidean distance and cosine similarity give equivalent nearest-neighbor rankings when applied to normalized vectors.

---

## When Euclidean Distance Breaks Down

Despite its ubiquity, the Euclidean distance has important limitations that you should understand before applying it blindly.

**Scale sensitivity**: The Euclidean distance treats all dimensions equally. If one feature is measured in meters (range 0 to 1) and another in millimeters (range 0 to 1000), the millimeter feature will dominate the distance calculation simply because its numerical values are larger. Two points that differ by 1 meter and 0 millimeters will appear closer than two points that differ by 0 meters and 100 millimeters, even though the first difference is physically much larger. The fix is to **standardize** or **normalize** the features before computing distances, so that all features contribute proportionally.

**Correlated features**: The Euclidean distance does not account for correlations between features. If height and arm span are both included as features, they carry largely redundant information, but the Euclidean distance double-counts their contribution. The **Mahalanobis distance** addresses this by incorporating the covariance matrix, effectively "whitening" the data before computing the distance.

**The curse of dimensionality**: In very high-dimensional spaces (hundreds or thousands of features), Euclidean distances become increasingly uninformative. As the number of dimensions grows, the distances between all pairs of points converge to roughly the same value. The ratio between the nearest and farthest neighbor approaches 1, which means the concept of "nearest neighbor" loses meaning. This phenomenon is a core challenge in high-dimensional data analysis and is one reason why dimensionality reduction (PCA, t-SNE, UMAP) is often applied before distance-based algorithms.

**Non-vector data**: The Euclidean distance only applies to numerical vectors of equal length. For strings, graphs, probability distributions, or other structured data, you need specialized distances (edit distance, graph kernels, KL divergence, etc.).

---

## Applications in Machine Learning

The Euclidean distance is a building block for many core ML algorithms.

**K-Nearest Neighbors (KNN)** classifies a new point by finding the $k$ training points closest to it (by Euclidean distance) and taking a majority vote on their labels. The entire algorithm is essentially a distance computation.

**K-Means clustering** assigns each point to the cluster whose centroid is closest (by Euclidean distance) and then recomputes centroids as the mean of all assigned points. The algorithm alternates between these two steps until convergence. The use of Euclidean distance is what makes the clusters spherical (roughly equal extent in all directions).

**PCA** finds the directions of maximum variance, and the reconstruction error in PCA is measured as the Euclidean distance between the original point and its projection onto the lower-dimensional subspace.

**Gaussian distributions** have probability contours shaped by the Mahalanobis distance, which reduces to the Euclidean distance when the covariance matrix is the identity. Fitting a Gaussian is essentially characterizing a cluster by its center and the typical Euclidean spread around it.

**Loss functions**: The mean squared error (MSE) loss widely used in regression is the squared Euclidean distance between the predicted and actual output vectors, averaged over the dataset. Minimizing MSE is equivalent to minimizing the average squared Euclidean distance.

**Vector search**: Embedding-based retrieval systems (semantic search, recommendation engines, RAG) store items as vectors and find the closest ones to a query vector. Euclidean distance (or equivalently, squared Euclidean distance for ranking purposes) is one of the standard similarity metrics alongside cosine similarity and dot product.
## What Cosine Similarity Measures

Cosine similarity measures the **angle** between two vectors, ignoring their magnitudes. Two vectors pointing in the same direction have cosine similarity of 1, regardless of whether one is twice as long as the other.

This makes it ideal for comparing documents, embeddings, or any data where the direction (relative proportions) matters more than the scale.

---

## The Formula

For two vectors $A$ and $B$:

$$
\cos(\theta) = \frac{A \cdot B}{||A|| \times ||B||}
$$

where:
- $A \cdot B$ is the dot product: $\sum_{i=1}^{n} A_i \times B_i$
- $||A||$ is the L2 norm (magnitude): $\sqrt{\sum_{i=1}^{n} A_i^2}$
- $||B||$ is the L2 norm of $B$
- $\theta$ is the angle between the vectors

---

## The Range of Values

Cosine similarity ranges from -1 to 1:

- **1**: vectors point in exactly the same direction (identical orientation)
- **0**: vectors are perpendicular (orthogonal, no similarity)
- **-1**: vectors point in exactly opposite directions

For non-negative vectors (like word counts or TF-IDF), the range is 0 to 1 because all components are positive.

---

## Step-by-Step Computation

**Example:** $A = [3, 4]$ and $B = [4, 3]$

**Step 1: Compute the dot product**

$A \cdot B = 3 \times 4 + 4 \times 3 = 12 + 12 = 24$

**Step 2: Compute the magnitude of A**

$||A|| = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$

**Step 3: Compute the magnitude of B**

$||B|| = \sqrt{4^2 + 3^2} = \sqrt{16 + 9} = \sqrt{25} = 5$

**Step 4: Divide**

$\cos(\theta) = \frac{24}{5 \times 5} = \frac{24}{25} = 0.96$

The vectors are very similar (pointing in nearly the same direction).

---

## Why Ignore Magnitude?

Consider document similarity with word counts:

**Document A:** mentions "python" 100 times, "code" 50 times
**Document B:** mentions "python" 10 times, "code" 5 times

These documents have the same word proportions (2:1 ratio of python to code). They are likely about the same topic. But their Euclidean distance is large because of the different scales.

Cosine similarity sees them as identical (similarity = 1) because the vectors $[100, 50]$ and $[10, 5]$ point in the same direction.

---

## Geometric Interpretation

In 2D, cosine similarity is literally the cosine of the angle:

- Two vectors at 0 degrees: $\cos(0) = 1$
- Two vectors at 45 degrees: $\cos(45°) \approx 0.707$
- Two vectors at 90 degrees: $\cos(90°) = 0$
- Two vectors at 180 degrees: $\cos(180°) = -1$

In higher dimensions, the same principle applies. The formula computes the cosine of the angle in n-dimensional space.

---

## Cosine Similarity vs. Euclidean Distance

**Euclidean distance:**

$$
d(A, B) = \sqrt{\sum_{i=1}^{n} (A_i - B_i)^2}
$$

Measures the straight-line distance between endpoints.

**Key differences:**

Euclidean distance is affected by magnitude. Vectors $[1, 0]$ and $[100, 0]$ have large Euclidean distance but cosine similarity of 1.

Cosine similarity is affected only by direction. It treats $[1, 2, 3]$ and $[2, 4, 6]$ as identical.

**When to use which:**

- Cosine similarity: text similarity, recommendation systems, comparing embeddings
- Euclidean distance: spatial data, clustering when magnitude matters

---

## Cosine Distance

Cosine distance is derived from cosine similarity:

$$
\text{cosine distance} = 1 - \text{cosine similarity}
$$

This converts similarity (higher is more similar) to distance (lower is more similar):

- Cosine similarity of 1 becomes distance of 0
- Cosine similarity of 0 becomes distance of 1
- Cosine similarity of -1 becomes distance of 2

Cosine distance is used when algorithms expect a distance metric rather than a similarity score.

---

## Handling Zero Vectors

If either vector is all zeros, the magnitude is 0, and division by zero occurs.

$$
||[0, 0, 0]|| = 0
$$

**Solutions:**

- Return 0 similarity (or undefined) for zero vectors
- Add a small epsilon to the denominator
- Filter out zero vectors before comparison

In practice, zero vectors rarely occur in meaningful data (a document with no words, an embedding of nothing).

---

## Efficient Computation

For normalized vectors (magnitude = 1), cosine similarity simplifies to just the dot product:

$$
\cos(\theta) = A \cdot B \quad \text{when } ||A|| = ||B|| = 1
$$

This is why many systems pre-normalize their vectors:
- Normalize once during indexing
- Compute similarity with a single dot product
- Much faster for large-scale retrieval

---

## Applications in Machine Learning

**Document similarity:**
TF-IDF vectors of documents compared with cosine similarity.

**Semantic search:**
Query embedding compared to document embeddings.

**Recommendation systems:**
User preference vectors compared to item vectors.

**Word embeddings:**
Word2Vec, GloVe use cosine similarity to find similar words.

**Image retrieval:**
CNN feature vectors compared with cosine similarity.

**Duplicate detection:**
Near-duplicate documents have high cosine similarity.

---

## Batch Computation

For comparing one query against many documents, use matrix multiplication:

If $Q$ is the query vector (1 x d) and $D$ is a matrix of document vectors (n x d):

$$
\text{similarities} = \frac{Q \cdot D^T}{||Q|| \times ||D||_{\text{row-wise}}}
$$

With pre-normalized vectors:

$$
\text{similarities} = Q \cdot D^T
$$

This computes all n similarities in one operation, leveraging optimized linear algebra libraries.
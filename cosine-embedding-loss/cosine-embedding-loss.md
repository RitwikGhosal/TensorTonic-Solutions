## Measuring Similarity with Cosine

Cosine similarity measures the angle between two vectors:

$$
\cos(\theta) = \frac{a \cdot b}{||a|| \cdot ||b||}
$$

Where:
- $a \cdot b$ is the dot product
- $||a||$ and $||b||$ are the vector magnitudes

Range:
- $\cos(\theta) = 1$: vectors point in the same direction (most similar)
- $\cos(\theta) = 0$: vectors are orthogonal (unrelated)
- $\cos(\theta) = -1$: vectors point in opposite directions (most dissimilar)

---

## Cosine Embedding Loss

Cosine embedding loss trains embeddings using cosine similarity instead of Euclidean distance:

$$
L = \begin{cases} 1 - \cos(e_1, e_2) & \text{if } y = 1 \\ \max(0, \cos(e_1, e_2) - m) & \text{if } y = -1 \end{cases}
$$

Where:
- $e_1, e_2$ are the embedding vectors
- $y = 1$ for similar pairs
- $y = -1$ for dissimilar pairs
- $m$ is the margin (typically between -1 and 1)

Note: the label convention ($y \in \{-1, 1\}$) differs from standard contrastive loss.

---

## Breaking Down the Two Cases

**Case 1: Similar pair ($y = 1$)**

$$
L = 1 - \cos(e_1, e_2)
$$
- If $\cos = 1$ (identical direction): loss = 0
- If $\cos = 0$ (orthogonal): loss = 1
- If $\cos = -1$ (opposite): loss = 2
- Minimize loss by maximizing cosine similarity

**Case 2: Dissimilar pair ($y = -1$)**

$$
L = \max(0, \cos(e_1, e_2) - m)
$$
- If $\cos \leq m$: loss = 0 (sufficiently dissimilar)
- If $\cos > m$: loss = $\cos - m$ (penalize being too similar)
- The margin $m$ is the maximum allowed similarity for dissimilar pairs

---

## The Role of the Margin

The margin $m$ sets the threshold for dissimilar pairs:

**$m = 0$:**
- Dissimilar pairs should be orthogonal or negatively correlated
- Strict separation

**$m = 0.5$:**
- Dissimilar pairs can have some positive correlation
- More lenient

**$m = -0.5$:**
- Dissimilar pairs should be somewhat negatively correlated
- Very strict separation

Common choice: $m = 0$ or $m = 0.1$

---

## Numerical Examples

**Example 1: Similar pair, aligned**
- $e_1 = [1, 0, 0]$, $e_2 = [0.9, 0.1, 0]$ (normalized: $[0.994, 0.111, 0]$)
- $\cos(e_1, e_2) \approx 0.994$
- Loss: $1 - 0.994 = 0.006$ (very small)

**Example 2: Similar pair, orthogonal**
- $e_1 = [1, 0, 0]$, $e_2 = [0, 1, 0]$
- $\cos(e_1, e_2) = 0$
- Loss: $1 - 0 = 1.0$ (high penalty)

**Example 3: Dissimilar pair, similar direction (with $m = 0$)**
- $e_1 = [1, 0, 0]$, $e_2 = [0.8, 0.6, 0]$
- $\cos(e_1, e_2) = 0.8$
- Loss: $\max(0, 0.8 - 0) = 0.8$ (should be pushed apart)

**Example 4: Dissimilar pair, opposite direction (with $m = 0$)**
- $e_1 = [1, 0, 0]$, $e_2 = [-1, 0, 0]$
- $\cos(e_1, e_2) = -1$
- Loss: $\max(0, -1 - 0) = 0$ (already maximally dissimilar)

---

## Cosine vs. Euclidean Distance

**Euclidean distance (L2):**
- Measures absolute position difference
- Affected by vector magnitude
- Two vectors can be far in L2 but have the same direction

**Cosine similarity:**
- Measures angular similarity
- Invariant to magnitude (only direction matters)
- Two vectors with the same direction have cosine = 1 regardless of length

**When to use cosine:**
- When direction matters more than magnitude
- When embeddings have varying norms
- Text embeddings (TF-IDF, word2vec): document length should not affect similarity
- When you want scale-invariant comparisons

**When to use Euclidean:**
- When magnitude carries information
- When absolute position in embedding space matters
- Often used with L2-normalized embeddings (then cosine and L2 are equivalent)

---

## Relationship to L2 Distance

For L2-normalized vectors (unit vectors on hypersphere):

$$
||e_1 - e_2||_2^2 = 2(1 - \cos(e_1, e_2))
$$

This means:
- If embeddings are normalized: cosine and L2 are monotonically related
- Maximizing cosine $\equiv$ minimizing L2 distance
- Many implementations normalize embeddings, making the choice less critical

---

## The Gradient

For similar pairs ($y = 1$):

$$
\frac{\partial L}{\partial e_1} = -\frac{e_2 - (e_1 \cdot e_2) e_1 / ||e_1||^2}{||e_1|| \cdot ||e_2||}
$$

This points in the direction that increases cosine similarity.

For dissimilar pairs ($y = -1$) when $\cos > m$:

$$
\frac{\partial L}{\partial e_1} = \frac{e_2 - (e_1 \cdot e_2) e_1 / ||e_1||^2}{||e_1|| \cdot ||e_2||}
$$

This points in the direction that decreases cosine similarity.

---

## Implementation Notes

**Normalization:**
- Often apply L2 normalization before computing loss
- This makes the loss focus purely on direction
- Simplifies gradient computation

**Numerical stability:**
- Avoid division by zero for zero-norm vectors
- Clip cosine values to [-1, 1] before computing loss

**Temperature scaling:**
- Sometimes cosine similarity is scaled: $\cos(e_1, e_2) / \tau$
- Temperature $\tau$ controls the "sharpness" of similarity
- Used in InfoNCE and related losses

---

## Where Cosine Embedding Loss Is Used

- **Sentence similarity**: comparing sentence embeddings (SBERT)
- **Document retrieval**: finding similar documents
- **Face verification**: comparing face embeddings
- **Duplicate detection**: finding near-duplicate items
- **Recommendation systems**: user-item similarity
- **Any task where angular similarity is more meaningful than absolute distance
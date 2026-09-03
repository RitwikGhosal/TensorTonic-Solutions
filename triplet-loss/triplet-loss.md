## The Idea Behind Triplet Loss

Triplet loss learns embeddings by comparing three samples at a time:
- **Anchor (a)**: a reference sample
- **Positive (p)**: a sample similar to the anchor (same class/identity)
- **Negative (n)**: a sample dissimilar to the anchor (different class/identity)

The goal: learn embeddings where anchor is closer to positive than to negative.

$$
d(a, p) < d(a, n)
$$

Where $d$ is a distance function (usually L2/Euclidean distance).

---

## The Triplet Loss Formula

$$
L = \max(0, d(a, p) - d(a, n) + m)
$$

Where:
- $d(a, p)$ is the distance between anchor and positive embeddings
- $d(a, n)$ is the distance between anchor and negative embeddings
- $m$ is the margin (a positive constant)

The loss is zero when: $d(a, n) > d(a, p) + m$

This means the negative must be farther than the positive by at least margin $m$.

---

## Understanding the Margin

The margin $m$ prevents trivial solutions:

**Without margin (m = 0):**
- Loss is 0 whenever d(a, n) > d(a, p)
- The model could satisfy this with d(a, p) = 0.99 and d(a, n) = 1.0
- Embeddings would not be discriminative

**With margin (e.g., m = 0.2):**
- Loss is 0 only when d(a, n) > d(a, p) + 0.2
- Forces clear separation between positive and negative
- Typical values: 0.2 to 1.0 (depends on embedding normalization)

---

## Numerical Examples

Let margin m = 0.5.

**Example 1: Good triplet (satisfied)**
- d(anchor, positive) = 0.3
- d(anchor, negative) = 1.0
- Loss: max(0, 0.3 - 1.0 + 0.5) = max(0, -0.2) = 0

**Example 2: Violated triplet**
- d(anchor, positive) = 0.3
- d(anchor, negative) = 0.4
- Loss: max(0, 0.3 - 0.4 + 0.5) = max(0, 0.4) = 0.4

**Example 3: Badly violated triplet**
- d(anchor, positive) = 0.5
- d(anchor, negative) = 0.2 (negative is closer!)
- Loss: max(0, 0.5 - 0.2 + 0.5) = max(0, 0.8) = 0.8

---

## Types of Triplets

**Easy triplets:** $d(a, n) > d(a, p) + m$
- Loss is 0
- Already satisfied, no learning signal

**Semi-hard triplets:** $d(a, p) < d(a, n) < d(a, p) + m$
- Negative is farther than positive, but not by enough
- Provides useful gradient
- Often the best for learning

**Hard triplets:** $d(a, n) < d(a, p)$
- Negative is closer than positive
- Large loss, strong gradient
- Can destabilize training if used exclusively

---

## Triplet Mining Strategies

Random triplet sampling is inefficient because most triplets are easy:
- For a well-trained model, most negatives are far from the anchor
- Easy triplets give zero gradient

**Batch-hard mining:**
- For each anchor, find the hardest positive (farthest same-class sample)
- For each anchor, find the hardest negative (closest different-class sample)
- Aggressive but can be unstable

**Batch-semi-hard mining:**
- For each anchor, use all positives
- Select negatives that are farther than the positive but closer than positive + margin
- More stable than batch-hard

**Offline mining:**
- Compute all embeddings first
- Select informative triplets based on current distances
- Expensive but precise

---

## The Gradient

For a violated triplet:

$$
\frac{\partial L}{\partial f(a)} = \frac{f(a) - f(p)}{d(a, p)} - \frac{f(a) - f(n)}{d(a, n)}
$$

This pushes the anchor:
- Toward the positive (first term)
- Away from the negative (second term)

Similar gradients apply to the positive and negative embeddings:
- Positive is pushed toward anchor
- Negative is pushed away from anchor

---

## Triplet Loss vs. Contrastive Loss

**Contrastive loss:**
- Uses pairs: (anchor, positive) or (anchor, negative)
- Two separate terms for similar and dissimilar pairs
- Requires absolute distance thresholds

**Triplet loss:**
- Uses triplets: (anchor, positive, negative)
- Single term comparing relative distances
- Only requires positive to be closer than negative

Triplet loss is often preferred because:
- It focuses on relative ordering (more natural for retrieval)
- Does not require tuning absolute distance thresholds
- Better sample efficiency with good mining

---

## Batch All Triplet Loss

Computing loss over all valid triplets in a batch:

For a batch with $P$ identities and $K$ samples per identity:
- Each sample can be an anchor
- Positives: other samples of same identity ($K-1$ per anchor)
- Negatives: samples of different identities ($P-1) \times K$ per anchor)

Total triplets per batch: $P \times K \times (K-1) \times (P-1) \times K$

This can be thousands of triplets, but most are easy. Filtering to semi-hard or hard triplets is essential.

---

## Squared Distance Variant

Some implementations use squared distances:

$$
L = \max(0, d(a, p)^2 - d(a, n)^2 + m)
$$

Advantages:
- Avoids computing square roots
- Computationally cheaper

Disadvantage:
- Scale of margin changes (need to adjust m accordingly)

---

## Where Triplet Loss Is Used

- **Face recognition**: FaceNet uses triplet loss to learn face embeddings
- **Person re-identification**: matching people across camera views
- **Image retrieval**: finding similar images
- **Signature verification**: verifying if two signatures are from the same person
- **Speaker verification**: matching voice recordings
- **One-shot learning**: learning to compare with few examples

Best practices:
- Use L2-normalized embeddings (unit sphere)
- Combine with good triplet mining
- Margin around 0.2-0.5 for normalized embeddings
- Monitor the fraction of active (non-zero loss) triplets
## The Goal of Contrastive Learning

Contrastive learning trains a model to produce embeddings where:
- **Similar items are close together** in embedding space
- **Dissimilar items are far apart**

This is different from classification:
- Classification assigns items to fixed categories
- Contrastive learning learns a general similarity metric

---

## The Contrastive Loss Formula

For a pair of samples with label $y$:

$$
L = (1 - y) \cdot \frac{1}{2} D^2 + y \cdot \frac{1}{2} \max(0, m - D)^2
$$

Where:
- $y = 0$ if the pair is similar (same class)
- $y = 1$ if the pair is dissimilar (different classes)
- $D = ||f(x_1) - f(x_2)||_2$ is the Euclidean distance between embeddings
- $m$ is the margin (a hyperparameter)
- $f$ is the embedding function (neural network)

---

## Breaking Down the Two Cases

**Case 1: Similar pair (y = 0)**

$$
L = \frac{1}{2} D^2
$$
- Minimize the squared distance between embeddings
- Pull similar items together
- No margin involved

**Case 2: Dissimilar pair (y = 1)**

$$
L = \frac{1}{2} \max(0, m - D)^2
$$
- If D >= m: loss is 0 (already far enough apart)
- If D < m: loss is 0.5 * (m - D)^2 (push them apart)
- Push dissimilar items at least $m$ apart

---

## The Role of the Margin

The margin $m$ defines the boundary between "close" and "far":
- Similar pairs: should have distance approximately 0
- Dissimilar pairs: should have distance >= m

**Choosing m:**
- Common values: 1.0, 2.0 (depends on embedding normalization)
- If embeddings are L2-normalized (unit sphere): m in [0.5, 1.5] is typical
- Larger margin: more separation, but harder to achieve
- Smaller margin: easier to satisfy, but less discriminative

---

## Numerical Examples

Let margin m = 2.0.

**Example 1: Similar pair, close**
- Embeddings: e1 = [1, 0], e2 = [0.9, 0.1]
- Distance: D = sqrt(0.1^2 + 0.1^2) = 0.14
- Loss: 0.5 * (0.14)^2 = 0.01 (small, good)

**Example 2: Similar pair, far**
- Embeddings: e1 = [1, 0], e2 = [-1, 0]
- Distance: D = 2.0
- Loss: 0.5 * (2.0)^2 = 2.0 (high penalty)

**Example 3: Dissimilar pair, close**
- Embeddings: e1 = [1, 0], e2 = [0.8, 0.2]
- Distance: D = 0.28
- Loss: 0.5 * (2.0 - 0.28)^2 = 1.48 (push apart)

**Example 4: Dissimilar pair, far**
- Embeddings: e1 = [1, 0], e2 = [-2, 0]
- Distance: D = 3.0 > m
- Loss: max(0, 2.0 - 3.0)^2 = 0 (already satisfied)

---

## The Gradient

**For similar pairs (y = 0):**

$$
\frac{\partial L}{\partial e_1} = (e_1 - e_2)
$$
Gradient points from e2 toward e1, so e1 moves toward e2.

**For dissimilar pairs (y = 1) when D < m:**

$$
\frac{\partial L}{\partial e_1} = -(m - D) \cdot \frac{e_1 - e_2}{D}
$$
Gradient points in the opposite direction, pushing e1 away from e2.

**For dissimilar pairs when D >= m:**

$$
\frac{\partial L}{\partial e_1} = 0
$$
No gradient. The pair is already sufficiently separated.

---

## Creating Training Pairs

Contrastive loss requires pairs of samples. Strategies:

**From labeled data:**
- Same class: positive pair (y = 0)
- Different classes: negative pair (y = 1)
- Sample randomly or use hard negative mining

**Self-supervised (no labels):**
- Augmentations of the same image: positive pair
- Different images: negative pair
- This is the basis of SimCLR, MoCo, etc.

**Mining strategies:**
- Random: sample pairs uniformly
- Hard negative mining: find dissimilar pairs that are currently close
- Semi-hard mining: dissimilar pairs within margin but not too close

---

## The Embedding Space

Well-trained embeddings form clusters:
- Similar items cluster together
- Different clusters are separated by at least margin $m$
- The margin creates "dead zones" between clusters

Properties of the learned space:
- Distances are meaningful (unlike raw feature space)
- New classes can be added without retraining (zero-shot)
- Nearest neighbor search becomes effective

---

## Contrastive vs. Triplet Loss

Contrastive loss works with pairs. Triplet loss works with triplets (anchor, positive, negative).

**Contrastive:**
- Two terms: pull similar, push dissimilar
- Simpler to understand
- Requires binary similar/dissimilar labels

**Triplet:**
- Single term: positive should be closer than negative by margin
- More explicit relative comparison
- Harder negative mining is more natural

Both achieve similar goals; triplet loss is often considered more sample efficient.

---

## Where Contrastive Loss Is Used

- **Face recognition**: embeddings where same person = close, different people = far
- **Signature verification**: same signer vs. different signers
- **Siamese networks**: comparing two inputs for similarity
- **Self-supervised learning**: learning representations without labels
- **One-shot learning**: recognizing new classes with few examples
- **Image retrieval**: finding similar images in a database
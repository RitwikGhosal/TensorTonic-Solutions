## The Problem with Standard GANs

In a standard GAN, the discriminator outputs a probability that the input is real. The generator tries to maximize this probability for fake samples. This setup has a fundamental problem: **vanishing gradients**.

When the discriminator becomes too good at distinguishing real from fake, its output saturates at 0 for fake samples. The gradient of a saturated sigmoid is nearly zero, so the generator receives almost no learning signal. Training stalls.

Another problem is **mode collapse**: the generator learns to produce only a few types of outputs that fool the discriminator, ignoring the full diversity of the real data distribution.

---

## The Wasserstein Distance

Wasserstein GAN (WGAN) replaces the discriminator with a **critic** that estimates the Wasserstein distance (also called Earth Mover's distance) between the real and generated distributions.

Intuitively, the Wasserstein distance measures the minimum "cost" to transform one distribution into another, where cost is the amount of probability mass moved times the distance moved.

The key advantage: Wasserstein distance provides meaningful gradients even when the distributions do not overlap. Unlike JS divergence (used in standard GANs), it does not saturate.

---

## The Critic Function

The critic $f$ is a neural network that outputs an unbounded real number (not a probability). It tries to assign high scores to real samples and low scores to fake samples.

The Wasserstein distance can be approximated as:

$$
W(P_r, P_g) \approx \max_f \left( \mathbb{E}_{x \sim P_r}[f(x)] - \mathbb{E}_{z \sim P_z}[f(G(z))] \right)
$$

where:

- $P_r$ is the real data distribution
- $P_g$ is the generated data distribution
- $G(z)$ is the generator output for noise $z$
- $f$ is the critic function

The critic maximizes the difference between its scores on real and fake samples.

---

## The Critic Loss

From the critic's perspective, it wants to:

- Give **high scores** to real samples: maximize $\mathbb{E}[f(x)]$
- Give **low scores** to fake samples: minimize $\mathbb{E}[f(G(z))]$

The critic loss (to be minimized) is:

$$
L_{\text{critic}} = \mathbb{E}[f(G(z))] - \mathbb{E}[f(x)]
$$

This is equivalent to maximizing $\mathbb{E}[f(x)] - \mathbb{E}[f(G(z))]$.

In code terms:

- Compute critic output on real batch: $\text{real\_scores} = f(x)$
- Compute critic output on fake batch: $\text{fake\_scores} = f(G(z))$
- Loss $= \text{mean}(\text{fake\_scores}) - \text{mean}(\text{real\_scores})$

---

## The Lipschitz Constraint

There is a critical requirement: the critic must be a **1-Lipschitz function**. This means:

$$
|f(x_1) - f(x_2)| \leq ||x_1 - x_2||
$$

for all pairs of inputs. The function cannot change faster than a slope of 1.

Without this constraint, the critic could assign arbitrarily large scores to real samples and arbitrarily negative scores to fake samples, and the Wasserstein distance estimate would be meaningless.

---

## Enforcing Lipschitz: Weight Clipping (Original WGAN)

The original WGAN paper enforced the Lipschitz constraint by **clipping** the critic's weights to a small range like $[-0.01, 0.01]$ after each update.

This works but has problems:

- If the clipping range is too small, the critic has limited capacity
- If the clipping range is too large, training becomes unstable
- Weight clipping biases the critic toward simple functions

---

## Enforcing Lipschitz: Gradient Penalty (WGAN-GP)

WGAN-GP (Gradient Penalty) uses a smarter approach. Instead of clipping weights, it adds a **penalty term** that discourages the gradient norm from deviating from 1.

The full critic loss becomes:

$$
L_{\text{critic}} = \mathbb{E}[f(G(z))] - \mathbb{E}[f(x)] + \lambda \mathbb{E}\left[(||\nabla_{\hat{x}} f(\hat{x})||_2 - 1)^2\right]
$$

where:

- $\hat{x}$ is a random interpolation between real and fake samples
- $\lambda$ is the penalty coefficient (typically 10)
- $||\nabla_{\hat{x}} f(\hat{x})||_2$ is the L2 norm of the critic's gradient at $\hat{x}$

---

## The Interpolation

The interpolated samples $\hat{x}$ are computed as:

$$
\hat{x} = \epsilon \cdot x + (1 - \epsilon) \cdot G(z)
$$

where $\epsilon$ is sampled uniformly from $[0, 1]$ for each sample in the batch.

This creates points along straight lines between real and fake samples. The gradient penalty is enforced at these interpolated points, which empirically works better than enforcing it only on real or fake samples.

---

## Computing the Gradient Penalty

For each interpolated sample $\hat{x}$:

1. Compute the critic output $f(\hat{x})$
2. Compute the gradient $\nabla_{\hat{x}} f(\hat{x})$ using automatic differentiation
3. Compute the L2 norm of this gradient: $||\nabla_{\hat{x}} f(\hat{x})||_2$
4. Penalize deviation from 1: $(||\nabla||_2 - 1)^2$

Average over the batch to get the gradient penalty term.

---

## Why Gradient Norm of 1?

A function is 1-Lipschitz if and only if its gradient norm is at most 1 everywhere. By encouraging the gradient norm to be exactly 1 (not just less than 1), we push the critic to use its full capacity while staying within the constraint.

The penalty $(||\nabla||_2 - 1)^2$ is zero when the norm is exactly 1 and positive otherwise.

---

## Training Dynamics

With WGAN-GP:

- The critic loss correlates with image quality (unlike standard GAN discriminator loss)
- Training is more stable across different architectures
- Mode collapse is reduced because the Wasserstein distance captures diversity
- No need for careful balancing of generator vs. discriminator updates

The critic is typically trained for 5 iterations per generator iteration to ensure it provides a good estimate of the Wasserstein distance.
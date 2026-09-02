## The Problem Batch Normalization Solves

Training deep neural networks is difficult because the distribution of each layer's inputs changes during training. As earlier layers update their weights, the statistics of their outputs shift. Later layers must constantly adapt to these shifting distributions.

This phenomenon is called **internal covariate shift**. It slows training because:
- Layers cannot assume stable input statistics
- Gradients can vanish or explode more easily
- Lower learning rates are required for stability

Batch Normalization (BatchNorm) addresses this by normalizing layer inputs to have consistent statistics.

---

## The Core Idea

For each mini-batch, normalize the activations to have zero mean and unit variance:

$$
\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

where:
- $x$ is the input activation
- $\mu_B$ is the mean over the mini-batch
- $\sigma_B^2$ is the variance over the mini-batch
- $\epsilon$ is a small constant for numerical stability (e.g., $10^{-5}$)

This normalization is applied independently to each feature/channel.

---

## The Full BatchNorm Transform

After normalization, BatchNorm applies a learnable scale and shift:

$$
y = \gamma \hat{x} + \beta
$$

where:
- $\gamma$ (gamma) is the learned scale parameter
- $\beta$ (beta) is the learned shift parameter
- Both have the same shape as the feature dimension

**Why scale and shift?**

Pure normalization (forcing mean=0, variance=1) might limit what the layer can represent. The learnable parameters allow the network to undo the normalization if needed. If $\gamma = \sigma$ and $\beta = \mu$, the original activation is recovered.

---

## Step-by-Step Computation

**Input:** Mini-batch of activations $x$ with shape (batch_size, features)

**Step 1: Compute batch mean**

$$
\mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i
$$

**Step 2: Compute batch variance**

$$
\sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2
$$

**Step 3: Normalize**

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

**Step 4: Scale and shift**

$$
y_i = \gamma \hat{x}_i + \beta
$$

---

## Worked Example

**Mini-batch of 4 samples, 1 feature:**

$x = [2, 4, 6, 8]$

**Step 1: Mean**
$\mu_B = (2 + 4 + 6 + 8) / 4 = 5$

**Step 2: Variance**
$\sigma_B^2 = ((2-5)^2 + (4-5)^2 + (6-5)^2 + (8-5)^2) / 4$
$= (9 + 1 + 1 + 9) / 4 = 5$

**Step 3: Normalize** (with $\epsilon = 0$)
$\hat{x} = [\frac{2-5}{\sqrt{5}}, \frac{4-5}{\sqrt{5}}, \frac{6-5}{\sqrt{5}}, \frac{8-5}{\sqrt{5}}]$
$= [-1.34, -0.45, 0.45, 1.34]$

**Step 4: Scale and shift** (with $\gamma = 2$, $\beta = 1$)
$y = 2 \cdot [-1.34, -0.45, 0.45, 1.34] + 1$
$= [-1.68, 0.10, 1.90, 3.68]$

---

## BatchNorm for Convolutional Layers

For conv layers with shape (batch, channels, height, width):

- Compute mean and variance **per channel** across batch, height, and width
- $\gamma$ and $\beta$ have shape (channels,)
- Each channel is normalized independently

This means spatial locations within a channel share the same normalization statistics.

---

## Training vs. Inference

**During training:**
- Use mini-batch statistics ($\mu_B$, $\sigma_B^2$)
- Statistics vary from batch to batch
- Also maintain running averages for inference

**During inference:**
- Use running averages accumulated during training
- Fixed statistics, deterministic output
- Running mean: $\mu_{\text{run}} = \alpha \mu_{\text{run}} + (1-\alpha) \mu_B$
- Running variance: similar exponential moving average
- Typical $\alpha = 0.9$ or $0.99$

This is why you must set the model to "eval mode" during inference.

---

## Benefits of Batch Normalization

**1. Faster training**
- Allows higher learning rates
- Reduces sensitivity to initialization
- Converges in fewer iterations

**2. Regularization effect**
- Batch statistics add noise (each sample's normalization depends on other samples in batch)
- Acts as mild regularization, can reduce need for dropout

**3. Reduces vanishing/exploding gradients**
- Keeps activations in a reasonable range
- Gradients flow more consistently through layers

**4. Less sensitivity to hyperparameters**
- Networks are more forgiving of suboptimal learning rates and initialization

---

## Limitations

**1. Batch size dependence**
- Small batches give noisy statistics
- Performance degrades with batch size < 16
- Not suitable for batch size = 1

**2. Different behavior train/test**
- Must track running statistics
- Can cause bugs if mode not set correctly

**3. Not ideal for RNNs**
- Sequence lengths vary
- Batch statistics across time steps are problematic
- Layer Normalization often preferred for RNNs

---

## Alternatives to BatchNorm

**Layer Normalization:**
Normalize across features, not batch. Works with batch size = 1. Preferred for Transformers and RNNs.

**Instance Normalization:**
Normalize each sample independently (across spatial dimensions). Used in style transfer.

**Group Normalization:**
Normalize across groups of channels. Compromise between Layer and Instance norm.

**Weight Normalization:**
Normalize weights instead of activations.

---

## Where to Place BatchNorm

**Common placement:** After linear/conv layer, before activation

Conv -> BatchNorm -> ReLU

**Alternative:** After activation

Conv -> ReLU -> BatchNorm

Both work; the first is more common. Recent architectures sometimes omit BatchNorm entirely (using other techniques).

---

## Learnable Parameters

For a layer with $C$ features/channels:

- $\gamma$: C parameters (initialized to 1)
- $\beta$: C parameters (initialized to 0)

Total: $2C$ learnable parameters per BatchNorm layer.

Non-learnable (tracking only):
- Running mean: C values
- Running variance: C values

---

## The Gradient Through BatchNorm

Backpropagation through BatchNorm is more complex than regular layers because each output depends on all inputs in the batch (through $\mu_B$ and $\sigma_B^2$).

The gradients are:

$$
\frac{\partial L}{\partial \gamma} = \sum_i \frac{\partial L}{\partial y_i} \hat{x}_i
$$

$$
\frac{\partial L}{\partial \beta} = \sum_i \frac{\partial L}{\partial y_i}
$$

The gradient w.r.t. input involves the chain rule through the normalization statistics.
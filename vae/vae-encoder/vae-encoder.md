# <span style="font-size: 20px;">VAE Encoder</span>

<span style="font-size: 14px;">The encoder, or recognition model, is the component of a Variational Autoencoder (VAE) that maps an input $x$ to the parameters of an approximate posterior distribution $q_\phi(z|x)$. Rather than producing a single latent point, it outputs two vectors -- a mean $\mu$ and a log-variance $\log \sigma^2$ -- that together define a Gaussian distribution over latent space. This design was introduced by Kingma and Welling in "Auto-Encoding Variational Bayes" (2014).</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">The VAE encoder is a neural network parameterized by weights $\phi$ that takes a data point $x$ and produces the sufficient statistics of a diagonal Gaussian distribution in latent space. It outputs two vectors:</span>

* <span style="font-size: 14px;">**$\mu$ (mean vector):** The center of the approximate posterior. This is where the encoder believes the input most likely lives in latent space.</span>
* <span style="font-size: 14px;">**$\log \sigma^2$ (log-variance vector):** The log of the variance, controlling the uncertainty or spread of the distribution around $\mu$.</span>

<span style="font-size: 14px;">These outputs come from two separate linear projections from a shared representation. In the simplest form -- a linear encoder with no hidden layers -- the encoder takes raw input $x \in \mathbb{R}^{d_x}$ and computes:</span>

$$
\mu = x W_\mu, \quad \log \sigma^2 = x W_{\log\sigma^2}
$$

<span style="font-size: 14px;">where $W_\mu \in \mathbb{R}^{d_x \times d_z}$ and $W_{\log\sigma^2} \in \mathbb{R}^{d_x \times d_z}$ are independent weight matrices, and $d_z$ is the latent dimension. The encoder returns the tuple $(\mu, \log \sigma^2)$, used downstream for sampling via the reparameterization trick and for computing the KL divergence.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">The encoder defines the approximate posterior as a diagonal Gaussian:</span>

$$
q_\phi(z|x) = \mathcal{N}(z; \mu_\phi(x), \text{diag}(\sigma_\phi^2(x)))
$$

* <span style="font-size: 14px;">**$q_\phi(z|x)$:** The approximate posterior, parameterized by encoder weights $\phi$.</span>
* <span style="font-size: 14px;">**$\mathcal{N}(z; \mu, \text{diag}(\sigma^2))$:** Multivariate Gaussian with mean $\mu$ and diagonal covariance. Each latent dimension is independent given $x$.</span>
* <span style="font-size: 14px;">**$\mu_\phi(x) \in \mathbb{R}^{d_z}$:** Mean vector. Element $\mu_j$ is the expected value of the $j$-th latent dimension.</span>
* <span style="font-size: 14px;">**$\sigma_\phi^2(x) \in \mathbb{R}^{d_z}$:** Variance vector, recovered as $\sigma^2 = \exp(\log \sigma^2)$ from the encoder's log-variance output.</span>

<span style="font-size: 14px;">The two linear projections:</span>

$$
\mu = x W_\mu \in \mathbb{R}^{d_z}
$$

$$
\log \sigma^2 = x W_{\log\sigma^2} \in \mathbb{R}^{d_z}
$$

<span style="font-size: 14px;">Given these parameters, a latent sample is drawn using the reparameterization trick (a separate component):</span>

$$
z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

<span style="font-size: 14px;">where $\sigma = \exp(\frac{1}{2} \log \sigma^2)$ and $\odot$ denotes element-wise multiplication. The encoder's job ends at producing $(\mu, \log \sigma^2)$; the sampling step is downstream.</span>

---

## <span style="font-size: 16px;">Why Output Distribution Parameters, Not Points</span>

<span style="font-size: 14px;">A standard autoencoder maps each input to a single point: $z = f(x)$. The VAE encoder instead maps each input to a distribution. This is not an arbitrary choice -- it is required by variational inference. The true posterior $p(z|x)$ is itself a distribution, but computing it requires an intractable integral over the entire latent space. The encoder's distribution $q_\phi(z|x)$ is a tractable approximation optimized to be close to the true posterior.</span>

<span style="font-size: 14px;">Outputting a distribution serves three essential purposes:</span>

* <span style="font-size: 14px;">**Variational inference requires it:** The ELBO objective involves an expectation under $q_\phi(z|x)$, specifically $\mathbb{E}_{q_\phi(z|x)}[\log p(x|z)]$. You cannot take an expectation under a point; you need a distribution to sample from and compute KL divergences against.</span>
* <span style="font-size: 14px;">**Regularization through the prior:** The KL term $D_{KL}(q_\phi(z|x) \| p(z))$ penalizes posteriors that deviate from the prior $p(z) = \mathcal{N}(0, I)$. This prevents the encoder from collapsing each input onto a deterministic point and ensures the latent space has smooth structure for interpolation and generation.</span>
* <span style="font-size: 14px;">**Enabling generation:** At test time, we sample $z \sim \mathcal{N}(0, I)$ and decode. For this to produce coherent outputs, the decoder must have seen latent codes near the prior during training. Distributional outputs that overlap with the prior ensure the decoder learns to handle prior samples.</span>

<span style="font-size: 14px;">The tension between reconstruction accuracy (narrow posteriors) and regularization (posteriors close to the prior) is the fundamental trade-off in VAE training, controlled by the relative weight of the reconstruction loss and the KL term.</span>

---

## <span style="font-size: 16px;">Why Log-Variance Instead of Variance</span>

<span style="font-size: 14px;">The encoder outputs $\log \sigma^2$ rather than $\sigma^2$ directly. This is a critical design choice with both numerical and optimization motivations.</span>

<span style="font-size: 14px;">**The constraint problem:** Variance must be strictly positive ($\sigma^2 > 0$). A linear layer can output any real number, including zero or negative values. Enforcing positivity with softplus or ReLU + epsilon introduces saturation or dead zones.</span>

<span style="font-size: 14px;">**The log-variance solution:** $\log \sigma^2$ ranges over $(-\infty, +\infty)$, matching a linear layer's natural output range. The variance is recovered as:</span>

$$
\sigma^2 = \exp(\log \sigma^2)
$$

<span style="font-size: 14px;">Since $\exp(\cdot)$ maps any real number to a strictly positive value, positivity of $\sigma^2$ is guaranteed by construction.</span>

<span style="font-size: 14px;">**Numerical range advantages:**</span>

* <span style="font-size: 14px;">**Small variances:** $\sigma^2 = 0.01$ corresponds to $\log \sigma^2 = -4.6$, a moderately negative number rather than a near-zero value that is hard for linear layers to produce stably.</span>
* <span style="font-size: 14px;">**Large variances:** $\sigma^2 = 10.0$ corresponds to $\log \sigma^2 = 2.3$, a manageable value.</span>
* <span style="font-size: 14px;">**Unit variance:** $\sigma^2 = 1.0$ corresponds to $\log \sigma^2 = 0.0$ -- the prior variance maps to zero, a natural resting point.</span>

<span style="font-size: 14px;">**KL divergence simplification:** The log-variance appears directly in the KL formula for a diagonal Gaussian against a standard normal:</span>

$$
D_{KL} = -\frac{1}{2} \sum_{j=1}^{d_z} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2 \right)
$$

<span style="font-size: 14px;">Here $\log \sigma_j^2$ is used directly from the encoder's output, and $\sigma_j^2 = \exp(\log \sigma_j^2)$ appears only once. This avoids $\log(\exp(\cdot))$ roundtrips that can cause numerical instability.</span>

---

## <span style="font-size: 16px;">The Two Projections</span>

<span style="font-size: 14px;">The mean and log-variance are computed by two completely separate linear projections sharing no weights. From input $x$ (or hidden representation $h$), the two projections operate independently:</span>

$$
\mu = h W_\mu + b_\mu
$$

$$
\log \sigma^2 = h W_{\log\sigma^2} + b_{\log\sigma^2}
$$

<span style="font-size: 14px;">**Why separate weights?** The two outputs encode fundamentally different information:</span>

* <span style="font-size: 14px;">**$\mu$ encodes location:** Where in latent space the input maps. This captures identity, class, and content features.</span>
* <span style="font-size: 14px;">**$\log \sigma^2$ encodes uncertainty:** How confident the encoder is about that location. Ambiguous inputs should have higher variance; clear inputs should have lower variance.</span>

<span style="font-size: 14px;">Shared weights would force a rigid coupling between location and uncertainty, preventing the encoder from independently adjusting where a point maps and how certain it is about that mapping.</span>

<span style="font-size: 14px;">**Independent learning dynamics:** The reconstruction loss primarily shapes $W_\mu$ (pushing means toward latent locations that reconstruct well), while the KL term shapes both $W_\mu$ and $W_{\log\sigma^2}$ (pushing the distribution toward the prior). Separate weights let each projection specialize without interference.</span>

<span style="font-size: 14px;">**Shapes:** Both projections output vectors of size $d_z$, so $W_\mu$ and $W_{\log\sigma^2}$ have the same shape. The encoder's total output is the tuple $(\mu, \log \sigma^2)$, a $2 \times d_z$-dimensional description of the approximate posterior.</span>

---

## <span style="font-size: 16px;">Paper Context: Kingma and Welling (2014)</span>

<span style="font-size: 14px;">In "Auto-Encoding Variational Bayes," the encoder is called the **recognition model**: "We introduce a recognition model $q_\phi(z|x)$: an approximation to the intractable true posterior $p_\theta(z|x)$." The key contribution is showing this recognition model can be trained jointly with the decoder by optimizing the ELBO:</span>

$$
\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))
$$

<span style="font-size: 14px;">The first term measures reconstruction quality under encoder samples; the second is KL divergence against the prior. The encoder's parameters $\phi$ appear in both -- it must balance useful codes for reconstruction with staying close to the prior.</span>

<span style="font-size: 14px;">Before the VAE, variational inference required deriving custom update equations per model, often relying on conjugate priors. The VAE showed that a neural network recognition model plus the reparameterization trick makes variational inference general and scalable via SGD. The paper demonstrates this on MNIST and Frey Face using MLPs with the two-projection architecture.</span>

---

## <span style="font-size: 16px;">The Amortized Inference Insight</span>

<span style="font-size: 14px;">Traditional variational inference optimizes separate variational parameters for each data point -- $N$ separate $(\mu_i, \log \sigma_i^2)$ pairs for $N$ observations. This **per-datapoint inference** scales poorly.</span>

<span style="font-size: 14px;">The VAE encoder performs **amortized inference**: a single network with shared parameters $\phi$ maps any input $x$ to its variational parameters. Instead of $O(N \times d_z)$ variational parameters, you have $O(|\phi|)$ network parameters that generalize across all inputs.</span>

<span style="font-size: 14px;">Practical consequences:</span>

* <span style="font-size: 14px;">**Generalization to unseen data:** The trained encoder computes posteriors for inputs never seen during training. Per-datapoint inference has no parameters for new examples.</span>
* <span style="font-size: 14px;">**Constant-time inference:** One forward pass gives $q_\phi(z|x)$, versus running an optimization loop per input.</span>
* <span style="font-size: 14px;">**Shared statistical strength:** The encoder learns common patterns across data, so information from one example improves inference for similar examples.</span>

<span style="font-size: 14px;">The trade-off is the **amortization gap**: one set of weights for all inputs means the encoder may not perfectly optimize parameters for any individual input. In practice, this gap is small and the efficiency gains are enormous. Kingma and Welling state: "the variational parameters $\phi$ are not optimized per datapoint but rather are shared across data points, hence amortizing the cost of inference."</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a minimal VAE encoder with input dimension $d_x = 3$ and latent dimension $d_z = 2$, with no hidden layers -- just two linear projections.</span>

<span style="font-size: 14px;">**Input vector:**</span>

$$
x = [1.0, \; 0.5, \; -0.5]
$$

<span style="font-size: 14px;">**Weight matrix for $\mu$:**</span>

$$
W_\mu = \begin{pmatrix} 0.2 & -0.1 \\ 0.4 & 0.3 \\ -0.3 & 0.5 \end{pmatrix}
$$

<span style="font-size: 14px;">**Weight matrix for $\log \sigma^2$:**</span>

$$
W_{\log\sigma^2} = \begin{pmatrix} 0.1 & 0.0 \\ -0.2 & 0.6 \\ 0.3 & -0.4 \end{pmatrix}
$$

### <span style="font-size: 14px;">Computing $\mu$</span>

<span style="font-size: 14px;">$\mu = x W_\mu$:</span>

$$
\mu_1 = (1.0)(0.2) + (0.5)(0.4) + (-0.5)(-0.3) = 0.2 + 0.2 + 0.15 = 0.55
$$

$$
\mu_2 = (1.0)(-0.1) + (0.5)(0.3) + (-0.5)(0.5) = -0.1 + 0.15 - 0.25 = -0.2
$$

$$
\mu = [0.55, \; -0.2]
$$

### <span style="font-size: 14px;">Computing $\log \sigma^2$</span>

<span style="font-size: 14px;">$\log \sigma^2 = x W_{\log\sigma^2}$:</span>

$$
(\log \sigma^2)_1 = (1.0)(0.1) + (0.5)(-0.2) + (-0.5)(0.3) = 0.1 - 0.1 - 0.15 = -0.15
$$

$$
(\log \sigma^2)_2 = (1.0)(0.0) + (0.5)(0.6) + (-0.5)(-0.4) = 0.0 + 0.3 + 0.2 = 0.5
$$

$$
\log \sigma^2 = [-0.15, \; 0.5]
$$

### <span style="font-size: 14px;">Interpreting the Output</span>

<span style="font-size: 14px;">The encoder returns $(\mu, \log \sigma^2) = ([0.55, -0.2], \; [-0.15, 0.5])$, defining a 2D diagonal Gaussian:</span>

<span style="font-size: 14px;">**Latent dimension 1:**</span>

* <span style="font-size: 14px;">**Mean:** $\mu_1 = 0.55$</span>
* <span style="font-size: 14px;">**Log-variance:** $-0.15$, so $\sigma_1^2 = \exp(-0.15) = 0.861$, $\sigma_1 = 0.928$</span>

<span style="font-size: 14px;">**Latent dimension 2:**</span>

* <span style="font-size: 14px;">**Mean:** $\mu_2 = -0.2$</span>
* <span style="font-size: 14px;">**Log-variance:** $0.5$, so $\sigma_2^2 = \exp(0.5) = 1.649$, $\sigma_2 = 1.284$</span>

<span style="font-size: 14px;">The first dimension has negative log-variance, giving variance below 1.0 (tighter than the prior). The second has positive log-variance, giving variance above 1.0 (wider than the prior). Both are valid -- $\exp(\cdot)$ guarantees positivity regardless of sign.</span>

<span style="font-size: 14px;">The approximate posterior for this input is:</span>

$$
q_\phi(z|x) = \mathcal{N}\left(z; \begin{pmatrix} 0.55 \\ -0.2 \end{pmatrix}, \begin{pmatrix} 0.861 & 0 \\ 0 & 1.649 \end{pmatrix} \right)
$$

<span style="font-size: 14px;">The zero off-diagonal entries reflect the diagonal Gaussian assumption -- each latent dimension is independent given the input.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Outputting variance instead of log-variance:** A linear layer can output zero or negative values, which are invalid as variances. If this raw output is passed to $\exp(\cdot)$ (assuming it is log-variance) or used in the KL formula, the loss will be wrong and training will diverge.</span>
* <span style="font-size: 14px;">**Sharing weights between $\mu$ and $\log \sigma^2$:** Using a single output of size $2 d_z$ that is split is fine -- the split is equivalent to two separate projections. But using a single output of size $d_z$ for both destroys the encoder's ability to independently control location and uncertainty.</span>
* <span style="font-size: 14px;">**Forgetting that log-variance can be negative:** Adding ReLU or clamping at zero is wrong. $\log \sigma^2 < 0$ simply means $\sigma^2 < 1$ (tighter than the prior), which is essential for the encoder to express high confidence. Clamping forces $\sigma^2 \geq 1$, preventing the encoder from ever being more certain than the prior.</span>
* <span style="font-size: 14px;">**Wrong output shapes:** Both $\mu$ and $\log \sigma^2$ must have shape $(\text{batch\_size}, d_z)$. Mismatched dimensionality causes the reparameterization trick and KL computation to fail or produce silently wrong results from broadcasting.</span>
* <span style="font-size: 14px;">**Confusing $\log \sigma^2$ with $\log \sigma$:** Some implementations output $\log \sigma$ (log standard deviation) instead of $\log \sigma^2$ (log variance). Both work, but the KL formula and reparameterization must match. Mixing them introduces a factor-of-2 error in the regularization term.</span>
* <span style="font-size: 14px;">**Ignoring the return format:** The encoder must return a tuple $(\mu, \log \sigma^2)$, not a concatenated tensor. Downstream components expect two separate tensors; forgetting to split causes shape mismatches.</span>

---
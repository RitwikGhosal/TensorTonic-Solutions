# <span style="font-size: 20px;">ELBO Loss</span>

<span style="font-size: 14px;">The Evidence Lower Bound (ELBO) loss is the training objective for Variational Autoencoders (VAEs). Introduced by Kingma and Welling in their 2014 paper "Auto-Encoding Variational Bayes," it combines two terms: a reconstruction loss that measures how faithfully the decoder recovers the input, and a KL divergence term that regularizes the latent space. Together, these two terms form a tractable lower bound on the intractable log-evidence $\log p(x)$.</span>

<span style="font-size: 14px;">In practice, the ELBO is maximized during training. Since gradient-based optimizers minimize, we minimize the negative ELBO, which equals the sum of the reconstruction loss and the KL divergence. The result is a single scalar loss that trains the encoder and decoder jointly.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">A VAE defines a generative model $p_\theta(x, z) = p_\theta(x|z) p(z)$ with a latent variable $z$, a prior $p(z) = \mathcal{N}(0, I)$, and a decoder $p_\theta(x|z)$. The goal is to maximize the marginal log-likelihood $\log p_\theta(x)$, but this requires integrating over all possible $z$ values, which is intractable.</span>

<span style="font-size: 14px;">To make training feasible, the VAE introduces an encoder network $q_\phi(z|x)$ that approximates the true posterior $p_\theta(z|x)$. The ELBO provides a lower bound on $\log p_\theta(x)$ that can be optimized with gradient descent. Since the ELBO is always less than or equal to $\log p_\theta(x)$, maximizing the ELBO pushes up the log-evidence from below.</span>

<span style="font-size: 14px;">The training loss is the negative ELBO: reconstruction loss plus KL divergence. Minimizing this loss simultaneously trains the encoder to produce useful latent codes and the decoder to reconstruct accurately.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

### <span style="font-size: 14px;">The ELBO</span>

<span style="font-size: 14px;">The evidence lower bound is:</span>

$$
\text{ELBO} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))
$$

<span style="font-size: 14px;">The first term is the expected log-likelihood of the reconstruction. The second term is the KL divergence between the approximate posterior and the prior. Since we minimize loss, we use the negative ELBO:</span>

$$
\mathcal{L} = -\text{ELBO} = \underbrace{-\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{Reconstruction Loss}} + \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{KL Divergence}}
$$

### <span style="font-size: 14px;">MSE Reconstruction Loss</span>

<span style="font-size: 14px;">When the decoder assumes a Gaussian output distribution with fixed variance, the negative log-likelihood reduces to mean squared error. For a single sample with $D$ features:</span>

$$
\mathcal{L}_{\text{recon}} = \sum_{i=1}^{D} (x_i - \hat{x}_i)^2
$$

<span style="font-size: 14px;">This is summed over features for each sample, then averaged over the batch of $B$ samples:</span>

$$
\mathcal{L}_{\text{recon}} = \frac{1}{B} \sum_{b=1}^{B} \sum_{i=1}^{D} (x_i^{(b)} - \hat{x}_i^{(b)})^2
$$

### <span style="font-size: 14px;">KL Divergence (Closed-Form)</span>

<span style="font-size: 14px;">When $q_\phi(z|x) = \mathcal{N}(\mu, \text{diag}(\sigma^2))$ and $p(z) = \mathcal{N}(0, I)$, the KL divergence has an analytical solution. For a latent space of dimension $L$:</span>

$$
D_{KL} = -\frac{1}{2} \sum_{j=1}^{L} \bigl(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\bigr)
$$

<span style="font-size: 14px;">Since the encoder outputs $\log \sigma^2$ (i.e., `log_var`) directly rather than $\sigma$, we substitute $\log \sigma_j^2$ for $\log(\sigma_j^2)$ and $\exp(\log \sigma_j^2)$ for $\sigma_j^2$:</span>

$$
D_{KL} = -\frac{1}{2} \sum_{j=1}^{L} \bigl(1 + \text{log\_var}_j - \mu_j^2 - \exp(\text{log\_var}_j)\bigr)
$$

<span style="font-size: 14px;">This is summed over latent dimensions per sample, then averaged over the batch.</span>

### <span style="font-size: 14px;">Total Loss</span>

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + D_{KL}
$$

---

## <span style="font-size: 16px;">Why Two Terms</span>

<span style="font-size: 14px;">The ELBO loss has two terms because training a VAE requires satisfying two competing goals simultaneously.</span>

<span style="font-size: 14px;">**Reconstruction ensures faithful decoding.** Without the reconstruction term, the encoder could map every input to the same point in latent space, and the decoder would output the dataset mean. The reconstruction loss forces the encoder-decoder pair to preserve information: the latent code $z$ must carry enough detail about $x$ for the decoder to reconstruct it.</span>

<span style="font-size: 14px;">**KL ensures a structured latent space.** Without the KL term, the encoder would encode each training example as a distinct, isolated point with near-zero variance. The latent space would be useless for generation because sampling from $p(z) = \mathcal{N}(0, I)$ would land in regions the decoder has never seen. The KL term forces the approximate posterior to stay close to the prior, keeping the latent space smooth and well-covered.</span>

<span style="font-size: 14px;">**Both are necessary.** Reconstruction alone gives a deterministic autoencoder with no generative capability. KL alone gives a model that ignores the data entirely. The ELBO combines them into a principled objective derived from variational inference, and the balance between the two terms determines the quality of both reconstruction and generation.</span>

---

## <span style="font-size: 16px;">The Reconstruction Term</span>

<span style="font-size: 14px;">The reconstruction term measures how well the decoder $p_\theta(x|z)$ recovers the input $x$ from the latent code $z$. In the ELBO derivation, this term is the expected log-likelihood $\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$.</span>

<span style="font-size: 14px;">When the decoder is modeled as a Gaussian with fixed unit variance, $p_\theta(x|z) = \mathcal{N}(x; \hat{x}, I)$, the log-likelihood becomes:</span>

$$
\log p_\theta(x|z) = -\frac{D}{2}\log(2\pi) - \frac{1}{2}\sum_{i=1}^{D}(x_i - \hat{x}_i)^2
$$

<span style="font-size: 14px;">Since the constant $-\frac{D}{2}\log(2\pi)$ does not depend on the model parameters, it is dropped during optimization. The negative of the remaining term gives the MSE reconstruction loss: $\sum_{i=1}^{D}(x_i - \hat{x}_i)^2$.</span>

<span style="font-size: 14px;">**Why MSE corresponds to a Gaussian decoder.** MSE follows directly from assuming the decoder outputs the mean of a Gaussian with identity covariance. If the decoder instead assumed a Bernoulli distribution (appropriate for binary data), the reconstruction loss would be binary cross-entropy. The loss function and the decoder's output distribution are two sides of the same coin.</span>

<span style="font-size: 14px;">**Reduction: sum over features, average over batch.** Each sample's reconstruction error sums over all $D$ feature dimensions. This sum is then averaged over the $B$ samples in the batch. Summing over features (rather than averaging) ensures the reconstruction term scales with the data dimensionality, which matches the natural scaling of the KL term.</span>

---

## <span style="font-size: 16px;">The KL Term</span>

<span style="font-size: 14px;">The KL divergence $D_{KL}(q_\phi(z|x) \| p(z))$ measures how much the encoder's approximate posterior deviates from the prior. It acts as a regularizer that prevents the encoder from assigning all its probability mass to a single point</span>

<span style="font-size: 14px;">**Closed-form for Gaussians.** Because both $q_\phi(z|x) = \mathcal{N}(\mu, \text{diag}(\sigma^2))$ and $p(z) = \mathcal{N}(0, I)$ are Gaussian, the KL divergence has an exact analytical formula. No sampling is needed. The formula decomposes across latent dimensions because the covariance is diagonal:</span>

$$
D_{KL} = -\frac{1}{2} \sum_{j=1}^{L} (1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2)
$$

<span style="font-size: 14px;">Each term in the sum has a clear interpretation. The $\mu_j^2$ term penalizes means that drift away from zero. The $\sigma_j^2$ term penalizes variances that grow too large. The $\log \sigma_j^2$ term penalizes variances that shrink too small (since $\log \sigma_j^2 \to -\infty$ as $\sigma_j^2 \to 0$). The constant $1$ ensures the KL is zero when $\mu_j = 0$ and $\sigma_j^2 = 1$, which is exactly the prior.</span>

<span style="font-size: 14px;">**Regularizes toward $\mathcal{N}(0, I)$.** The KL term reaches its minimum of zero when every latent dimension has mean zero and unit variance. Any deviation from this standard normal prior increases the loss. This encourages the encoder to spread its latent codes around the origin with roughly unit spread, creating a smooth, continuous latent space that the decoder can navigate during generation.</span>

---

## <span style="font-size: 16px;">Balancing Reconstruction and KL</span>

<span style="font-size: 14px;">The two terms pull the model in opposite directions, creating a fundamental tension in VAE training.</span>

<span style="font-size: 14px;">**Reconstruction wants expressive latent codes.** To minimize reconstruction error, the encoder should spread different inputs far apart in latent space and use very small variances (near-deterministic encoding). This lets the decoder learn a precise mapping from $z$ back to $x$.</span>

<span style="font-size: 14px;">**KL wants them near the prior.** To minimize KL, all latent distributions should collapse to $\mathcal{N}(0, I)$ regardless of the input, destroying all information about the specific input.</span>

<span style="font-size: 14px;">**The standard ELBO weights both equally** (coefficient 1.0 on each). In practice, this often favors reconstruction, because the sum over $D$ features can produce large values while the KL sum over $L$ latent dimensions (where $L \ll D$) is comparatively small.</span>

<span style="font-size: 14px;">**Beta-VAE controls the tradeoff.** Higgins et al. (2017) introduced the beta-VAE, which scales the KL term by a coefficient $\beta$:</span>

$$
\mathcal{L}_{\beta} = \mathcal{L}_{\text{recon}} + \beta \cdot D_{KL}
$$

<span style="font-size: 14px;">When $\beta > 1$, the KL penalty is stronger, pushing the latent space toward greater disentanglement at the cost of blurrier reconstructions. When $\beta < 1$, reconstruction quality improves but the latent space becomes less regular. The original VAE corresponds to $\beta = 1$.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Kingma and Welling (2014) introduced the VAE in "Auto-Encoding Variational Bayes." The central challenge was that the true posterior $p_\theta(z|x)$ is intractable because computing $p_\theta(x) = \int p_\theta(x|z)p(z)dz$ requires integrating over the entire latent space.</span>

<span style="font-size: 14px;">**ELBO as a tractable lower bound on log-evidence.** The paper showed that for any approximate posterior $q_\phi(z|x)$:</span>

$$
\log p_\theta(x) = \text{ELBO} + D_{KL}(q_\phi(z|x) \| p_\theta(z|x)) \geq \text{ELBO}
$$

<span style="font-size: 14px;">Since the KL divergence is always non-negative, the ELBO is a lower bound on $\log p_\theta(x)$. Maximizing the ELBO simultaneously increases the data log-likelihood and tightens the approximation of the posterior.</span>

<span style="font-size: 14px;">**The SGVB estimator.** The paper's key contribution was the Stochastic Gradient Variational Bayes (SGVB) estimator. By applying the reparameterization trick ($z = \mu + \sigma \odot \epsilon$ where $\epsilon \sim \mathcal{N}(0, I)$), gradients of the ELBO with respect to both $\phi$ and $\theta$ can be computed via standard backpropagation. This made it possible to train deep generative models with scalable mini-batch SGD. The paper demonstrated VAEs on MNIST and Frey Face datasets, showing smooth latent space interpolation and meaningful generation from the prior.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a small VAE with $D = 4$ input features and $L = 2$ latent dimensions. We trace through the ELBO loss for a batch of $B = 2$ samples.</span>

<span style="font-size: 14px;">**Inputs and reconstructions:**</span>

<span style="font-size: 14px;">$x^{(1)} = [1.0, 0.5, 0.8, 0.3]$, $\hat{x}^{(1)} = [0.9, 0.6, 0.7, 0.4]$</span>

<span style="font-size: 14px;">$x^{(2)} = [0.2, 0.7, 0.4, 0.9]$, $\hat{x}^{(2)} = [0.3, 0.5, 0.5, 0.8]$</span>

<span style="font-size: 14px;">**Encoder outputs:**</span>

<span style="font-size: 14px;">$\mu^{(1)} = [0.5, -0.3]$, $\text{log\_var}^{(1)} = [-0.2, 0.1]$</span>

<span style="font-size: 14px;">$\mu^{(2)} = [-0.4, 0.6]$, $\text{log\_var}^{(2)} = [0.3, -0.5]$</span>

### <span style="font-size: 14px;">Step 1: Reconstruction Loss (MSE)</span>

<span style="font-size: 14px;">**Sample 1:** $(1.0 - 0.9)^2 + (0.5 - 0.6)^2 + (0.8 - 0.7)^2 + (0.3 - 0.4)^2 = 0.01 + 0.01 + 0.01 + 0.01 = 0.04$</span>

<span style="font-size: 14px;">**Sample 2:** $(0.2 - 0.3)^2 + (0.7 - 0.5)^2 + (0.4 - 0.5)^2 + (0.9 - 0.8)^2 = 0.01 + 0.04 + 0.01 + 0.01 = 0.07$</span>

<span style="font-size: 14px;">**Batch average:** $\mathcal{L}_{\text{recon}} = \frac{0.04 + 0.07}{2} = 0.055$</span>

### <span style="font-size: 14px;">Step 2: KL Divergence</span>

<span style="font-size: 14px;">Using $D_{KL} = -0.5 \sum_j (1 + \text{log\_var}_j - \mu_j^2 - \exp(\text{log\_var}_j))$:</span>

<span style="font-size: 14px;">**Sample 1, dim 1:** $1 + (-0.2) - 0.5^2 - \exp(-0.2) = 1 - 0.2 - 0.25 - 0.8187 = -0.2687$</span>

<span style="font-size: 14px;">**Sample 1, dim 2:** $1 + 0.1 - (-0.3)^2 - \exp(0.1) = 1 + 0.1 - 0.09 - 1.1052 = -0.0952$</span>

<span style="font-size: 14px;">**Sample 1 KL:** $-0.5 \times (-0.2687 + (-0.0952)) = -0.5 \times (-0.3639) = 0.1820$</span>

<span style="font-size: 14px;">**Sample 2, dim 1:** $1 + 0.3 - (-0.4)^2 - \exp(0.3) = 1 + 0.3 - 0.16 - 1.3499 = -0.2099$</span>

<span style="font-size: 14px;">**Sample 2, dim 2:** $1 + (-0.5) - 0.6^2 - \exp(-0.5) = 1 - 0.5 - 0.36 - 0.6065 = -0.4665$</span>

<span style="font-size: 14px;">**Sample 2 KL:** $-0.5 \times (-0.2099 + (-0.4665)) = -0.5 \times (-0.6764) = 0.3382$</span>

<span style="font-size: 14px;">**Batch average:** $D_{KL} = \frac{0.1820 + 0.3382}{2} = 0.2601$</span>

### <span style="font-size: 14px;">Step 3: Total Loss</span>

$$
\mathcal{L}_{\text{total}} = 0.055 + 0.2601 = 0.3151
$$

<span style="font-size: 14px;">The function returns `{"total": 0.3151, "recon": 0.055, "kl": 0.2601}`. The KL dominates because the encoder means and variances deviate from the prior $\mathcal{N}(0, I)$. As training progresses, the KL shrinks as the encoder learns to stay closer to the prior.</span>

---

## <span style="font-size: 16px;">Common Pitfalls</span>

<span style="font-size: 14px;">**Wrong reduction (sum vs. mean over features).** The reconstruction loss should sum over features per sample, then average over the batch. If you average over features instead of summing, the reconstruction term becomes $D$ times too small relative to KL, causing the model to ignore the data (posterior collapse). The KL should likewise sum over latent dimensions per sample, then average over the batch.</span>

<span style="font-size: 14px;">**Wrong sign on the KL term.** The KL divergence is non-negative by definition. The formula $-0.5 \sum(1 + \log\sigma^2 - \mu^2 - \sigma^2)$ produces a positive value when the posterior differs from the prior. Flipping the sign makes the KL negative, encouraging the encoder to move away from the prior. If your total loss is ever less than the reconstruction loss alone, the sign is wrong.</span>

<span style="font-size: 14px;">**Forgetting to return a dict with all three values.** The problem requires returning a dictionary with keys `total`, `recon`, and `kl`. Returning only the total loss, or returning the values as a tuple instead of a dict, will fail the tests. All three values should be Python floats, not numpy arrays.</span>

<span style="font-size: 14px;">**Computing reconstruction on the wrong pair.** The MSE compares the original input $x$ with the decoder output $\hat{x}$ (`x_recon`). A subtle mistake is computing MSE between $x$ and $z$ (the latent code), or between $x$ and $\mu$ (the encoder mean). The reconstruction loss always measures the difference between what went into the encoder and what came out of the decoder.</span>

<span style="font-size: 14px;">**Using `mean` for KL across latent dims.** The KL formula requires summing over all $L$ latent dimensions per sample. Using `np.mean` instead of `np.sum` along the latent axis divides by $L$, underweighting regularization. The batch dimension should use mean, but the latent dimension must use sum.</span>
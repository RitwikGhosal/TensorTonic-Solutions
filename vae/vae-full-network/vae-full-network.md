# <span style="font-size: 20px;">Complete VAE</span>

<span style="font-size: 14px;">The Variational Autoencoder (VAE) is a generative model introduced by Kingma & Welling (2014) in "Auto-Encoding Variational Bayes." It combines a probabilistic encoder that maps data to a latent distribution, a reparameterization trick that enables gradient-based optimization through stochastic sampling, and a decoder that reconstructs data from latent samples. The complete VAE assembles these three components into a single pipeline with six weight matrices: W_mu, b_mu, W_logvar, b_logvar, W_dec, and b_dec.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">The complete VAE is a generative model that learns a continuous latent representation of data. Unlike a standard autoencoder that maps inputs to fixed latent vectors, the VAE maps each input to a probability distribution in latent space, then samples from that distribution to reconstruct the input. This stochastic bottleneck forces the model to learn a smooth, structured latent space from which new data can be generated.</span>

<span style="font-size: 14px;">The architecture has three stages wired in sequence. The **encoder** takes input $x \in \mathbb{R}^D$ and produces two vectors: a mean $\mu \in \mathbb{R}^L$ and a log-variance $\log \sigma^2 \in \mathbb{R}^L$, where $L$ is the latent dimension. The **reparameterization trick** converts these distribution parameters into a concrete latent sample $z \in \mathbb{R}^L$ using external noise $\epsilon \sim \mathcal{N}(0, I)$. The **decoder** takes $z$ and produces a reconstruction $\hat{x} \in \mathbb{R}^D$ in the original data space.</span>

<span style="font-size: 14px;">The paper describes this as a framework for "efficient approximate inference and learning with directed probabilistic graphical models." By parameterizing the approximate posterior $q_\phi(z|x)$ as a neural network and using the reparameterization trick, the entire model becomes end-to-end differentiable and trainable with standard backpropagation.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">Let $x \in \mathbb{R}^D$ be the input, $\epsilon \in \mathbb{R}^L$ be noise sampled from $\mathcal{N}(0, I)$, and $L$ be the latent dimension.</span>

<span style="font-size: 14px;">**Equation 1 -- Encode mean.** Projects the input to the mean of the approximate posterior:</span>

$$
\mu = x \cdot W_\mu + b_\mu
$$

<span style="font-size: 14px;">Here $W_\mu \in \mathbb{R}^{D \times L}$ and $b_\mu \in \mathbb{R}^L$. A linear affine transform with no activation. Each element of $\mu$ represents the center of the learned distribution for that latent dimension.</span>

<span style="font-size: 14px;">**Equation 2 -- Encode log-variance.** Projects the input to the log-variance:</span>

$$
\log \sigma^2 = x \cdot W_{\log\text{var}} + b_{\log\text{var}}
$$

<span style="font-size: 14px;">Here $W_{\log\text{var}} \in \mathbb{R}^{D \times L}$ and $b_{\log\text{var}} \in \mathbb{R}^L$. The network outputs log-variance rather than variance because log-variance is unconstrained (any real number), while variance must be positive.</span>

<span style="font-size: 14px;">**Equation 3 -- Reparameterize.** Converts distribution parameters to a concrete sample:</span>

$$
z = \mu + \sigma \odot \epsilon, \quad \text{where} \quad \sigma = \exp(0.5 \cdot \log \sigma^2)
$$

<span style="font-size: 14px;">The standard deviation is recovered via $\exp(0.5 \cdot \log \sigma^2) = \exp(\log \sigma) = \sigma$. The element-wise product $\sigma \odot \epsilon$ scales each noise dimension, then adding $\mu$ shifts the sample to the learned location. Gradients flow through $\mu$ and $\sigma$ but not through $\epsilon$.</span>

<span style="font-size: 14px;">**Equation 4 -- Decode.** Maps the latent sample back to data space:</span>

$$
\hat{x} = z \cdot W_{\text{dec}} + b_{\text{dec}}
$$

<span style="font-size: 14px;">Here $W_{\text{dec}} \in \mathbb{R}^{L \times D}$ and $b_{\text{dec}} \in \mathbb{R}^D$. A linear affine transform with no activation, mapping from the low-dimensional latent space back to the original input dimensionality.</span>

---

## <span style="font-size: 16px;">The Three Stages</span>

### <span style="font-size: 14px;">Stage 1: Encode</span>

<span style="font-size: 14px;">The encoder receives input $x$ of shape $(D,)$ and applies two parallel affine transformations to produce $\mu$ and $\log \sigma^2$, both of shape $(L,)$. These two vectors parameterize a diagonal Gaussian $\mathcal{N}(\mu, \text{diag}(\sigma^2))$ in latent space. The encoder is the recognition model $q_\phi(z|x)$ that approximates the intractable true posterior $p(z|x)$.</span>

### <span style="font-size: 14px;">Stage 2: Reparameterize</span>

<span style="font-size: 14px;">Takes $\mu$, $\log \sigma^2$, and external noise $\epsilon$ to produce $z$. First, $\sigma = \exp(0.5 \cdot \log \sigma^2)$ converts log-variance to standard deviation. Then $z = \mu + \sigma \odot \epsilon$ applies the location-scale transform. This stage is deterministic given $\epsilon$, meaning $\partial z / \partial \mu = I$ and $\partial z / \partial \sigma = \text{diag}(\epsilon)$, allowing gradients to reach the encoder weights.</span>

### <span style="font-size: 14px;">Stage 3: Decode</span>

<span style="font-size: 14px;">Takes $z$ of shape $(L,)$ and maps it to $\hat{x}$ of shape $(D,)$ via $z \cdot W_{\text{dec}} + b_{\text{dec}}$. The decoder is the generative model $p_\theta(x|z)$ defining the likelihood of data given a latent code. No activation function is applied, so the output is a linear reconstruction.</span>

---

## <span style="font-size: 16px;">The Forward Pass</span>

<span style="font-size: 14px;">The `forward` method takes two arguments: input data $x$ and noise $\epsilon$. It chains the three stages in sequence and returns a dictionary with three keys.</span>

<span style="font-size: 14px;">**Step 1.** Compute $\mu = x \cdot W_\mu + b_\mu$ and $\log \sigma^2 = x \cdot W_{\log\text{var}} + b_{\log\text{var}}$.</span>

<span style="font-size: 14px;">**Step 2.** Compute $\sigma = \exp(0.5 \cdot \log \sigma^2)$, then $z = \mu + \sigma \odot \epsilon$.</span>

<span style="font-size: 14px;">**Step 3.** Compute $\hat{x} = z \cdot W_{\text{dec}} + b_{\text{dec}}$.</span>

<span style="font-size: 14px;">**Return value.** `{"recon": x_hat, "mu": mu, "log_var": log_var}`. The reconstruction is needed for the reconstruction loss. The mean and log-variance are needed for the KL divergence term. Returning all three in one dictionary lets the loss function compute both terms without re-running the forward pass.</span>

<span style="font-size: 14px;">The noise $\epsilon$ is passed as an explicit argument rather than sampled internally. This makes the forward pass deterministic given its inputs, which is essential for reproducible testing. In a training loop, the caller samples $\epsilon \sim \mathcal{N}(0, I)$ before each forward call.</span>

---

## <span style="font-size: 16px;">The Generate Method</span>

<span style="font-size: 14px;">The `generate` method takes a latent vector $z$ directly and returns $\hat{x} = z \cdot W_{\text{dec}} + b_{\text{dec}}$. It bypasses the encoder and reparameterization entirely, running only the decoder.</span>

<span style="font-size: 14px;">To generate new data, sample $z \sim \mathcal{N}(0, I)$ from the standard normal prior and pass it to `generate`. The decoder maps this random latent vector to data space, producing a synthetic sample resembling the training distribution. The method also accepts batches: if $z$ has shape $(N, L)$, the output has shape $(N, D)$.</span>

<span style="font-size: 14px;">The separation between `forward` and `generate` reflects the two modes of a trained VAE. Forward is for training and requires both data and noise. Generate is for inference and requires only a latent vector. The decoder weights $W_{\text{dec}}$ and $b_{\text{dec}}$ are shared between both methods.</span>

---

## <span style="font-size: 16px;">Training vs Generation</span>

<span style="font-size: 14px;">**Training mode.** The full pipeline runs: $x \to \text{encode}(\mu, \log \sigma^2) \to \text{reparameterize}(z) \to \text{decode}(\hat{x})$. The loss is the negative ELBO with two terms: reconstruction loss $\|x - \hat{x}\|^2$ and KL divergence $D_{\text{KL}}(q_\phi(z|x) \| p(z))$. Reconstruction loss encourages accuracy. KL regularizes the latent distribution toward $\mathcal{N}(0, I)$, ensuring the latent space is smooth. Gradients flow from the loss through the decoder, through reparameterization (via $\mu$ and $\sigma$, not $\epsilon$), and into encoder weights.</span>

<span style="font-size: 14px;">**Generation mode.** Only the decoder runs: $z \sim \mathcal{N}(0, I) \to \text{decode}(\hat{x})$. No encoder is needed. The KL regularization during training ensures random samples from $\mathcal{N}(0, I)$ land in latent regions the decoder maps to realistic outputs. Without KL, the encoder might map training data to arbitrary latent regions, and prior samples would produce meaningless outputs.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Kingma & Welling (2014) presented the Auto-Encoding Variational Bayes (AEVB) algorithm. The core problem was efficient inference in directed probabilistic models with continuous latent variables where the true posterior $p(z|x)$ is intractable.</span>

<span style="font-size: 14px;">The key contribution was amortized variational inference: instead of optimizing variational parameters per data point (as in traditional VI), a single encoder network $q_\phi(z|x)$ learns to map any input to its approximate posterior in one forward pass. This makes inference cost constant per data point.</span>

<span style="font-size: 14px;">The reparameterization trick was the technical innovation enabling this. By expressing $z$ as a deterministic function of $\mu$, $\sigma$, and external noise $\epsilon$, gradients of the expected reconstruction loss could be computed via standard backpropagation, avoiding high-variance score function estimators (REINFORCE).</span>

<span style="font-size: 14px;">The generative model framework positions the VAE within latent variable models. The prior $p(z) = \mathcal{N}(0, I)$ defines the latent distribution. The likelihood $p_\theta(x|z)$ (decoder) defines how latent codes generate data. AEVB jointly optimizes generative parameters $\theta$ and inference parameters $\phi$ by maximizing the ELBO on marginal log-likelihood $\log p(x)$.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a VAE with input dimension $D = 3$ and latent dimension $L = 2$. The six weight matrices:</span>

$$
W_\mu = \begin{pmatrix} 0.2 & -0.1 \\ 0.3 & 0.4 \\ -0.2 & 0.1 \end{pmatrix}, \quad b_\mu = \begin{pmatrix} 0.1 \\ -0.1 \end{pmatrix}
$$

$$
W_{\log\text{var}} = \begin{pmatrix} 0.1 & 0.3 \\ -0.2 & 0.1 \\ 0.4 & -0.3 \end{pmatrix}, \quad b_{\log\text{var}} = \begin{pmatrix} -0.5 \\ -0.5 \end{pmatrix}
$$

$$
W_{\text{dec}} = \begin{pmatrix} 0.3 & -0.2 & 0.1 \\ 0.2 & 0.4 & -0.3 \end{pmatrix}, \quad b_{\text{dec}} = \begin{pmatrix} 0.0 \\ 0.1 \\ -0.1 \end{pmatrix}
$$

<span style="font-size: 14px;">Input and noise:</span>

$$
x = \begin{pmatrix} 1.0 \\ -0.5 \\ 0.3 \end{pmatrix}, \quad \epsilon = \begin{pmatrix} 0.5 \\ -0.8 \end{pmatrix}
$$

<span style="font-size: 14px;">**Step 1: Encode mean.** $\mu = x \cdot W_\mu + b_\mu$:</span>

<span style="font-size: 14px;">$\mu[0] = 1.0(0.2) + (-0.5)(0.3) + 0.3(-0.2) + 0.1 = 0.20 - 0.15 - 0.06 + 0.10 = 0.09$</span>

<span style="font-size: 14px;">$\mu[1] = 1.0(-0.1) + (-0.5)(0.4) + 0.3(0.1) - 0.1 = -0.10 - 0.20 + 0.03 - 0.10 = -0.37$</span>

<span style="font-size: 14px;">**Step 2: Encode log-variance.** $\log \sigma^2 = x \cdot W_{\log\text{var}} + b_{\log\text{var}}$:</span>

<span style="font-size: 14px;">$\log\sigma^2[0] = 1.0(0.1) + (-0.5)(-0.2) + 0.3(0.4) - 0.5 = 0.10 + 0.10 + 0.12 - 0.50 = -0.18$</span>

<span style="font-size: 14px;">$\log\sigma^2[1] = 1.0(0.3) + (-0.5)(0.1) + 0.3(-0.3) - 0.5 = 0.30 - 0.05 - 0.09 - 0.50 = -0.34$</span>

<span style="font-size: 14px;">**Step 3: Reparameterize.** $\sigma = \exp(0.5 \cdot \log \sigma^2)$:</span>

<span style="font-size: 14px;">$\sigma[0] = \exp(-0.09) = 0.9139, \quad \sigma[1] = \exp(-0.17) = 0.8437$</span>

<span style="font-size: 14px;">$z = \mu + \sigma \odot \epsilon = [0.09 + 0.9139 \times 0.5, \; -0.37 + 0.8437 \times (-0.8)] = [0.5470, -1.0450]$</span>

<span style="font-size: 14px;">**Step 4: Decode.** $\hat{x} = z \cdot W_{\text{dec}} + b_{\text{dec}}$:</span>

<span style="font-size: 14px;">$\hat{x}[0] = 0.5470(0.3) + (-1.0450)(0.2) + 0.0 = 0.1641 - 0.2090 = -0.0449$</span>

<span style="font-size: 14px;">$\hat{x}[1] = 0.5470(-0.2) + (-1.0450)(0.4) + 0.1 = -0.1094 - 0.4180 + 0.1 = -0.4274$</span>

<span style="font-size: 14px;">$\hat{x}[2] = 0.5470(0.1) + (-1.0450)(-0.3) - 0.1 = 0.0547 + 0.3135 - 0.1 = 0.2682$</span>

<span style="font-size: 14px;">**Result.** `{"recon": [-0.0449, -0.4274, 0.2682], "mu": [0.09, -0.37], "log_var": [-0.18, -0.34]}`</span>

<span style="font-size: 14px;">**Generate example.** Calling `generate(z=[0.5, -1.0])`: $\hat{x} = [0.5(0.3)+(-1.0)(0.2), \; 0.5(-0.2)+(-1.0)(0.4)+0.1, \; 0.5(0.1)+(-1.0)(-0.3)-0.1] = [-0.05, -0.40, 0.25]$.</span>

---

## <span style="font-size: 16px;">The VAE as Generative Model</span>

<span style="font-size: 14px;">After training, the VAE generates new data through `generate`: sample $z \sim \mathcal{N}(0, I)$, then decode to get $\hat{x} = z \cdot W_{\text{dec}} + b_{\text{dec}}$. Each random $z$ produces a different synthetic data point.</span>

<span style="font-size: 14px;">Generation quality depends on how well training aligned the encoder's output distribution with the prior. The KL term pushes $q_\phi(z|x)$ toward $\mathcal{N}(0, I)$ for every training input. When this succeeds, the aggregate posterior $q_\phi(z) = \frac{1}{N}\sum_i q_\phi(z|x_i)$ approximates the prior, so random prior samples produce outputs resembling the training distribution.</span>

<span style="font-size: 14px;">The latent space also supports interpolation. Given two data points $x_1$ and $x_2$, encode both to get $\mu_1$ and $\mu_2$, then decode points along $\alpha \mu_1 + (1 - \alpha) \mu_2$ for $\alpha \in [0, 1]$. KL regularization ensures these intermediate points decode to plausible data, creating smooth transitions.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting reparameterization in forward.** Passing $\mu$ directly to the decoder instead of computing $z = \mu + \sigma \odot \epsilon$ collapses the VAE to a deterministic autoencoder. Without stochastic sampling, there is no KL divergence to regularize, and the latent space will not support generation from the prior.</span>

* <span style="font-size: 14px;">**Using wrong epsilon shape.** The noise $\epsilon$ must have shape $(L,)$ matching the latent dimension, not $(D,)$ matching the input dimension. If $D \neq L$, the element-wise product $\sigma \odot \epsilon$ will broadcast incorrectly or raise a dimension error.</span>

* <span style="font-size: 14px;">**Returning wrong dictionary keys.** The forward method must return exactly `{"recon": ..., "mu": ..., "log_var": ...}`. Using names like "reconstruction", "mean", or "logvar" breaks downstream loss computation. Note `log_var` uses an underscore, not camelCase.</span>

* <span style="font-size: 14px;">**Not separating forward from generate.** The `generate` method must use only the decoder. Calling `forward` inside `generate` would require $x$ and $\epsilon$ as inputs, defeating the purpose of generation from the prior where no input data exists.</span>

* <span style="font-size: 14px;">**Applying activation functions when not specified.** This implementation uses no activations in encoder or decoder. Adding sigmoid to decoder output or ReLU after encoder projections changes model behavior. The problem specifies linear affine transforms only.</span>

* <span style="font-size: 14px;">**Computing variance instead of standard deviation.** The reparameterization uses $\sigma = \exp(0.5 \cdot \log \sigma^2)$, not $\sigma^2 = \exp(\log \sigma^2)$. Multiplying $\epsilon$ by variance instead of standard deviation scales the noise incorrectly.</span>

---
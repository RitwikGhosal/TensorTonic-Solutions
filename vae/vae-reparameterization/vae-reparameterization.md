# <span style="font-size: 20px;">Reparameterization Trick</span>

<span style="font-size: 14px;">The reparameterization trick (Kingma & Welling, 2014) is a technique that enables backpropagation through stochastic sampling in variational autoencoders (VAEs). By expressing a random variable as a deterministic function of its parameters plus external noise, it transforms a non-differentiable sampling operation into a differentiable computation, making gradient-based optimization of the VAE objective possible.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">In a VAE, the encoder maps an input $x$ to a distribution over latent variables, parameterized by a mean vector $\mu$ and a log-variance vector $\log \sigma^2$. To generate a latent code $z$, the model must sample from this distribution: $z \sim \mathcal{N}(\mu, \sigma^2 I)$. The problem is that sampling is a stochastic operation with no well-defined gradient. Backpropagation cannot flow through a random number generator.</span>

<span style="font-size: 14px;">The reparameterization trick resolves this by separating the randomness from the learned parameters. Instead of sampling $z$ directly from $\mathcal{N}(\mu, \sigma^2 I)$, the trick first samples noise from a fixed distribution $\epsilon \sim \mathcal{N}(0, I)$, then constructs $z$ as a deterministic function of $\mu$, $\sigma$, and $\epsilon$. This makes $z$ differentiable with respect to $\mu$ and $\sigma$, allowing standard backpropagation to optimize the encoder.</span>

<span style="font-size: 14px;">Kingma & Welling state: "We reparameterize $\tilde{z}$ as a deterministic variable $\tilde{z} = g_\phi(\epsilon, x)$ where $\epsilon$ is an auxiliary noise variable." The function $g$ encodes the transformation from fixed noise to the desired distribution, and because $g$ is differentiable, gradients flow through it to the encoder parameters $\phi$.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">The encoder outputs two vectors for each input $x$:</span>

$$
\mu = f_\mu(x), \quad \log \sigma^2 = f_{\log \sigma^2}(x)
$$

<span style="font-size: 14px;">where $f_\mu$ and $f_{\log \sigma^2}$ are neural networks (typically sharing earlier layers). The notation $\log \sigma^2$ is often written as $\texttt{log\_var}$ in code.</span>

<span style="font-size: 14px;">The standard deviation $\sigma$ is recovered from $\log \sigma^2$ via:</span>

$$
\sigma = \exp\!\left(\tfrac{1}{2} \log \sigma^2\right)
$$

<span style="font-size: 14px;">The reparameterized sample is then:</span>

$$
z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

<span style="font-size: 14px;">where $\odot$ denotes element-wise multiplication. Each component $z_i = \mu_i + \sigma_i \cdot \epsilon_i$ is independently computed. The noise $\epsilon$ has the same shape as $\mu$ and $\sigma$.</span>

---

## <span style="font-size: 16px;">The Problem: Sampling Is Non-Differentiable</span>

<span style="font-size: 14px;">Training a VAE requires optimizing the Evidence Lower Bound (ELBO), which involves computing expectations over the latent distribution $q_\phi(z|x) = \mathcal{N}(\mu, \sigma^2 I)$. The standard approach is to estimate these expectations with Monte Carlo samples: draw $z \sim q_\phi(z|x)$, evaluate the reconstruction loss and KL divergence, and backpropagate through the entire computation graph.</span>

<span style="font-size: 14px;">The bottleneck is the sampling step $z \sim \mathcal{N}(\mu, \sigma^2 I)$. This operation takes $\mu$ and $\sigma$ as inputs and produces a random output $z$, but it has no gradient. Consider what $\partial z / \partial \mu$ would mean: changing $\mu$ shifts the distribution from which $z$ is drawn, but any specific sample $z$ is a random realization, not a smooth function of $\mu$. The sampling operation is a black box that breaks the computation graph.</span>

<span style="font-size: 14px;">Without the reparameterization trick, the only alternative is REINFORCE-style gradient estimators (score function estimators), which compute $\nabla_\phi \mathbb{E}_{q_\phi}[f(z)] = \mathbb{E}_{q_\phi}[f(z) \nabla_\phi \log q_\phi(z|x)]$. These are valid but suffer from extremely high variance, making training slow and unstable.</span>

---

## <span style="font-size: 16px;">The Solution: Move Randomness to an External Variable</span>

<span style="font-size: 14px;">The core insight is to factor the sampling process into two parts: a fixed source of randomness and a deterministic transformation:</span>

$$
\epsilon \sim \mathcal{N}(0, I), \quad z = \mu + \sigma \odot \epsilon
$$

<span style="font-size: 14px;">The random number $\epsilon$ is sampled once at the beginning and then treated as a fixed constant throughout the forward and backward pass. The node that computes $z$ is a deterministic function of three inputs: $\mu$ (a learned parameter), $\sigma$ (derived from a learned parameter), and $\epsilon$ (a constant). Because $z = \mu + \sigma \odot \epsilon$ is just addition and multiplication, it is fully differentiable with respect to $\mu$ and $\sigma$.</span>

<span style="font-size: 14px;">The stochasticity has not disappeared. Each training iteration still uses a different $\epsilon$, so $z$ is still a random variable across the training procedure. But within a single forward-backward pass, $\epsilon$ is fixed, and $z$ is deterministic. The randomness is external to the computation graph rather than internal to it.</span>

<span style="font-size: 14px;">Kingma & Welling call this the Stochastic Gradient Variational Bayes (SGVB) estimator. The paper shows that even a single Monte Carlo sample ($L = 1$) per datapoint per update is sufficient for practical training, because the reparameterized gradient estimator has low enough variance.</span>

---

## <span style="font-size: 16px;">Why This Works: Gradient Analysis</span>

<span style="font-size: 14px;">The reparameterized form $z = \mu + \sigma \odot \epsilon$ yields clean, well-defined gradients with respect to both encoder outputs.</span>

<span style="font-size: 14px;">**Gradient with respect to** $\mu$**:**</span>

$$
\frac{\partial z}{\partial \mu} = I
$$

<span style="font-size: 14px;">The identity. Shifting $\mu$ by $\Delta \mu$ shifts every component of $z$ by the same amount, regardless of the noise. Gradients from the reconstruction loss flow directly back to $\mu$ without distortion.</span>

<span style="font-size: 14px;">**Gradient with respect to** $\sigma$**:**</span>

$$
\frac{\partial z}{\partial \sigma} = \epsilon
$$

<span style="font-size: 14px;">The gradient is exactly the noise vector. When $\epsilon_i$ is large, the gradient with respect to $\sigma_i$ is large, because stretching the distribution has a bigger effect on that component. When $\epsilon_i$ is near zero, the gradient is small. This is geometrically intuitive: the sensitivity of $z$ to changes in $\sigma$ is proportional to how far the noise pushed the sample from the mean.</span>

<span style="font-size: 14px;">The chain rule continues through $\sigma$ to $\log \sigma^2$. Since $\sigma = \exp(0.5 \cdot \log \sigma^2)$:</span>

$$
\frac{\partial z}{\partial \log \sigma^2} = \epsilon \odot \left(0.5 \cdot \sigma\right) = 0.5 \cdot \sigma \odot \epsilon
$$

<span style="font-size: 14px;">This is the gradient that the encoder's $\log \sigma^2$ head receives. The factor of $0.5$ comes from the square root in the variance-to-standard-deviation conversion.</span>

---

## The $\exp(0.5 \cdot \log \sigma^2)$ Conversion

<span style="font-size: 14px;">A common source of confusion is why the encoder outputs $\log \sigma^2$ (log-variance) rather than $\sigma$ directly, and what the coefficient $0.5$ does.</span>

<span style="font-size: 14px;">**Why output log-variance instead of sigma?** The standard deviation $\sigma$ must be strictly positive. If the encoder output $\sigma$ directly, a constraint or activation function (like softplus) would be needed. By outputting $\log \sigma^2$, the encoder can produce any real number on $(-\infty, +\infty)$, which is the natural output range of a linear layer. Positivity is automatically satisfied by the exponential during conversion.</span>

<span style="font-size: 14px;">**Why log-variance and not log-sigma?** Either works mathematically. The log-variance convention aligns naturally with the KL divergence formula:</span>

$$
D_{KL} = -\frac{1}{2} \sum_{j=1}^{d} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)
$$

<span style="font-size: 14px;">Here $\log \sigma_j^2$ appears directly, so using log-variance as the encoder output avoids an extra log operation. In code, $\texttt{log\_var}$ is used as-is in the KL term.</span>

<span style="font-size: 14px;">**The 0.5 coefficient derivation:**</span>

$$
\sigma = \sqrt{\sigma^2} = \sqrt{\exp(\log \sigma^2)} = \exp\!\left(\tfrac{1}{2} \log \sigma^2\right)
$$

<span style="font-size: 14px;">The $0.5$ is the exponent from the square root. In code: $\texttt{sigma = torch.exp(0.5 * log\_var)}$. The factor is not arbitrary -- it is the mathematical consequence of converting from variance to standard deviation in log-space.</span>

---

## <span style="font-size: 16px;">Paper Context: Kingma & Welling (2014)</span>

<span style="font-size: 14px;">The reparameterization trick is the central technical contribution of "Auto-Encoding Variational Bayes" (Kingma & Welling, 2014). The paper introduces the VAE framework, combining a probabilistic encoder $q_\phi(z|x)$ (recognition model) with a probabilistic decoder $p_\theta(x|z)$ (generative model). The training objective is to maximize the ELBO:</span>

$$
\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))
$$

<span style="font-size: 14px;">The first term is reconstruction quality. The second regularizes the approximate posterior toward the prior $p(z) = \mathcal{N}(0, I)$. For Gaussian $q_\phi$, the KL term has a closed-form solution. The reconstruction term requires sampling from $q_\phi(z|x)$, and optimizing it with respect to $\phi$ is where the reparameterization trick is essential.</span>

<span style="font-size: 14px;">Before this paper, variational inference in neural networks was impractical for high-dimensional latent spaces because REINFORCE-style gradient estimators had variance too high for stable training. The reparameterization trick produces gradient estimates with variance low enough that a single sample per datapoint suffices. Kingma & Welling demonstrated this on MNIST and Frey Face datasets.</span>

<span style="font-size: 14px;">The paper also notes that the trick is not limited to Gaussian distributions. It applies to any distribution expressible as a differentiable transformation of fixed noise -- any location-scale family. For distributions where no such reparameterization exists (e.g., discrete distributions), alternatives like the Gumbel-Softmax trick were later developed (Jang et al., 2017; Maddison et al., 2017).</span>

---

## Numerical Example ($d = 3$)

<span style="font-size: 14px;">Consider a latent dimension of $d = 3$. The encoder has produced:</span>

$$
\mu = [1.0, -0.5, 0.3], \quad \log \sigma^2 = [0.0, -1.0, 0.6]
$$

<span style="font-size: 14px;">**Step 1 -- Convert log-variance to standard deviation:**</span>

$$
\sigma = \exp(0.5 \cdot \log \sigma^2) = \exp([0.0, -0.5, 0.3]) = [1.0, 0.6065, 1.3499]
$$

<span style="font-size: 14px;">Element by element:</span>

* <span style="font-size: 14px;">$\sigma_1 = \exp(0.5 \cdot 0.0) = \exp(0) = 1.0$</span>
* <span style="font-size: 14px;">$\sigma_2 = \exp(0.5 \cdot (-1.0)) = \exp(-0.5) = 0.6065$</span>
* <span style="font-size: 14px;">$\sigma_3 = \exp(0.5 \cdot 0.6) = \exp(0.3) = 1.3499$</span>

<span style="font-size: 14px;">**Step 2 -- Sample noise from the standard normal:**</span>

$$
\epsilon = [0.5, -1.2, 0.8]
$$

<span style="font-size: 14px;">Each component is an independent draw from $\mathcal{N}(0, 1)$.</span>

<span style="font-size: 14px;">**Step 3 -- Compute the reparameterized sample:**</span>

$$
z = \mu + \sigma \odot \epsilon = [1.0, -0.5, 0.3] + [1.0, 0.6065, 1.3499] \odot [0.5, -1.2, 0.8]
$$

* <span style="font-size: 14px;">$z_1 = 1.0 + 1.0 \times 0.5 = 1.5$</span>
* <span style="font-size: 14px;">$z_2 = -0.5 + 0.6065 \times (-1.2) = -0.5 + (-0.7278) = -1.2278$</span>
* <span style="font-size: 14px;">$z_3 = 0.3 + 1.3499 \times 0.8 = 0.3 + 1.0799 = 1.3799$</span>

$$
z = [1.5, -1.2278, 1.3799]
$$

<span style="font-size: 14px;">**Step 4 -- Verify gradients:**</span>

* <span style="font-size: 14px;">$\partial z / \partial \mu = [1, 1, 1]$ (identity, each $z_i$ shifts one-to-one with $\mu_i$)</span>
* <span style="font-size: 14px;">$\partial z / \partial \sigma = \epsilon = [0.5, -1.2, 0.8]$</span>
* <span style="font-size: 14px;">$\partial z / \partial \log \sigma^2 = 0.5 \cdot \sigma \odot \epsilon = [0.25, -0.3639, 0.5400]$</span>

<span style="font-size: 14px;">All gradients are concrete numbers. No stochastic estimation was needed to compute them.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Using log-variance directly as sigma.** Writing $z = \mu + \texttt{log\_var} \cdot \epsilon$ instead of $z = \mu + \exp(0.5 \cdot \texttt{log\_var}) \cdot \epsilon$. Since $\texttt{log\_var}$ can be negative, this produces $z$ values that collapse toward $\mu$, effectively killing the stochasticity. The latent space will be degenerate and reconstructions blurry.</span>

* <span style="font-size: 14px;">**Wrong coefficient in the exponent.** Writing $\exp(\texttt{log\_var})$ instead of $\exp(0.5 \cdot \texttt{log\_var})$ computes $\sigma^2$ (variance) instead of $\sigma$ (standard deviation). The reparameterized sample becomes $z = \mu + \sigma^2 \cdot \epsilon$, which overestimates the spread when $\sigma > 1$ and underestimates it when $\sigma < 1$, distorting the latent space geometry.</span>

* <span style="font-size: 14px;">**Forgetting element-wise multiplication.** The operation $\sigma \odot \epsilon$ is element-wise, not a dot product or matrix multiplication. Each latent dimension $z_i$ depends only on its own $\sigma_i$ and $\epsilon_i$. Using a dot product collapses the latent vector into a scalar. In code, this should be a simple $\texttt{*}$ operator between tensors of the same shape.</span>

* <span style="font-size: 14px;">**Generating epsilon with the wrong shape.** The noise $\epsilon$ must match the shape of $\mu$ and $\log \sigma^2$. If the encoder outputs shape $(B, d)$, then $\epsilon$ must also be $(B, d)$. Sampling $\epsilon$ of shape $(d,)$ and broadcasting across the batch gives every sample the same noise, defeating the purpose of stochastic sampling.</span>

* <span style="font-size: 14px;">**Sampling epsilon once and reusing it.** The noise must be freshly sampled at every forward pass. Reusing the same $\epsilon$ across iterations makes $z$ a fixed deterministic function, eliminating the Monte Carlo estimation the ELBO requires. The model would converge to a deterministic bottleneck autoencoder, not a VAE.</span>

* <span style="font-size: 14px;">**Confusing log-variance with log-standard-deviation.** Some implementations output $\log \sigma$ instead of $\log \sigma^2$. In that case, the conversion is $\sigma = \exp(\log \sigma)$ with no factor of $0.5$. Mixing conventions -- outputting $\log \sigma$ but applying $\exp(0.5 \cdot \texttt{output})$ -- produces $\sigma^{1/2}$ instead of $\sigma$, compressing the latent distribution. Always verify which convention the encoder uses.</span>

---
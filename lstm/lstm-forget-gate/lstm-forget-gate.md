# <span style="font-size: 20px;">Forget Gate</span>

<span style="font-size: 14px;">The forget gate is the gating mechanism inside an LSTM cell that decides, for each dimension of the cell state, how much of the previous value to retain versus discard. Introduced by Gers, Schmidhuber, and Cummins (2000) as an extension to the original LSTM architecture of Hochreiter and Schmidhuber (1997), the forget gate is now a standard component in every modern LSTM implementation.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">The forget gate is a learned, element-wise filter applied to the cell state $c_{t-1}$ before the cell state update at time step $t$. It produces a vector $f_t \in \mathbb{R}^H$ where each element lies in $[0, 1]$. A value of $0$ means "completely forget this dimension of the cell state," while a value of $1$ means "keep this dimension unchanged." Values between $0$ and $1$ represent partial retention.</span>

<span style="font-size: 14px;">The gate examines two inputs: the previous hidden state $h_{t-1}$ and the current input $x_t$. These are concatenated, multiplied by a learned weight matrix $W_f$, shifted by a bias $b_f$, and passed through sigmoid. The result is a per-dimension retention mask applied to the cell state via element-wise multiplication.</span>

<span style="font-size: 14px;">Within the full LSTM cell, the forget gate is the first operation that touches the cell state at each time step. The complete cell state update is $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$, where the forget gate controls the first term (what to keep from the past) and the input gate $i_t$ with the candidate $\tilde{c}_t$ control the second term (what new information to write). The forget gate and input gate operate independently: the cell can simultaneously forget old content in some dimensions and write new content in others.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">Let $x_t \in \mathbb{R}^D$ be the input at time step $t$ and $h_{t-1} \in \mathbb{R}^H$ be the previous hidden state. The forget gate computation proceeds in three stages.</span>

<span style="font-size: 14px;">**Stage 1 -- Concatenation.** The previous hidden state and current input are stacked into a single vector:</span>

$$
[h_{t-1}, x_t] \in \mathbb{R}^{H+D}
$$

<span style="font-size: 14px;">This concatenation places $h_{t-1}$ first, followed by $x_t$. The ordering must be consistent with how $W_f$ is defined: the first $H$ columns of $W_f$ multiply $h_{t-1}$, the last $D$ columns multiply $x_t$.</span>

<span style="font-size: 14px;">**Stage 2 -- Affine transformation.** The concatenated vector is transformed by a weight matrix and bias:</span>

$$
z_f = W_f \cdot [h_{t-1}, x_t] + b_f
$$

<span style="font-size: 14px;">Here $W_f \in \mathbb{R}^{H \times (H+D)}$ and $b_f \in \mathbb{R}^H$. The matrix-vector product produces $z_f \in \mathbb{R}^H$, a vector of pre-activation logits. Each element $z_f[i]$ is a weighted combination of all elements from $h_{t-1}$ and $x_t$, plus a bias term.</span>

<span style="font-size: 14px;">**Stage 3 -- Sigmoid activation.** The logits are squashed to $[0, 1]$:</span>

$$
f_t = \sigma(z_f) = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

<span style="font-size: 14px;">The sigmoid $\sigma(x) = \frac{1}{1 + e^{-x}}$ maps each element independently to $(0, 1)$. This gives the forget gate its name: each output dimension independently "decides" what fraction of the corresponding cell state dimension to forget.</span>

<span style="font-size: 14px;">The forget gate output $f_t$ is then applied to the cell state via element-wise multiplication:</span>

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

<span style="font-size: 14px;">The $f_t \odot c_{t-1}$ term is where the forget gate does its work. If $f_t[i] = 0.1$, then $c_t[i]$ retains only $10\%$ of its previous value in dimension $i$ before the input gate writes new content.</span>

---

## <span style="font-size: 16px;">Why a Forget Gate</span>

<span style="font-size: 14px;">The original LSTM proposed by Hochreiter and Schmidhuber in 1997 did not include a forget gate. The cell state update in the original formulation was purely additive: $c_t = c_{t-1} + i_t \odot \tilde{c}_t$. The input gate controlled what new information entered the cell state, but there was no mechanism to remove old information. Once a value was written into the cell state, it persisted indefinitely.</span>

<span style="font-size: 14px;">This design created a fundamental problem. In tasks where context changes -- such as language modeling, where the topic of a sentence shifts, or time series prediction, where the underlying regime changes -- the cell state accumulated stale information. Without a way to reset or decay old values, the cell state could grow without bound, leading to numerical instability (saturated activations, exploding gradients) and an inability to adapt to new contexts.</span>

<span style="font-size: 14px;">Gers, Schmidhuber, and Cummins addressed this in their 2000 paper "Learning to Forget: Continual Prediction with LSTM." They observed that "the forget gate learns to reset memory cells when their contents are out of date." The addition was simple -- multiply the cell state by a learned gating vector before adding new content -- but it fundamentally changed the LSTM's behavior. The cell state update went from $c_t = c_{t-1} + i_t \odot \tilde{c}_t$ to $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$.</span>

<span style="font-size: 14px;">The forget gate proved so effective that it became permanent. Every modern LSTM implementation -- PyTorch, TensorFlow, JAX -- includes the forget gate by default. The "LSTM" as understood today always refers to the version with the forget gate.</span>

---

## <span style="font-size: 16px;">The Sigmoid as a Soft Switch</span>

<span style="font-size: 14px;">The choice of sigmoid for the forget gate activation is deliberate and serves multiple purposes.</span>

<span style="font-size: 14px;">**Smooth [0, 1] range.** The sigmoid maps any real-valued input to the open interval $(0, 1)$. This makes it a natural choice for a gate that needs to represent a "fraction to keep." A hard threshold (step function) would produce only binary decisions -- fully keep or fully forget -- with no gradient flowing through in either case. The sigmoid provides a smooth, differentiable transition that allows the network to learn nuanced retention levels through gradient descent.</span>

<span style="font-size: 14px;">**Per-dimension independence.** The sigmoid is applied element-wise. Each of the $H$ dimensions of the forget gate output is computed independently. Dimension 3 might output $f_t[3] = 0.95$ (keep almost everything) while dimension 7 outputs $f_t[7] = 0.02$ (nearly forget everything), all within the same time step. Different dimensions of the cell state can encode different types of information with different persistence requirements.</span>

<span style="font-size: 14px;">**Gradient properties.** The sigmoid derivative is $\sigma'(x) = \sigma(x)(1 - \sigma(x))$, with a maximum of $0.25$ at $x = 0$. When the gate is saturated (near $0$ or $1$), the gradient approaches zero. This means the network can "commit" to keeping or forgetting a dimension -- once saturated, the gate stops changing, stabilizing training. It also means the gate can get stuck if initialized in a saturated regime, which motivates careful bias initialization.</span>

<span style="font-size: 14px;">**Comparison with tanh.** The candidate cell state uses tanh, which maps to $[-1, 1]$, because cell state values need to be both positive and negative. The forget gate uses sigmoid because retention fractions are inherently non-negative -- it makes no sense to "retain negative $30\%$" of a cell state dimension.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Hochreiter and Schmidhuber published "Long Short-Term Memory" in 1997, introducing the LSTM architecture to solve the vanishing gradient problem in recurrent neural networks. The original architecture had input gates and output gates but no forget gate. The cell state was a "constant error carousel" -- information entered through the input gate and persisted forever, protected from gradient decay by the gating mechanism.</span>

<span style="font-size: 14px;">The absence of a forget gate was intentional. Hochreiter and Schmidhuber prioritized preventing gradient vanishing, and an additive cell state update ensures that gradients flow through the cell state without attenuation. The cost was that the cell could never unlearn: once a pattern was written into the cell state, it stayed there.</span>

<span style="font-size: 14px;">Three years later, Gers, Schmidhuber, and Cummins (2000) published "Learning to Forget: Continual Prediction with LSTM," introducing the forget gate. They demonstrated that on continual prediction tasks -- where the model must track a changing environment -- the forget gate significantly improved performance by enabling the LSTM to reset its internal state when context changed.</span>

<span style="font-size: 14px;">Greff et al. (2017) conducted a large-scale study of LSTM variants ("LSTM: A Search Space Odyssey") and found that the forget gate was the single most important component. Removing the forget gate caused the largest performance degradation across all tasks tested, more than removing any other single gate. This confirmed the practical importance of the ability to selectively erase memory.</span>

<span style="font-size: 14px;">The forget gate also introduced a critical implementation detail: **bias initialization**. Jozefowicz, Zaremba, and Sutskever (2015) showed that initializing the forget gate bias to a large positive value (typically $1.0$ or $2.0$) is essential. This ensures the forget gate starts near $1$ (keep everything), preventing the network from forgetting information before it has learned what is worth remembering. PyTorch follows this practice by default.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a minimal LSTM with hidden size $H = 2$ and input size $D = 2$. At time step $t$, the inputs are:</span>

$$
h_{t-1} = \begin{pmatrix} 0.4 \\ -0.1 \end{pmatrix}, \quad x_t = \begin{pmatrix} 0.7 \\ 0.2 \end{pmatrix}
$$

<span style="font-size: 14px;">The forget gate weight matrix and bias are:</span>

$$
W_f = \begin{pmatrix} 0.5 & -0.3 & 0.2 & 0.1 \\ -0.1 & 0.4 & 0.3 & -0.2 \end{pmatrix}, \quad b_f = \begin{pmatrix} 1.0 \\ 1.0 \end{pmatrix}
$$

<span style="font-size: 14px;">Note the bias is initialized to $1.0$, following the Jozefowicz et al. (2015) recommendation.</span>

<span style="font-size: 14px;">**Step 1: Concatenation.** Stack $h_{t-1}$ and $x_t$:</span>

$$
[h_{t-1}, x_t] = [0.4, -0.1, 0.7, 0.2]
$$

<span style="font-size: 14px;">**Step 2: Linear transformation.** Compute $W_f \cdot [h_{t-1}, x_t] + b_f$:</span>

<span style="font-size: 14px;">Row 1: $0.5(0.4) + (-0.3)(-0.1) + 0.2(0.7) + 0.1(0.2) + 1.0 = 0.20 + 0.03 + 0.14 + 0.02 + 1.0 = 1.39$</span>

<span style="font-size: 14px;">Row 2: $(-0.1)(0.4) + 0.4(-0.1) + 0.3(0.7) + (-0.2)(0.2) + 1.0 = -0.04 - 0.04 + 0.21 - 0.04 + 1.0 = 1.09$</span>

$$
z_f = \begin{pmatrix} 1.39 \\ 1.09 \end{pmatrix}
$$

<span style="font-size: 14px;">**Step 3: Sigmoid.** Apply $\sigma$ element-wise:</span>

<span style="font-size: 14px;">$f_t[0] = \sigma(1.39) = \frac{1}{1 + e^{-1.39}} = \frac{1}{1 + 0.2491} = \frac{1}{1.2491} = 0.8006$</span>

<span style="font-size: 14px;">$f_t[1] = \sigma(1.09) = \frac{1}{1 + e^{-1.09}} = \frac{1}{1 + 0.3362} = \frac{1}{1.3362} = 0.7484$</span>

$$
f_t = \begin{pmatrix} 0.8006 \\ 0.7484 \end{pmatrix}
$$

<span style="font-size: 14px;">**Interpretation.** Both values are well above $0.5$, which is expected given the positive bias initialization. Dimension 0 retains about $80\%$ of its previous cell state, dimension 1 retains about $75\%$. If the previous cell state was $c_{t-1} = [1.5, -0.8]$, then after the forget gate the retained portion would be $f_t \odot c_{t-1} = [0.8006 \times 1.5, \, 0.7484 \times (-0.8)] = [1.2009, -0.5987]$. The input gate and candidate would then add new content on top of this retained base.</span>

---

## <span style="font-size: 16px;">Connection to GRU</span>

<span style="font-size: 14px;">The Gated Recurrent Unit (Cho et al., 2014) takes a different approach to the forget-or-keep decision. Instead of having separate forget and input gates, the GRU uses a single **update gate** $z_t$ that controls both operations simultaneously:</span>

$$
h_t = z_t \odot h_{t-1} + (1 - z_t) \odot \tilde{h}_t
$$

<span style="font-size: 14px;">When $z_t = 1$ in the GRU, the old hidden state is fully retained and no new content is written. This is analogous to $f_t = 1$ and $i_t = 0$ in the LSTM. When $z_t = 0$, the old state is fully replaced by the candidate, analogous to $f_t = 0$ and $i_t = 1$.</span>

<span style="font-size: 14px;">The critical difference is that the LSTM's forget gate and input gate are **independent**. The LSTM can have $f_t = 1$ and $i_t = 1$ simultaneously (keep everything and also add new content), or $f_t = 0$ and $i_t = 0$ (forget everything and add nothing). The GRU constrains these to sum to $1$: the more it keeps, the less it can write, and vice versa. This constraint reduces parameters but limits expressiveness. Additionally, the GRU lacks a separate cell state -- the LSTM's forget gate operates on $c_t$ (internal memory shielded by the output gate), while the GRU's update gate operates directly on $h_t$ (the exposed hidden state).</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Wrong concatenation order.** The weight matrix $W_f$ is defined with respect to a specific concatenation order. If $W_f$ expects $[h_{t-1}, x_t]$ but the implementation concatenates as $[x_t, h_{t-1}]$, the first $H$ columns of $W_f$ will multiply $x_t$ instead of $h_{t-1}$, and the last $D$ columns will multiply $h_{t-1}$ instead of $x_t$. No dimension error occurs since $H + D$ is the same either way, but the gate produces incorrect values. This is a silent bug that degrades performance without crashing.</span>

* <span style="font-size: 14px;">**Forgetting the sigmoid.** Omitting the sigmoid and using the raw affine output as the gate value means gate values are unbounded: they can be negative (flipping cell state signs) or greater than $1$ (amplifying cell state values). The cell state quickly diverges, producing NaN values within a few time steps. Every LSTM gate -- forget, input, and output -- must pass through sigmoid.</span>

* <span style="font-size: 14px;">**Initializing forget gate bias to zero.** With zero bias, the forget gate starts at $\sigma(0) = 0.5$ for all dimensions, meaning the cell immediately forgets half of its state at every time step. Over $T$ steps, information decays as $0.5^T$: after just 10 steps, only $0.1\%$ of the original information remains. Jozefowicz et al. (2015) showed that initializing $b_f = 1.0$ (so the gate starts near $\sigma(1) \approx 0.73$) dramatically improves training. Some frameworks use $b_f = 2.0$ for even stronger initial retention.</span>

* <span style="font-size: 14px;">**Confusing the forget gate with the input gate.** The forget gate controls what to **discard** from the previous cell state; the input gate controls what **new** information to write. In $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$, swapping $f_t$ and $i_t$ means the gate controlling retention instead controls new writes. The network may still train, but convergence is slower because the roles are semantically inverted.</span>

* <span style="font-size: 14px;">**Applying forget gate to hidden state instead of cell state.** The forget gate modulates $c_{t-1}$ (the cell state), not $h_{t-1}$ (the hidden state). The hidden state is the input to the gate computation, but the gate's output multiplies the cell state. Writing $f_t \odot h_{t-1}$ instead of $f_t \odot c_{t-1}$ bypasses the cell state entirely, and the cell state accumulates without decay -- reverting to the original 1997 problem that the forget gate was designed to solve.</span>

* <span style="font-size: 14px;">**Using tanh instead of sigmoid for the gate.** The tanh activation maps to $[-1, 1]$, which would allow the gate to flip the sign of cell state dimensions when the output is negative. A gate value of $-1$ would negate the cell state rather than forget it. The forget gate must output non-negative values to function as a retention fraction, which is why sigmoid (mapping to $(0, 1)$) is the correct activation. The candidate $\tilde{c}_t$ uses tanh because cell state values need to be both positive and negative, but the gates themselves must always use sigmoid.</span>

---
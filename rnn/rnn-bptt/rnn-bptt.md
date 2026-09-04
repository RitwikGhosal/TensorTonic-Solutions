# <span style="font-size: 20px;">Backpropagation Through Time</span>

<span style="font-size: 14px;">Backpropagation Through Time (BPTT) is the algorithm used to compute gradients in recurrent neural networks. First formalized by Werbos (1990) in "Backpropagation Through Time: What It Does and How to Do It," BPTT unrolls the recurrence across time steps and applies the standard chain rule backward through the resulting computational graph. It is the foundation for training every RNN variant, from vanilla RNNs to LSTMs and GRUs.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">In a vanilla RNN, the forward pass at each time step $t$ computes a new hidden state $h_t$ from the previous hidden state $h_{t-1}$ and the current input $x_t$:</span>

$$
h_t = \tanh(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b)
$$

<span style="font-size: 14px;">The backward pass must compute gradients of the loss with respect to all parameters ($W_{hh}$, $W_{xh}$, $b$) and propagate error signals back to earlier time steps. BPTT does this by conceptually unrolling the RNN into a feed-forward network with $T$ layers (one per time step), then running standard backpropagation on this unrolled graph. Each "layer" shares the same weights, so weight gradients are accumulated across all time steps.</span>

<span style="font-size: 14px;">This problem focuses on the single-step backward pass: given the hidden state $h_t$, the previous hidden state $h_{t-1}$, the recurrent weight matrix $W_{hh}$, and an incoming gradient $dh_{\text{next}}$ from the future, compute the gradient flowing backward to $h_{t-1}$ and the gradient of the loss with respect to $W_{hh}$. This single-step computation is the atomic building block that, when applied repeatedly from step $T$ back to step $0$, implements full BPTT.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">The forward pass at a single time step computes a pre-activation $z_t$ and then applies the tanh nonlinearity:</span>

$$
z_t = W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b
$$

$$
h_t = \tanh(z_t)
$$

<span style="font-size: 14px;">The backward pass through this step, given the gradient of the loss with respect to $h_t$ (denoted $dh_{\text{next}}$), proceeds as follows:</span>

<span style="font-size: 14px;">**Equation 1 -- Pre-activation gradient.** The gradient through the tanh nonlinearity:</span>

$$
d_{\tanh} = (1 - h_t^2) \odot dh_{\text{next}}
$$

<span style="font-size: 14px;">Here $\odot$ denotes element-wise multiplication. The term $(1 - h_t^2)$ is the derivative of tanh evaluated at $z_t$, expressed using the output $h_t$ directly. Each element of $d_{\tanh}$ represents how much the loss changes with respect to the corresponding element of the pre-activation $z_t$.</span>

<span style="font-size: 14px;">**Equation 2 -- Weight gradient for $W_{hh}$.** The gradient of the loss with respect to the recurrent weight matrix at this time step:</span>

$$
dW_{hh} = d_{\tanh}^T \cdot h_{t-1}
$$

<span style="font-size: 14px;">This is an outer product: $d_{\tanh}$ has shape $(H,)$ and $h_{t-1}$ has shape $(H,)$, producing $dW_{hh}$ with shape $(H, H)$. Each entry $dW_{hh}[i, j]$ measures how much the loss changes when $W_{hh}[i, j]$ changes, considering only the contribution at time step $t$.</span>

<span style="font-size: 14px;">**Equation 3 -- Previous hidden state gradient.** The gradient flowing backward to the previous time step:</span>

$$
dh_{\text{prev}} = d_{\tanh} \cdot W_{hh}
$$

<span style="font-size: 14px;">Here $d_{\tanh}$ has shape $(1, H)$ and $W_{hh}$ has shape $(H, H)$, so $dh_{\text{prev}}$ has shape $(1, H)$. This gradient becomes $dh_{\text{next}}$ for time step $t - 1$, forming the backward chain.</span>

---

## <span style="font-size: 16px;">The Tanh Derivative</span>

<span style="font-size: 14px;">The tanh activation maps any real number to $(-1, 1)$. Its derivative is central to BPTT for vanilla RNNs:</span>

$$
\frac{d}{dz}\tanh(z) = 1 - \tanh^2(z)
$$

<span style="font-size: 14px;">Since $h_t = \tanh(z_t)$, we substitute directly: $\frac{d}{dz_t}\tanh(z_t) = 1 - h_t^2$. This is the key computational shortcut. There is no need to store or recompute $z_t$ during the backward pass -- the forward pass already computed $h_t$, so the derivative is available immediately as $1 - h_t^2$. This saves both memory and computation.</span>

<span style="font-size: 14px;">The derivative has important numerical properties. At $h_t = 0$ (the origin), $1 - 0^2 = 1$, so the gradient passes through at full strength. As $|h_t|$ approaches $1$ (the saturation region), $1 - h_t^2$ approaches $0$, effectively killing the gradient. A hidden state of $h_t = 0.99$ produces a local derivative of $1 - 0.99^2 = 0.0199$, attenuating the gradient by a factor of roughly $50$.</span>

<span style="font-size: 14px;">The maximum derivative value is exactly $1$, occurring at $h_t = 0$. The tanh derivative can never amplify a gradient -- it can only pass it through unchanged or shrink it. This asymmetry between attenuation and amplification is one of the fundamental reasons vanilla RNNs struggle with long-range dependencies.</span>

---

## <span style="font-size: 16px;">The Chain Rule Through Time</span>

<span style="font-size: 14px;">The full BPTT algorithm applies the single-step backward pass repeatedly, starting from the final time step $T$ and working backward to $t = 0$. At each step, the gradient with respect to $h_t$ has two components: the gradient from the loss at time $t$ and the gradient propagated backward from time step $t + 1$.</span>

<span style="font-size: 14px;">The backward chain:</span>

$$
dh_T \rightarrow dh_{T-1} \rightarrow dh_{T-2} \rightarrow \ldots \rightarrow dh_1 \rightarrow dh_0
$$

<span style="font-size: 14px;">At each arrow, two operations occur: multiplication by the tanh derivative $(1 - h_t^2)$ and multiplication by $W_{hh}$. Expanding the gradient flowing from step $T$ to step $k$:</span>

$$
\frac{\partial h_T}{\partial h_k} = \prod_{t=k+1}^{T} \text{diag}(1 - h_t^2) \cdot W_{hh}
$$

<span style="font-size: 14px;">This is a product of $T - k$ Jacobian matrices. Each Jacobian at step $t$ is $\text{diag}(1 - h_t^2) \cdot W_{hh}$, where $\text{diag}(1 - h_t^2)$ is a diagonal matrix with the tanh derivatives on the diagonal. The gradient at step $k$ depends on this entire product chain.</span>

<span style="font-size: 14px;">The total weight gradient is the sum over all time steps: $\nabla_{W_{hh}} L = \sum_{t=1}^{T} dW_{hh}^{(t)}$. The value of $d_{\tanh}$ at each step depends on all gradients propagated from future steps, so this accumulation reflects the influence of $W_{hh}$ on every time step in the sequence.</span>

---

## <span style="font-size: 16px;">Why This Leads to Vanishing Gradients</span>

<span style="font-size: 14px;">The product $\prod_{t=k+1}^{T} \text{diag}(1 - h_t^2) \cdot W_{hh}$ grows or shrinks exponentially with the number of time steps $T - k$. Since every element of $(1 - h_t^2)$ lies in $(0, 1]$, the diagonal matrix can only shrink vectors or leave them unchanged. If the largest singular value of $W_{hh}$ is less than $1$, then each Jacobian is a contraction, and the product of $T - k$ contractions decays exponentially toward zero.</span>

<span style="font-size: 14px;">Consider a concrete scenario. Suppose the tanh derivatives average around $0.5$ and $W_{hh}$ has a spectral radius of $0.9$. The effective per-step attenuation is roughly $0.5 \times 0.9 = 0.45$. After $20$ time steps, the gradient has been multiplied by $0.45^{20} \approx 1.2 \times 10^{-7}$. The error signal from step $T$ arrives at step $T - 20$ as essentially zero. Learning long-range dependencies becomes impossible.</span>

<span style="font-size: 14px;">Bengio et al. (1994) provided the foundational analysis in "Learning Long-Term Dependencies with Gradient Descent is Difficult." They proved that for vanilla RNNs, gradients either vanish or explode exponentially as the temporal gap increases. The vanishing case is far more common because the tanh derivative's upper bound of $1$ biases the system toward contraction. Gradient clipping addresses the exploding case, but there is no symmetric fix for vanishing -- the information is simply lost. This analysis directly motivated gated architectures like the LSTM and GRU.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Werbos (1990) introduced BPTT in "Backpropagation Through Time: What It Does and How to Do It," published in Proceedings of the IEEE. The paper described how to apply backpropagation to dynamical systems by unrolling recurrence relations into static computational graphs. Werbos framed the key insight as: "Learning in recurrent networks requires the propagation of error back through time." To train a system with memory, the error signal must traverse every step where that memory was used.</span>

<span style="font-size: 14px;">The paper distinguished between two modes. **Full BPTT** unrolls the entire sequence from $t = 0$ to $t = T$ before computing any gradients, requiring storage of all hidden states. **Truncated BPTT** limits the backward pass to a fixed number of steps $k$, trading gradient accuracy for reduced memory. Truncated BPTT with $k$ steps means dependencies longer than $k$ steps cannot be learned at all, but it makes training on very long sequences practical. Modern frameworks like PyTorch default to full BPTT within each mini-batch sequence.</span>

<span style="font-size: 14px;">Bengio et al. (1994) built on this foundation, providing rigorous analysis of why BPTT fails to capture long-range dependencies in vanilla RNNs. They showed that the gradient magnitude decreases exponentially with the length of the dependency, making it mathematically impossible for standard gradient descent to assign credit to events far in the past. This paper established the theoretical justification for architectures with explicit memory mechanisms.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a single backward step with hidden size $H = 2$:</span>

$$
h_t = \begin{pmatrix} 0.8 & -0.6 \end{pmatrix}, \quad h_{t-1} = \begin{pmatrix} 0.3 & 0.7 \end{pmatrix}
$$

$$
W_{hh} = \begin{pmatrix} 0.5 & -0.3 \\ 0.2 & 0.8 \end{pmatrix}, \quad dh_{\text{next}} = \begin{pmatrix} 1.0 & -0.5 \end{pmatrix}
$$

<span style="font-size: 14px;">**Step 1: Tanh derivative.** $(1 - h_t^2)$:</span>

<span style="font-size: 14px;">Element 0: $1 - (0.8)^2 = 1 - 0.64 = 0.36$</span>

<span style="font-size: 14px;">Element 1: $1 - (-0.6)^2 = 1 - 0.36 = 0.64$</span>

$$
(1 - h_t^2) = \begin{pmatrix} 0.36 & 0.64 \end{pmatrix}
$$

<span style="font-size: 14px;">**Step 2: Pre-activation gradient.** $d_{\tanh} = (1 - h_t^2) \odot dh_{\text{next}}$:</span>

<span style="font-size: 14px;">Element 0: $0.36 \times 1.0 = 0.36$</span>

<span style="font-size: 14px;">Element 1: $0.64 \times (-0.5) = -0.32$</span>

$$
d_{\tanh} = \begin{pmatrix} 0.36 & -0.32 \end{pmatrix}
$$

<span style="font-size: 14px;">**Step 3: Weight gradient.** $dW_{hh} = d_{\tanh}^T \cdot h_{t-1}$:</span>

<span style="font-size: 14px;">$dW_{hh}[0, 0] = 0.36 \times 0.3 = 0.108$</span>

<span style="font-size: 14px;">$dW_{hh}[0, 1] = 0.36 \times 0.7 = 0.252$</span>

<span style="font-size: 14px;">$dW_{hh}[1, 0] = (-0.32) \times 0.3 = -0.096$</span>

<span style="font-size: 14px;">$dW_{hh}[1, 1] = (-0.32) \times 0.7 = -0.224$</span>

$$
dW_{hh} = \begin{pmatrix} 0.108 & 0.252 \\ -0.096 & -0.224 \end{pmatrix}
$$

<span style="font-size: 14px;">**Step 4: Previous hidden state gradient.** $dh_{\text{prev}} = d_{\tanh} \cdot W_{hh}$:</span>

<span style="font-size: 14px;">$dh_{\text{prev}}[0] = 0.36 \times 0.5 + (-0.32) \times 0.2 = 0.18 - 0.064 = 0.116$</span>

<span style="font-size: 14px;">$dh_{\text{prev}}[1] = 0.36 \times (-0.3) + (-0.32) \times 0.8 = -0.108 - 0.256 = -0.364$</span>

$$
dh_{\text{prev}} = \begin{pmatrix} 0.116 & -0.364 \end{pmatrix}
$$

<span style="font-size: 14px;">The returned tuple is $(dh_{\text{prev}}, dW_{hh}) = ([0.116, -0.364], [[0.108, 0.252], [-0.096, -0.224]])$. Notice the attenuation: $dh_{\text{next}}$ had magnitude $\sqrt{1.0^2 + 0.5^2} = 1.118$, while $dh_{\text{prev}}$ has magnitude $\sqrt{0.116^2 + 0.364^2} = 0.382$. In one step, the gradient magnitude dropped by a factor of roughly $3$.</span>

---

## <span style="font-size: 16px;">Connection to LSTM and GRU</span>

<span style="font-size: 14px;">The vanishing gradient problem in vanilla RNN BPTT directly motivated the LSTM (Hochreiter and Schmidhuber, 1997). The LSTM introduces a **cell state** $c_t$ updated through additive operations: $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$. When $f_t \approx 1$ and $i_t \approx 0$, the cell state passes through unchanged: $c_t \approx c_{t-1}$. This creates a gradient highway where $\partial c_t / \partial c_{t-1} \approx 1$, regardless of how many time steps separate the signal from the loss.</span>

<span style="font-size: 14px;">Hochreiter and Schmidhuber called this the **constant error carousel (CEC)**: the cell state can carry information and gradients across arbitrarily long sequences without exponential decay. The forget gate, input gate, and output gate control what enters, persists in, and exits the carousel. This is fundamentally different from the vanilla RNN, where every backward step forces the gradient through a tanh derivative and a matrix multiplication, guaranteeing eventual decay.</span>

<span style="font-size: 14px;">The GRU (Cho et al., 2014) achieves a similar effect with fewer parameters. The update gate $z_t$ interpolates between old and new hidden states: $h_t = z_t \odot h_{t-1} + (1 - z_t) \odot \tilde{h}_t$. When $z_t \approx 1$, the gradient passes through the $z_t \odot h_{t-1}$ path nearly unchanged. Both architectures replace the problematic multiplicative chain with additive shortcuts that allow gradients to flow across many time steps with minimal attenuation.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Using $\tanh(z_t)$ instead of $h_t$ for the derivative.** The derivative $1 - \tanh^2(z_t)$ is mathematically identical to $1 - h_t^2$, but the implementation should use $h_t$ directly. Recomputing $\tanh(z_t)$ requires storing or recomputing $z_t$, which wastes memory. If $z_t$ is reconstructed incorrectly, the resulting gradient errors are extremely difficult to debug.</span>

* <span style="font-size: 14px;">**Wrong matrix transpose in $dh_{\text{prev}}$.** The gradient $dh_{\text{prev}} = d_{\tanh} \cdot W_{hh}$ uses $W_{hh}$ directly when $d_{\tanh}$ is a row vector of shape $(1, H)$. Transposing $W_{hh}$ here produces an incorrect gradient. The correct rule depends on how the forward pass is written: if the forward is $z = W_{hh} \cdot h_{t-1}$ (matrix times column vector), then $dh_{t-1} = W_{hh}^T \cdot d_{\tanh}^T$, which is equivalent to $d_{\tanh} \cdot W_{hh}$ in row-vector form. Mismatching the convention is a common source of silent errors.</span>

* <span style="font-size: 14px;">**Confusing per-step $dW_{hh}$ with the total gradient.** Each time step produces its own $dW_{hh}^{(t)} = d_{\tanh}^T \cdot h_{t-1}$. The total gradient is $\nabla_{W_{hh}} L = \sum_{t=1}^{T} dW_{hh}^{(t)}$. Forgetting to accumulate (using only the last step's gradient) means the update ignores the influence of $W_{hh}$ on all earlier time steps. The network may still train slowly, masking the bug, but long-range learning is completely broken.</span>

* <span style="font-size: 14px;">**Forgetting element-wise multiplication for $d_{\tanh}$.** The pre-activation gradient $d_{\tanh} = (1 - h_t^2) \odot dh_{\text{next}}$ is an element-wise (Hadamard) product, not a dot product or matrix multiplication. Using a dot product would collapse the vector to a scalar. Both $(1 - h_t^2)$ and $dh_{\text{next}}$ have shape $(H,)$ or $(1, H)$, and the result must have the same shape.</span>

* <span style="font-size: 14px;">**Neglecting gradient clipping for the exploding case.** While vanishing gradients are more common, the opposite occurs when $W_{hh}$ has large singular values. The gradient magnitude grows exponentially, causing parameter updates to overshoot catastrophically. Gradient clipping (scaling the gradient when its norm exceeds a threshold) is the standard remedy.</span>

* <span style="font-size: 14px;">**Assuming $dh_{\text{next}}$ comes only from the next time step.** At each time step $t$, the total gradient with respect to $h_t$ may include contributions from the loss at time $t$ itself plus the gradient propagated from step $t + 1$. These two sources must be summed. Ignoring the local loss gradient means the network only learns from the final time step's loss.</span>

---
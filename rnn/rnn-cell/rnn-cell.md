# <span style="font-size: 20px;">Single RNN Cell</span>

<span style="font-size: 14px;">The single RNN cell is the fundamental recurrent computation unit introduced by Elman (1990) in "Finding Structure in Time." It takes a current input vector $x_t$ and the previous hidden state $h_{t-1}$, combines them through learned weight matrices, applies a tanh nonlinearity, and produces a new hidden state $h_t$. This is the simplest possible recurrent architecture, and every gated variant (LSTM, GRU) builds directly on top of its core idea.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">The single RNN cell, also called the vanilla RNN cell or Elman cell, is the most basic recurrent neural network unit. At each time step $t$, it receives two inputs: the current input vector $x_t \in \mathbb{R}^{D}$ (where $D$ is the input dimensionality) and the previous hidden state $h_{t-1} \in \mathbb{R}^{H}$ (where $H$ is the hidden dimensionality). It produces one output: the new hidden state $h_t \in \mathbb{R}^{H}$.</span>

<span style="font-size: 14px;">The cell performs a single affine transformation on both inputs simultaneously, then passes the result through a tanh activation function. There are no gates, no cell state, no output projection. The hidden state $h_t$ serves as both the cell's output and the memory that gets passed to the next time step. For a batch of inputs with batch size $B$, the output shape is $(B, H)$.</span>

<span style="font-size: 14px;">Elman described this as a network where "the context layer provides a memory of the prior internal state." The context layer is simply $h_{t-1}$ fed back as an additional input. This feedback loop is what makes the network recurrent: the same weights are applied at every time step, but the hidden state carries forward a compressed summary of all previous inputs.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">The complete forward pass of a single RNN cell is defined by one equation:</span>

$$
h_t = \tanh(x_t \cdot W_{xh}^T + h_{t-1} \cdot W_{hh}^T + b_h)
$$

<span style="font-size: 14px;">This equation has four components: (1) the input contribution $x_t \cdot W_{xh}^T$, (2) the recurrent contribution $h_{t-1} \cdot W_{hh}^T$, (3) the bias $b_h$, and (4) the element-wise tanh activation.</span>

<span style="font-size: 14px;">**Input contribution.** The term $x_t \cdot W_{xh}^T$ projects the input from $\mathbb{R}^D$ into $\mathbb{R}^H$. The weight matrix $W_{xh} \in \mathbb{R}^{H \times D}$ determines how each input feature influences each hidden dimension. The transpose ensures conforming dimensions: $(B, D) \times (D, H) = (B, H)$.</span>

<span style="font-size: 14px;">**Recurrent contribution.** The term $h_{t-1} \cdot W_{hh}^T$ projects the previous hidden state back into hidden space. The weight matrix $W_{hh} \in \mathbb{R}^{H \times H}$ is always square because both input and output live in $\mathbb{R}^H$. This is the recurrent connection: it allows information from all previous time steps, compressed into $h_{t-1}$, to influence the current computation.</span>

<span style="font-size: 14px;">**Bias.** The vector $b_h \in \mathbb{R}^{H}$ is added element-wise after both matrix multiplications. It shifts the pre-activation values, allowing the network to learn non-zero outputs even when both $x_t$ and $h_{t-1}$ are zero vectors. During the addition, $b_h$ is broadcast across the batch dimension.</span>

<span style="font-size: 14px;">**Tanh activation.** The tanh function is applied element-wise to the sum, squashing each element independently into the range $(-1, 1)$ and producing the final hidden state $h_t$.</span>

---

## <span style="font-size: 16px;">The Two Input Streams</span>

<span style="font-size: 14px;">The RNN cell has exactly two input streams that combine additively before the activation function.</span>

<span style="font-size: 14px;">**Stream 1: Current input $x_t$.** This is the new information arriving at time step $t$. In a language model, $x_t$ might be the embedding vector for the current word. The input passes through $W_{xh}$ to be projected into hidden space. This stream has no memory: it sees only the data at the current time step.</span>

<span style="font-size: 14px;">**Stream 2: Previous hidden state $h_{t-1}$.** This is the memory of the network, containing a compressed representation of everything the network has seen from time steps $1$ through $t-1$. The hidden state passes through $W_{hh}$ to be projected back into hidden space. At $t = 1$, $h_0$ is typically initialized to a zero vector, meaning the network starts with no memory.</span>

<span style="font-size: 14px;">The two projected streams are summed element-wise. Each hidden dimension receives a contribution from the current input and a contribution from the previous state, and these are simply added together. The tanh then compresses the sum. This is fundamentally different from gated architectures like LSTM and GRU, where multiplicative gates control how the two streams interact. In the vanilla RNN cell, there is no mechanism to selectively ignore the input or selectively forget the previous state.</span>

---

## <span style="font-size: 16px;">Why Tanh</span>

<span style="font-size: 14px;">The choice of tanh as the activation function is deliberate and has several important properties for recurrent computation.</span>

<span style="font-size: 14px;">**Bounded output.** Tanh maps any real number to $(-1, 1)$, preventing unbounded growth of the hidden state. Without bounding, repeated matrix multiplications across time steps could cause hidden state values to explode. Since $h_t$ is fed back at the next step, unbounded values would compound and cause numerical overflow.</span>

<span style="font-size: 14px;">**Zero-centered.** Unlike sigmoid, which maps to $(0, 1)$ with mean $0.5$, tanh is centered at zero. Hidden state values can be both positive and negative. Zero-centered activations avoid systematic bias in gradient updates to subsequent layers.</span>

<span style="font-size: 14px;">**Stronger gradients near zero.** The derivative of tanh at $x = 0$ is $1$, the maximum possible value. For sigmoid, the maximum derivative is $0.25$. Tanh passes gradients more effectively in the linear regime near zero. The derivative is $\tanh'(x) = 1 - \tanh^2(x)$, so for $|h_t| \ll 1$, the gradient is close to $1$.</span>

<span style="font-size: 14px;">**Saturation regime.** For large $|x|$, tanh saturates: the gradient approaches zero. This bounds values but also contributes to the vanishing gradient problem over long sequences, since gradients must pass through tanh derivatives at every time step.</span>

---

## <span style="font-size: 16px;">Weight Matrices</span>

<span style="font-size: 14px;">The single RNN cell has three learnable parameter groups.</span>

* <span style="font-size: 14px;">**$W_{xh} \in \mathbb{R}^{H \times D}$:** The input-to-hidden weight matrix with $H \times D$ parameters. Each row corresponds to one hidden dimension, each column to one input feature.</span>
* <span style="font-size: 14px;">**$W_{hh} \in \mathbb{R}^{H \times H}$:** The hidden-to-hidden weight matrix with $H \times H$ parameters. Always square because it maps hidden space to itself. Responsible for the recurrent dynamics.</span>
* <span style="font-size: 14px;">**$b_h \in \mathbb{R}^{H}$:** The hidden bias vector with $H$ parameters.</span>

<span style="font-size: 14px;">**Total parameter count:**</span>

$$
H \times D + H \times H + H = H(D + H + 1)
$$

<span style="font-size: 14px;">With $H = 256$ and $D = 128$: $256 \times 128 + 256 \times 256 + 256 = 32{,}768 + 65{,}536 + 256 = 98{,}560$ parameters. An LSTM with the same dimensions has $4H(D + H) + 4H = 394{,}240$ parameters, roughly four times more.</span>

<span style="font-size: 14px;">**Dimension convention.** The convention $W_{xh} \in \mathbb{R}^{H \times D}$ means $H$ rows and $D$ columns. When computing $x_t \cdot W_{xh}^T$, the transpose gives $(B, D) \times (D, H) = (B, H)$. This matches PyTorch's `nn.RNN` where weight matrices are stored as $(H, D)$ and $(H, H)$.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Jeffrey Elman introduced the Simple Recurrent Network in his 1990 paper "Finding Structure in Time." The paper addressed how neural networks can process sequential data where relevant context extends over variable-length windows. Feed-forward networks require fixed-size inputs, making them inherently unsuitable for sequences of arbitrary length.</span>

<span style="font-size: 14px;">Elman's solution was to augment a feed-forward network with a "context layer" that copies the hidden layer's activations at time $t$ and feeds them back as additional input at time $t+1$. This is exactly the $h_{t-1}$ term. The paper demonstrated that this mechanism was sufficient for the network to learn temporal structure: when trained on word sequences, hidden states spontaneously organized into clusters reflecting grammatical categories (nouns, verbs, adjectives) without explicit supervision.</span>

<span style="font-size: 14px;">An important predecessor was the Jordan network (Jordan, 1986), which fed the network's output rather than its hidden state back as context. The Elman architecture proved more flexible because hidden states are not constrained to match any output format. The hidden state develops its own internal representation optimized for temporal processing, independent of task requirements.</span>

<span style="font-size: 14px;">The paper's most striking finding was that hidden state representations learned meaningful structure without supervision. Hierarchical clustering of hidden states revealed that the network had discovered syntactic and semantic categories purely from statistical structure. This demonstrated that recurrent hidden states are learned representations capturing the underlying structure of the data, not merely a technical mechanism for passing information forward.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a single RNN cell with $H = 2$ and $D = 3$. At time step $t$:</span>

$$
x_t = \begin{pmatrix} 0.5 & -0.3 & 0.8 \end{pmatrix}, \quad h_{t-1} = \begin{pmatrix} 0.1 & -0.4 \end{pmatrix}
$$

<span style="font-size: 14px;">The weight matrices and bias:</span>

$$
W_{xh} = \begin{pmatrix} 0.2 & -0.1 & 0.3 \\ 0.4 & 0.5 & -0.2 \end{pmatrix}, \quad W_{hh} = \begin{pmatrix} 0.1 & -0.3 \\ 0.2 & 0.6 \end{pmatrix}, \quad b_h = \begin{pmatrix} 0.1 \\ -0.1 \end{pmatrix}
$$

<span style="font-size: 14px;">**Step 1: Input contribution.** Compute $x_t \cdot W_{xh}^T$. Since $W_{xh}$ is $(2, 3)$, $W_{xh}^T$ is $(3, 2)$, and $x_t$ is $(1, 3)$, the result is $(1, 2)$:</span>

<span style="font-size: 14px;">Element [0]: $0.5 \times 0.2 + (-0.3) \times (-0.1) + 0.8 \times 0.3 = 0.10 + 0.03 + 0.24 = 0.37$</span>

<span style="font-size: 14px;">Element [1]: $0.5 \times 0.4 + (-0.3) \times 0.5 + 0.8 \times (-0.2) = 0.20 - 0.15 - 0.16 = -0.11$</span>

<span style="font-size: 14px;">**Step 2: Recurrent contribution.** Compute $h_{t-1} \cdot W_{hh}^T$:</span>

<span style="font-size: 14px;">Element [0]: $0.1 \times 0.1 + (-0.4) \times 0.2 = 0.01 - 0.08 = -0.07$</span>

<span style="font-size: 14px;">Element [1]: $0.1 \times (-0.3) + (-0.4) \times 0.6 = -0.03 - 0.24 = -0.27$</span>

<span style="font-size: 14px;">**Step 3: Sum and add bias.**</span>

<span style="font-size: 14px;">Pre-activation[0]: $0.37 + (-0.07) + 0.1 = 0.40$</span>

<span style="font-size: 14px;">Pre-activation[1]: $-0.11 + (-0.27) + (-0.1) = -0.48$</span>

<span style="font-size: 14px;">**Step 4: Apply tanh.**</span>

<span style="font-size: 14px;">$h_t[0] = \tanh(0.40) = 0.3799$</span>

<span style="font-size: 14px;">$h_t[1] = \tanh(-0.48) = -0.4462$</span>

<span style="font-size: 14px;">**Final result:** $h_t = [0.3799, -0.4462]$. Both values are in $(-1, 1)$ as guaranteed by tanh. The input contribution dominated dimension 0 (pulling it positive) while the recurrent contribution dominated dimension 1 (pulling it negative). The bias $+0.1$ in dimension 0 reinforced the positive direction; the bias $-0.1$ in dimension 1 amplified the negative direction.</span>

---

## <span style="font-size: 16px;">The Elman Cell vs Gated Cells</span>

<span style="font-size: 14px;">The vanilla RNN cell is the simplest possible recurrent unit. It performs one matrix multiply on the input, one on the previous state, adds them with a bias, and applies tanh. There are no gates, no memory cells, no multiplicative interactions.</span>

<span style="font-size: 14px;">**LSTM.** Hochreiter and Schmidhuber (1997) introduced the LSTM to address the vanishing gradient problem. The LSTM adds three gates (forget, input, output) and a separate cell state $c_t$ that acts as a linear conveyor belt for gradients. The cell state carries information across many time steps with minimal gradient decay because the forget gate allows a near-identity mapping: $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$. The vanilla RNN has no equivalent mechanism.</span>

<span style="font-size: 14px;">**GRU.** Cho et al. (2014) proposed the GRU as a simpler gated alternative with two gates (reset and update) and no separate cell state. The update gate interpolates between the old hidden state and a candidate: $h_t = z_t \odot h_{t-1} + (1 - z_t) \odot \tilde{h}_t$. This interpolation provides a direct gradient path through $z_t \odot h_{t-1}$, similar to the LSTM's cell state.</span>

<span style="font-size: 14px;">**The vanishing gradient limitation.** During backpropagation through time, the gradient passes through the Jacobian $\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(1 - h_t^2) \cdot W_{hh}^T$ at every time step. Over $T$ steps, this involves a product of $T$ Jacobians. If the spectral norm of $W_{hh}$ is less than $1$, gradients vanish exponentially. If greater than $1$, they explode. Gradient clipping addresses exploding gradients, but vanishing gradients require architectural changes like gating. This is why vanilla RNN cells struggle with dependencies longer than roughly 10-20 time steps.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Wrong matrix multiplication order.** The correct computation is $x_t \cdot W_{xh}^T$, not $W_{xh} \cdot x_t$. With batched inputs of shape $(B, D)$, the multiplication $(B, D) \times (D, H)$ produces $(B, H)$. Writing $W_{xh} \cdot x_t$ attempts $(H, D) \times (B, D)$, which fails or, if $B = D$ by coincidence, produces silently wrong results of shape $(H, D)$ instead of $(B, H)$.</span>

* <span style="font-size: 14px;">**Forgetting the tanh activation.** Without tanh, the cell becomes a linear recurrence: $h_t = x_t \cdot W_{xh}^T + h_{t-1} \cdot W_{hh}^T + b_h$. A linear recurrence collapses to a single linear transformation regardless of time steps. The network loses nonlinear pattern learning, and hidden states can grow without bound.</span>

* <span style="font-size: 14px;">**Wrong weight dimensions.** $W_{xh}$ must be $(H, D)$ and $W_{hh}$ must be $(H, H)$. A common mistake is making $W_{xh}$ square as $(D, D)$ or $(H, H)$ when $D \neq H$, or swapping to $(D, H)$ but forgetting to remove the transpose.</span>

* <span style="font-size: 14px;">**Confusing $W_{xh}$ and $W_{hh}$.** The subscript convention is: $W_{xh}$ maps from $x$ (input) to $h$ (hidden), $W_{hh}$ maps from $h$ to $h$. Swapping them when $D = H$ causes no dimension error but applies recurrent weights to the wrong input stream, producing incorrect learned representations.</span>

* <span style="font-size: 14px;">**Batch dimension handling.** The bias $b_h \in \mathbb{R}^H$ must be broadcast across the batch dimension. A common error is manually tiling the bias to $(B, H)$ instead of relying on broadcasting, which wastes memory and can introduce bugs if batch size changes.</span>

* <span style="font-size: 14px;">**Initializing $h_0$ with wrong shape.** The initial hidden state must have shape $(B, H)$ to match the batch. Passing shape $(H,)$ without the batch dimension causes a broadcasting mismatch. A safe default is `torch.zeros(batch_size, hidden_dim)`.</span>

---
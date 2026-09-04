# <span style="font-size: 20px;">Complete Vanilla RNN</span>

<span style="font-size: 14px;">The Vanilla RNN, also called the Elman network, is the foundational recurrent architecture for processing sequential data. Introduced by Jeffrey Elman in "Finding Structure in Time" (1990), it chains a simple recurrent cell over $T$ time steps, projecting each hidden state to an output, and returns both the full output sequence and the final hidden state.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">A complete Vanilla RNN is a sequence-to-sequence model built from three components: (1) recurrent weight matrices that define a single-step cell, (2) a loop that unrolls this cell across $T$ time steps, and (3) an output projection that maps each hidden state to a prediction vector. The network takes an input tensor $X \in \mathbb{R}^{B \times T \times D}$, an optional initial hidden state $h_0 \in \mathbb{R}^{B \times H}$, and produces two outputs: the output sequence $y_{\text{seq}} \in \mathbb{R}^{B \times T \times O}$ and the final hidden state $h_{\text{final}} \in \mathbb{R}^{B \times H}$.</span>

<span style="font-size: 14px;">The recurrent cell at each time step combines the current input $x_t$ with the previous hidden state $h_{t-1}$ through two separate linear transformations, adds a bias, and passes the result through $\tanh$. The output projection then applies an affine transformation to produce the output logits. This two-stage design separates the network's memory mechanism (the recurrence) from its prediction mechanism (the projection), allowing the hidden dimension $H$ and output dimension $O$ to differ.</span>

<span style="font-size: 14px;">The Vanilla RNN is the simplest recurrent architecture: a single hidden state vector with no gating, no cell state, and no attention. Every aspect of memory management is delegated to the $\tanh$ nonlinearity and the learned weight matrices. This simplicity makes it the ideal starting point for understanding recurrence, even though practical systems have replaced it with gated variants.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">Let $x_t \in \mathbb{R}^D$ be the input at time step $t$ and $h_{t-1} \in \mathbb{R}^H$ be the previous hidden state ($h_0$ defaults to zeros). The network applies two equations at every time step.</span>

<span style="font-size: 14px;">**Equation 1 -- Recurrent cell.** Computes the new hidden state:</span>

$$
h_t = \tanh(x_t W_{xh}^T + h_{t-1} W_{hh}^T + b_h)
$$

<span style="font-size: 14px;">Here $W_{xh} \in \mathbb{R}^{H \times D}$ maps input to hidden space, $W_{hh} \in \mathbb{R}^{H \times H}$ maps the previous hidden state to hidden space, and $b_h \in \mathbb{R}^H$ is the hidden bias. The $\tanh$ squashes each element to $(-1, 1)$, bounding the hidden state.</span>

<span style="font-size: 14px;">**Equation 2 -- Output projection.** Maps each hidden state to the output space:</span>

$$
y_t = h_t W_{hy}^T + b_y
$$

<span style="font-size: 14px;">Here $W_{hy} \in \mathbb{R}^{O \times H}$ maps from hidden to output space and $b_y \in \mathbb{R}^O$ is the output bias. This is a linear layer with no activation. The raw $y_t$ can be logits (classification) or predictions (regression).</span>

<span style="font-size: 14px;">**Initialization.** All weight matrices use Xavier initialization with scale $\sqrt{2 / (\text{fan\_in} + \text{fan\_out})}$, and all biases are initialized to zero. Xavier is critical for the recurrent setting because $W_{hh}$ is applied at every time step. If the initial scale is too large, activations explode within a few steps; if too small, they vanish to zero before the sequence ends.</span>

---

## <span style="font-size: 16px;">The Three Weight Matrices</span>

<span style="font-size: 14px;">The Vanilla RNN has exactly three weight matrices and two bias vectors. Each plays a distinct role.</span>

<span style="font-size: 14px;">**$W_{xh} \in \mathbb{R}^{H \times D}$ -- Input-to-hidden.** Transforms the raw input at each time step into hidden space. It sees only the current $x_t$ with no access to history. Its role is feature extraction from the input for the hidden state update. Xavier scale: $\sqrt{2 / (D + H)}$.</span>

<span style="font-size: 14px;">**$W_{hh} \in \mathbb{R}^{H \times H}$ -- Hidden-to-hidden.** The recurrence matrix, the defining feature of the RNN. Because $h_t$ depends on $h_{t-1}$ through $W_{hh}$, and $h_{t-1}$ on $h_{t-2}$ through the same $W_{hh}$, this single matrix is applied $T$ times per forward pass. The eigenvalues of $W_{hh}$ directly control whether information persists, decays, or explodes across time steps. Xavier scale: $\sqrt{2 / (2H)} = \sqrt{1/H}$.</span>

<span style="font-size: 14px;">**$W_{hy} \in \mathbb{R}^{O \times H}$ -- Hidden-to-output.** Projects the hidden state into output space at each time step. Not part of the recurrence, so it does not affect information flow between time steps. For a language model with vocabulary size $V$, $O = V$ and $W_{hy}$ maps the $H$-dimensional hidden state to $V$-dimensional logits. Xavier scale: $\sqrt{2 / (H + O)}$.</span>

<span style="font-size: 14px;">Total parameter count: $H(D + H + O) + H + O$. For $D = 128$, $H = 256$, $O = 64$: $256(128 + 256 + 64) + 256 + 64 = 114{,}688 + 320 = 115{,}008$.</span>

---

## <span style="font-size: 16px;">Xavier Initialization</span>

<span style="font-size: 14px;">Xavier initialization (Glorot and Bengio, 2010) sets the initial weight scale so that activation variance remains approximately constant through the network. For a weight matrix with $n_{\text{in}}$ inputs and $n_{\text{out}}$ outputs:</span>

$$
W_{ij} \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)
$$

<span style="font-size: 14px;">In code: $W = \text{randn}(n_{\text{out}}, n_{\text{in}}) \times \sqrt{2 / (n_{\text{in}} + n_{\text{out}})}$.</span>

<span style="font-size: 14px;">**Why Xavier matters for RNNs.** In a feedforward network, each weight matrix is applied once. In the Vanilla RNN, $W_{hh}$ is applied $T$ times. If the singular values of $W_{hh}$ are slightly greater than 1, hidden state magnitude grows exponentially as $\|h_t\| \sim \sigma_{\max}^t$. If slightly less than 1, it decays exponentially. Xavier targets a variance that keeps expected singular values near 1.</span>

<span style="font-size: 14px;">**Xavier vs He.** He initialization uses scale $\sqrt{2 / n_{\text{in}}}$, designed for ReLU where half the activations are zeroed out. The Vanilla RNN uses $\tanh$, which is active everywhere (derivative in $(0, 1]$). Xavier is the correct choice. Using He with $\tanh$ over-scales weights, pushing activations toward saturation where gradients are near zero.</span>

---

## <span style="font-size: 16px;">The Forward Pass</span>

<span style="font-size: 14px;">The forward pass takes $X \in \mathbb{R}^{B \times T \times D}$ and optional $h_0 \in \mathbb{R}^{B \times H}$ (defaults to zeros). It proceeds in three stages.</span>

<span style="font-size: 14px;">**Stage 1: Initialize.** Set $h_{\text{curr}} = h_0$. If not provided, create a zero tensor of shape $(B, H)$. This is a "blank slate" memory with no prior context.</span>

<span style="font-size: 14px;">**Stage 2: Unroll the recurrence.** For $t = 0, 1, \ldots, T-1$: extract $x_t = X[:, t, :]$, apply the cell $h_{\text{curr}} = \tanh(x_t W_{xh}^T + h_{\text{curr}} W_{hh}^T + b_h)$, store $h_{\text{curr}}$. After the loop, stack all states into $H_{\text{all}} \in \mathbb{R}^{B \times T \times H}$.</span>

<span style="font-size: 14px;">**Stage 3: Output projection.** Reshape $H_{\text{all}}$ from $(B, T, H)$ to $(B \cdot T, H)$, apply $y_{\text{flat}} = H_{\text{flat}} W_{hy}^T + b_y$ to get $(B \cdot T, O)$, reshape back to $(B, T, O)$. This batched projection avoids $T$ separate matrix multiplications.</span>

<span style="font-size: 14px;">Return the tuple $(y_{\text{seq}}, h_{\text{final}})$. The caller can use $h_{\text{final}}$ to initialize the next forward pass when processing in chunks, or ignore it for single-sequence tasks.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Jeffrey Elman published "Finding Structure in Time" in 1990, introducing the Simple Recurrent Network (SRN). The paper's central insight was that a network with recurrent connections could learn temporal structure without explicit programming. Elman described the architecture as providing "a mechanism for learning to encode and decode temporal patterns."</span>

<span style="font-size: 14px;">The key discovery was that hidden units self-organized into meaningful representations. When trained on word sequences from simple grammatical sentences, the hidden state vectors clustered by grammatical category: nouns grouped together, verbs grouped together, with semantic subcategories emerging within each group. The network received no linguistic information and learned these categories purely from next-word prediction.</span>

<span style="font-size: 14px;">Elman's architecture used a "context layer" that copied the hidden state and fed it back as input at the next step. Mathematically identical to the direct recurrence $h_t = f(x_t, h_{t-1})$ used today, the copy mechanism made the recurrence explicit in the era's frameworks.</span>

<span style="font-size: 14px;">The applications (next-word prediction, temporal pattern discovery) are the ancestors of modern language modeling. GPT-style models do the same thing conceptually -- predicting the next token given history -- but with attention replacing recurrence, billions of parameters replacing hundreds, and web-scale text replacing toy grammars.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Vanilla RNN with $D = 2$, $H = 2$, $O = 2$, $T = 3$, $B = 1$:</span>

$$
W_{xh} = \begin{pmatrix} 0.3 & -0.1 \\ 0.2 & 0.4 \end{pmatrix}, \quad W_{hh} = \begin{pmatrix} 0.1 & -0.2 \\ 0.3 & 0.1 \end{pmatrix}
$$

$$
W_{hy} = \begin{pmatrix} 0.5 & -0.3 \\ -0.1 & 0.4 \end{pmatrix}, \quad b_h = \begin{pmatrix} 0 \\ 0 \end{pmatrix}, \quad b_y = \begin{pmatrix} 0 \\ 0 \end{pmatrix}
$$

$$
x_0 = \begin{pmatrix} 1.0 \\ 0.5 \end{pmatrix}, \quad x_1 = \begin{pmatrix} -0.5 \\ 0.3 \end{pmatrix}, \quad x_2 = \begin{pmatrix} 0.2 \\ -0.8 \end{pmatrix}
$$

<span style="font-size: 14px;">**$h_0 = (0, 0)$.**</span>

<span style="font-size: 14px;">**Step $t = 0$:** $x_0 W_{xh}^T$: element 1 = $1.0(0.3) + 0.5(-0.1) = 0.25$, element 2 = $1.0(0.2) + 0.5(0.4) = 0.40$. Since $h_0$ is zeros, pre-activation = $(0.25, 0.40)$. $h_1 = (\tanh(0.25), \tanh(0.40)) = (0.2449, 0.3799)$.</span>

<span style="font-size: 14px;">Output: $y_0 = h_1 W_{hy}^T$. Element 1 = $0.2449(0.5) + 0.3799(-0.3) = 0.0085$. Element 2 = $0.2449(-0.1) + 0.3799(0.4) = 0.1275$. $y_0 = (0.0085, 0.1275)$.</span>

<span style="font-size: 14px;">**Step $t = 1$:** $x_1 W_{xh}^T = (-0.18, 0.02)$. $h_1 W_{hh}^T$: element 1 = $0.2449(0.1) + 0.3799(-0.2) = -0.0515$, element 2 = $0.2449(0.3) + 0.3799(0.1) = 0.1115$. Pre-activation = $(-0.2315, 0.1315)$. $h_2 = (-0.2274, 0.1307)$.</span>

<span style="font-size: 14px;">Output: $y_1$. Element 1 = $(-0.2274)(0.5) + 0.1307(-0.3) = -0.1529$. Element 2 = $(-0.2274)(-0.1) + 0.1307(0.4) = 0.0750$. $y_1 = (-0.1529, 0.0750)$.</span>

<span style="font-size: 14px;">**Step $t = 2$:** $x_2 W_{xh}^T = (0.14, -0.28)$. $h_2 W_{hh}^T$: element 1 = $(-0.2274)(0.1) + 0.1307(-0.2) = -0.0489$, element 2 = $(-0.2274)(0.3) + 0.1307(0.1) = -0.0551$. Pre-activation = $(0.0911, -0.3351)$. $h_3 = (0.0909, -0.3232)$.</span>

<span style="font-size: 14px;">Output: $y_2$. Element 1 = $0.0909(0.5) + (-0.3232)(-0.3) = 0.1424$. Element 2 = $0.0909(-0.1) + (-0.3232)(0.4) = -0.1384$. $y_2 = (0.1424, -0.1384)$.</span>

<span style="font-size: 14px;">**Result:** $y_{\text{seq}}$ shape $(1, 3, 2)$: $(0.0085, 0.1275)$, $(-0.1529, 0.0750)$, $(0.1424, -0.1384)$. $h_{\text{final}} = (0.0909, -0.3232)$. Note $h_3$ depends on $x_2$ directly, on $x_1$ through $h_2$, and on $x_0$ through $h_1$: the recurrent chain encodes all prior history.</span>

---

## <span style="font-size: 16px;">Limitations</span>

<span style="font-size: 14px;">**Vanishing gradients.** During BPTT, the gradient with respect to early $h_k$ involves a Jacobian product $\prod_{t=k+1}^{T} \frac{\partial h_t}{\partial h_{t-1}}$. Each factor includes $\tanh'$ (values in $(0, 1]$) times $W_{hh}$. When the spectral radius of $W_{hh}$ is below 1, this product shrinks exponentially, making it impossible to learn dependencies beyond roughly 10-20 steps.</span>

<span style="font-size: 14px;">**Exploding gradients.** If the spectral radius exceeds 1, gradients grow exponentially. Gradient clipping mitigates this but is a band-aid. The Vanilla RNN is uniquely vulnerable because the same $W_{hh}$ is applied at every step with no gating.</span>

<span style="font-size: 14px;">**Short-term memory only.** Information from early time steps is progressively overwritten. For tasks requiring long-range context (document summarization, distant agreement), the Vanilla RNN fails to propagate signals far enough.</span>

<span style="font-size: 14px;">**Replaced by gated architectures.** The LSTM (1997) introduced gated cell states with additive updates. The GRU (2014) simplified to two gates. Transformers (2017) replaced recurrence with self-attention entirely. The Vanilla RNN remains pedagogically important but is rarely used in production.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Wrong weight initialization scale.** Using $\sqrt{2 / n_{\text{in}}}$ (He) instead of $\sqrt{2 / (n_{\text{in}} + n_{\text{out}})}$ (Xavier). For $W_{xh}$ with $D = 128$, $H = 256$: Xavier gives $\sqrt{2/384} = 0.0722$, He gives $\sqrt{2/128} = 0.125$. The 73% larger He scale pushes $\tanh$ activations toward saturation from the first forward pass, producing near-zero gradients before training begins.</span>

* <span style="font-size: 14px;">**Forgetting the output projection.** Returning hidden states directly instead of applying $y_t = h_t W_{hy}^T + b_y$. When $H \neq O$, this causes a shape mismatch. When $H = O$, it runs silently but produces wrong values because raw hidden states (bounded to $(-1, 1)$) differ from projected logits.</span>

* <span style="font-size: 14px;">**Returning $y_{\text{seq}}$ alone instead of $(y_{\text{seq}}, h_{\text{final}})$.** The final hidden state is essential for initializing the next chunk in long sequences, serving as a sequence encoding for classification, or initializing a decoder in seq2seq models.</span>

* <span style="font-size: 14px;">**Xavier vs He confusion.** Xavier targets $\tanh$/sigmoid with $\sqrt{2 / (n_{\text{in}} + n_{\text{out}})}$. He targets ReLU with $\sqrt{2 / n_{\text{in}}}$, compensating for ReLU zeroing half the activations. Since the Vanilla RNN uses $\tanh$, Xavier is correct.</span>

* <span style="font-size: 14px;">**Wrong loop direction.** The recurrence must iterate $t = 0$ to $T - 1$ in order, since $h_t$ depends on $h_{t-1}$. Iterating in reverse or processing all steps in parallel breaks the causal chain and produces states that do not encode history.</span>

* <span style="font-size: 14px;">**Incorrect reshaping for batched projection.** Reshaping $H_{\text{all}}$ from $(B, T, H)$ to $(B \cdot T, H)$ must use row-major (C) order. Fortran order or transposing before flattening scrambles time steps, assigning hidden states to wrong output positions.</span>

---
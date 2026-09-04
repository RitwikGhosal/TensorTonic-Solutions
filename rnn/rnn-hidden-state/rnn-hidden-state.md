# <span style="font-size: 20px;">Hidden State Initialization</span>

<span style="font-size: 14px;">The hidden state is the memory vector of a recurrent neural network. At the start of every sequence, this vector must be initialized before the network can process its first token. Elman (1990) introduced the concept of "context units" that carry activations from one time step to the next, and the standard practice is to initialize these units to zeros. This zero initialization ensures deterministic, bias-free behavior at the beginning of every sequence.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">The hidden state $h_t \in \mathbb{R}^H$ is a fixed-size vector that an RNN maintains and updates at every time step. It acts as the network's short-term memory, encoding a compressed summary of all inputs processed so far. Before the first time step, there is no "previous" hidden state, so the network needs an initial value $h_0$.</span>

<span style="font-size: 14px;">**Hidden state initialization** is the process of creating this vector $h_0$. The standard choice is to set every element to zero: $h_0 = \mathbf{0} \in \mathbb{R}^{B \times H}$, where $B$ is the batch size and $H$ is the hidden dimension. This produces a zero tensor with the correct shape, ensuring the first recurrence step has a well-defined input with no prior context assumptions.</span>

<span style="font-size: 14px;">The hidden state is not a learned parameter. It is a **runtime variable** that changes at every time step and resets at the start of every new sequence. Every subsequent state $h_1, h_2, \ldots, h_T$ depends on $h_0$ through the recurrence, so the choice of initialization propagates through the entire sequence.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

### <span style="font-size: 14px;">Initialization</span>

<span style="font-size: 14px;">The initial hidden state is a tensor of zeros matching the batch and hidden dimensions:</span>

$$
h_0 = \mathbf{0} \in \mathbb{R}^{B \times H}
$$

<span style="font-size: 14px;">where $B$ is the number of sequences in the batch and $H$ is the number of hidden units.</span>

### <span style="font-size: 14px;">Recurrence</span>

<span style="font-size: 14px;">At each subsequent time step $t = 1, 2, \ldots, T$, the hidden state is updated by combining the current input with the previous hidden state:</span>

$$
h_t = \tanh(x_t W_{xh} + h_{t-1} W_{hh} + b)
$$

<span style="font-size: 14px;">where:</span>

* <span style="font-size: 14px;">$x_t \in \mathbb{R}^{B \times D}$ is the input at time step $t$ ($D$ is the input feature dimension)</span>
* <span style="font-size: 14px;">$W_{xh} \in \mathbb{R}^{D \times H}$ is the input-to-hidden weight matrix</span>
* <span style="font-size: 14px;">$W_{hh} \in \mathbb{R}^{H \times H}$ is the hidden-to-hidden (recurrent) weight matrix</span>
* <span style="font-size: 14px;">$b \in \mathbb{R}^H$ is the bias vector, broadcast across the batch</span>
* <span style="font-size: 14px;">$\tanh$ is applied element-wise, squashing each value to $(-1, 1)$</span>

<span style="font-size: 14px;">The critical observation is that $h_0$ enters $h_1$ through the term $h_0 W_{hh}$. When $h_0 = \mathbf{0}$, this term vanishes, so $h_1 = \tanh(x_1 W_{xh} + b)$. The first step depends only on the first input and the learned parameters, with no contribution from prior context.</span>

---

## <span style="font-size: 16px;">The Hidden State as Memory</span>

<span style="font-size: 14px;">At every time step, $h_t$ encodes a summary of all inputs from $x_1$ through $x_t$. This is not a lossless recording. The hidden state has fixed size $H$ regardless of how many steps have elapsed, so it must compress arbitrarily long histories into $H$ floating-point numbers. This compression is **lossy** by necessity: after 10 steps or 1000 steps, the representation is still the same $H$ values.</span>

<span style="font-size: 14px;">The hidden state acts as a **running summary** that the network refines at each step. When the network reads a new token, it folds the new information into the existing summary through the nonlinear recurrence rather than appending to a growing list. The weight matrices $W_{xh}$ and $W_{hh}$ learn which features of the input and which aspects of the previous summary matter for the task. Information from early steps gets progressively overwritten, with the tanh nonlinearity and weight magnitudes determining what is preserved.</span>

<span style="font-size: 14px;">This fixed-size representation is both the strength and limitation of vanilla RNNs. The strength is computational efficiency: each step takes $O(H^2 + HD)$ operations regardless of sequence length. The limitation is that important early information can fade as the sequence grows, closely related to the vanishing gradient problem.</span>

---

## <span style="font-size: 16px;">Why Zero Initialization</span>

<span style="font-size: 14px;">Initializing $h_0$ to zeros is the standard choice for several reasons, each rooted in practical considerations about training stability and reproducibility.</span>

<span style="font-size: 14px;">**No prior context assumption.** A zero vector carries no information. At the start of a sequence, the network has no context, so a zero hidden state honestly represents this absence. Any non-zero initialization would inject artificial "prior knowledge" into the first time step, biasing processing in a direction that may not match the data.</span>

<span style="font-size: 14px;">**Deterministic behavior.** Given the same input sequence and model parameters, a zero-initialized hidden state produces identical outputs every time. This reproducibility is essential for debugging, testing, and comparing models. Random initialization would make the same input produce different outputs on every forward pass.</span>

<span style="font-size: 14px;">**Gradient simplicity.** When $h_0$ is a constant (zeros), it is not a learnable parameter and requires no gradient update. This simplifies the computation graph. Gradients still flow through $h_0$ to reach $W_{hh}$ and $W_{xh}$, but $h_0$ itself needs no update step.</span>

<span style="font-size: 14px;">**Interaction with tanh.** The $\tanh$ activation is centered at zero with its steepest gradient at zero ($\tanh'(0) = 1$). When $h_0 = \mathbf{0}$, the pre-activation at the first step is $x_1 W_{xh} + b$, operating in the region where $\tanh$ is most sensitive. The network can make full use of its dynamic range from the very first step, rather than starting in a saturated regime where gradients are small.</span>

<span style="font-size: 14px;">**Alternative initializations.** While zero is standard, two alternatives are occasionally used:</span>

* <span style="font-size: 14px;">**Random initialization:** $h_0 \sim \mathcal{N}(0, 0.01)$. Introduces non-determinism and is rarely used because early training steps quickly push the hidden state away from any initial value.</span>
* <span style="font-size: 14px;">**Learned initialization:** $h_0$ is treated as a trainable parameter updated via gradient descent. This can improve performance on tasks where the beginning of the sequence has consistent structure (e.g., language models always starting with a BOS token). Adds $H$ parameters to the model.</span>

---

## <span style="font-size: 16px;">How Information Flows</span>

<span style="font-size: 14px;">The hidden state creates a chain of dependencies threading through the entire sequence:</span>

$$
h_0 \rightarrow h_1 \rightarrow h_2 \rightarrow \cdots \rightarrow h_T
$$

<span style="font-size: 14px;">This is a strictly **sequential dependency chain**. Computing $h_t$ requires $h_{t-1}$, which requires $h_{t-2}$, all the way back to $h_0$. Unlike transformers, which attend to all positions in parallel, the RNN must process tokens one at a time in order.</span>

<span style="font-size: 14px;">**Forward pass.** At each step, the network mixes new input $x_t$ (through $W_{xh}$) with accumulated history $h_{t-1}$ (through $W_{hh}$). The relative influence of new input versus old memory depends on the learned weight magnitudes. If $W_{hh}$ has large eigenvalues, old information persists strongly. If $W_{xh}$ dominates, each step is driven primarily by the current input.</span>

<span style="font-size: 14px;">**Backward pass (BPTT).** Gradients flow backward through the same chain, passing through the Jacobians $\frac{\partial h_{t+1}}{\partial h_t}$ at every step. When these Jacobians have spectral norms below 1, gradients vanish exponentially. When they exceed 1, gradients explode. The initialization $h_0 = \mathbf{0}$ is the anchor point of this chain: every subsequent state is a deterministic function of $h_0$, the inputs, and the parameters.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Jeffrey Elman introduced recurrent hidden states in his 1990 paper "Finding Structure in Time." The paper proposed the **Simple Recurrent Network (SRN)**, also known as the Elman network, which added "context units" to a feedforward architecture. These context units are exactly what we now call the hidden state.</span>

<span style="font-size: 14px;">In Elman's formulation, the context units receive a copy of the hidden layer activations from the previous time step. The paper states: "The context units provide a simple type of memory. The activations of the hidden units at time $t-1$ provide input to the hidden units at time $t$." This one-step copy mechanism is the recurrence that transforms a static feedforward network into a dynamic sequence processor, creating the temporal dependency chain.</span>

<span style="font-size: 14px;">Elman demonstrated that this architecture could learn temporal structure from data. His experiments included predicting the next item in sequences with grammatical structure and discovering word categories without explicit labels. The hidden state representations revealed that the network induced grammatical categories purely from word co-occurrence statistics.</span>

<span style="font-size: 14px;">The SRN departed from the prevailing approach of explicit temporal windows. Instead of telling the network how much history to consider, the recurrent hidden state lets the network learn its own memory policy. Elman noted that "the notion of time is reduced to the effect that prior events have on current processing," meaning the network has no direct access to the past, only the compressed summary in the hidden state.</span>

---

## <span style="font-size: 16px;">The Hidden State Bottleneck</span>

<span style="font-size: 14px;">The hidden dimension $H$ is fixed before training, creating a fundamental bottleneck: $H$ numbers must represent accumulated information from sequences of arbitrary length. Whether the network has processed 5 tokens or 5000, the summary is always $H$ values.</span>

<span style="font-size: 14px;">**Information grows, capacity does not.** A sequence of $T$ tokens from vocabulary size $V$ carries $T \log_2 V$ bits. As $T$ grows, input information grows linearly while hidden state capacity stays constant at $32H$ bits. The network must learn to discard irrelevant information and retain only what matters.</span>

<span style="font-size: 14px;">**Temporal decay.** Information from early steps decays through repeated nonlinear transformations. Each recurrence mixes old and new information, with tanh bounding values to $(-1, 1)$. After many steps, early input contributions become negligible. This is why vanilla RNNs struggle with long-range dependencies.</span>

<span style="font-size: 14px;">**Why not increase $H$?** $W_{hh}$ has $H^2$ parameters and each step costs $O(H^2)$. Doubling $H$ quadruples both. Typical values are 128 to 1024, with diminishing returns because vanishing gradients limit effective memory regardless of $H$.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a batch of $B = 2$ sequences processed by an RNN with hidden dimension $H = 3$ and input dimension $D = 2$.</span>

### <span style="font-size: 14px;">Step 1: Create the Initial Hidden State</span>

<span style="font-size: 14px;">The initial hidden state is a zero tensor of shape $(B, H) = (2, 3)$:</span>

$$
h_0 = \begin{pmatrix} 0.0 & 0.0 & 0.0 \\ 0.0 & 0.0 & 0.0 \end{pmatrix}
$$

<span style="font-size: 14px;">Row 0 is the first sequence in the batch, row 1 the second. Every value is zero because neither sequence has been processed yet.</span>

### <span style="font-size: 14px;">Step 2: Define the First Input and Weights</span>

<span style="font-size: 14px;">The input at $t = 1$ has shape $(B, D) = (2, 2)$:</span>

$$
x_1 = \begin{pmatrix} 0.5 & -0.3 \\ 0.8 & 0.1 \end{pmatrix}
$$

<span style="font-size: 14px;">The input-to-hidden weight matrix and bias:</span>

$$
W_{xh} = \begin{pmatrix} 0.2 & -0.4 & 0.3 \\ 0.1 & 0.5 & -0.2 \end{pmatrix}, \quad b = \begin{pmatrix} 0.1 \\ -0.1 \\ 0.0 \end{pmatrix}
$$

<span style="font-size: 14px;">The recurrent weight matrix $W_{hh} \in \mathbb{R}^{3 \times 3}$ exists but does not matter for this step because $h_0 W_{hh} = \mathbf{0}$.</span>

### <span style="font-size: 14px;">Step 3: Compute $h_1$</span>

<span style="font-size: 14px;">Since $h_0 = \mathbf{0}$, the recurrent term $h_0 W_{hh}$ is a zero matrix. The computation reduces to:</span>

$$
h_1 = \tanh(x_1 W_{xh} + b)
$$

<span style="font-size: 14px;">**Batch element 0** ($x_1 = [0.5, -0.3]$). Compute $x_1 W_{xh}$:</span>

* <span style="font-size: 14px;">Column 0: $0.5(0.2) + (-0.3)(0.1) = 0.10 - 0.03 = 0.07$</span>
* <span style="font-size: 14px;">Column 1: $0.5(-0.4) + (-0.3)(0.5) = -0.20 - 0.15 = -0.35$</span>
* <span style="font-size: 14px;">Column 2: $0.5(0.3) + (-0.3)(-0.2) = 0.15 + 0.06 = 0.21$</span>

<span style="font-size: 14px;">Adding bias: $[0.07 + 0.1, -0.35 - 0.1, 0.21 + 0.0] = [0.17, -0.45, 0.21]$</span>

<span style="font-size: 14px;">Applying tanh: $[\tanh(0.17), \tanh(-0.45), \tanh(0.21)] = [0.1685, -0.4219, 0.2070]$</span>

<span style="font-size: 14px;">**Batch element 1** ($x_1 = [0.8, 0.1]$). Same procedure yields pre-activation $[0.27, -0.37, 0.22]$, then $\tanh$: $[0.2638, -0.3537, 0.2165]$.</span>

<span style="font-size: 14px;">The resulting first hidden state:</span>

$$
h_1 = \begin{pmatrix} 0.1685 & -0.4219 & 0.2070 \\ 0.2638 & -0.3537 & 0.2165 \end{pmatrix}
$$

<span style="font-size: 14px;">Because $h_0$ was zeros, the two batch elements produced different hidden states purely from their different inputs. If $h_0$ had been non-zero, the recurrent term $h_0 W_{hh}$ would have added the same offset to both, blending in artificial prior state unrelated to the actual input data.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Wrong shape for $h_0$.** The initial hidden state must have shape $(B, H)$, not $(H,)$ or $(1, H)$. A missing batch dimension causes the matrix addition to fail or silently broadcast incorrectly. Always match the batch size of the input.</span>

* <span style="font-size: 14px;">**Wrong dtype.** The hidden state must have the same floating-point dtype as the model weights (typically `float32`). Creating $h_0$ as an integer tensor of zeros will cause type mismatch errors during the matrix multiplication $h_0 W_{hh}$. Use the appropriate floating-point zero constructor (e.g., `torch.zeros` defaults to `float32`, but `np.zeros` defaults to `float64`).</span>

* <span style="font-size: 14px;">**Non-zero initialization causing inconsistency.** If $h_0$ is initialized with random values, the same input sequence produces different outputs on different runs, breaking deterministic testing. Unless there is a specific reason (such as a learned $h_0$), always use zeros.</span>

* <span style="font-size: 14px;">**Forgetting the batch dimension.** A common mistake is creating $h_0$ with shape $(H,)$ for a single sequence and then failing to add the batch dimension when batching. The hidden state is a **per-sequence** quantity: each sequence in the batch has its own independent hidden state trajectory. The batch dimension is not optional.</span>

* <span style="font-size: 14px;">**Confusing $h$ with cell state $c$ in LSTMs.** Vanilla RNNs have only one state vector $h$. LSTMs have two: $h$ and cell state $c$. When moving from RNN to LSTM code, forgetting to initialize $c_0$ alongside $h_0$ causes errors. When implementing a vanilla RNN, the hidden state is the only recurrent variable.</span>

* <span style="font-size: 14px;">**Not resetting $h_0$ between sequences.** During training, the hidden state should be reset to zeros at the start of each new batch (unless doing truncated BPTT). Carrying $h_T$ from one batch as $h_0$ for the next creates inter-batch dependencies that prevent shuffling and cause gradient issues.</span>

---
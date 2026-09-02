# <span style="font-size: 20px;">Complete LSTM Network</span>

<span style="font-size: 14px;">The Long Short-Term Memory (LSTM) network, introduced by Hochreiter and Schmidhuber in "Long Short-Term Memory" (1997), processes sequential data by unrolling a single LSTM cell across $T$ time steps. At each step, the cell receives the current input $x_t$, the previous hidden state $h_{t-1}$, and the previous cell state $C_{t-1}$, producing updated states $(h_t, C_t)$. After iterating over the full sequence, a linear output projection maps each hidden state to the output space. The paper showed that this architecture "can learn to bridge minimal time lags in excess of 1000 discrete time steps."</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">A complete LSTM network wraps a single LSTM cell inside a loop that iterates over a sequence of length $T$. The cell itself contains four gates (forget, input, candidate, output) that regulate information flow. The network adds two components on top of the raw cell: initialization of both state vectors $h_0$ and $C_0$ to zeros, and an output projection layer that maps each hidden state $h_t \in \mathbb{R}^H$ to an output $y_t \in \mathbb{R}^{O}$ through a learned affine transformation.</span>

<span style="font-size: 14px;">The distinction between a single LSTM cell and a complete LSTM network is that the cell defines one step of the recurrence, while the network orchestrates the full sequence. The cell is stateless: given $(x_t, h_{t-1}, C_{t-1})$, it produces $(h_t, C_t)$. The network manages state initialization, the iteration loop, collecting outputs at each step, and the final projection.</span>

<span style="font-size: 14px;">Two state vectors propagate through time. The hidden state $h_t$ serves as the cell's output and is visible to downstream layers. The cell state $C_t$ is the internal memory, shielded by the output gate, never directly exposed outside the cell. Both must be carried from step to step; forgetting to propagate either one breaks the recurrence.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">Let $x_t \in \mathbb{R}^D$ be the input at time step $t$, $h_{t-1} \in \mathbb{R}^H$ the previous hidden state, and $C_{t-1} \in \mathbb{R}^H$ the previous cell state. The LSTM cell computes four gates, then the output projection follows.</span>

<span style="font-size: 14px;">**Equation 1 -- Forget gate.** Decides what to erase from the cell state:</span>

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

<span style="font-size: 14px;">**Equation 2 -- Input gate.** Decides what new information to store:</span>

$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
$$

<span style="font-size: 14px;">**Equation 3 -- Candidate cell state.** Proposes new values to add:</span>

$$
\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)
$$

<span style="font-size: 14px;">**Equation 4 -- Cell state update.** Combines forgetting old and writing new:</span>

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

<span style="font-size: 14px;">**Equation 5 -- Output gate.** Controls what the cell exposes:</span>

$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
$$

<span style="font-size: 14px;">**Equation 6 -- Hidden state.** Filters the cell state through the output gate:</span>

$$
h_t = o_t \odot \tanh(C_t)
$$

<span style="font-size: 14px;">**Equation 7 -- Output projection.** Maps hidden state to output dimension:</span>

$$
y_t = W_y \cdot h_t + b_y
$$

<span style="font-size: 14px;">Here $W_f, W_i, W_c, W_o \in \mathbb{R}^{H \times (H+D)}$ are the gate weight matrices, $b_f, b_i, b_c, b_o \in \mathbb{R}^H$ are the gate biases, $W_y \in \mathbb{R}^{O \times H}$ is the projection weight, and $b_y \in \mathbb{R}^{O}$ is the projection bias. Equations 1-6 execute inside the cell at each time step. Equation 7 is applied after the cell, once per step.</span>

---

## <span style="font-size: 16px;">The Unrolling Loop</span>

<span style="font-size: 14px;">The core of the network is a loop that iterates $T$ times, once per time step. Before the loop, both state vectors are initialized: $h_0 = \mathbf{0} \in \mathbb{R}^H$ and $C_0 = \mathbf{0} \in \mathbb{R}^H$. The input is a tensor $X \in \mathbb{R}^{T \times D}$ (single sample) or $X \in \mathbb{R}^{B \times T \times D}$ (batch of $B$ samples).</span>

<span style="font-size: 14px;">At each iteration $t = 1, 2, \ldots, T$, the loop extracts $x_t$, feeds it with $h_{t-1}$ and $C_{t-1}$ into the LSTM cell, and receives $h_t$ and $C_t$. The hidden state $h_t$ is appended to a list. After the loop, the list $[h_1, h_2, \ldots, h_T]$ is stacked into a tensor of shape $(T, H)$ or $(B, T, H)$.</span>

<span style="font-size: 14px;">The pseudocode:</span>

<span style="font-size: 14px;">1. Initialize $h = \mathbf{0}^H$, $C = \mathbf{0}^H$</span>

<span style="font-size: 14px;">2. For $t = 1$ to $T$: compute $(h, C) = \text{LSTMCell}(x_t, h, C)$, append $h$ to outputs</span>

<span style="font-size: 14px;">3. Stack outputs, apply $y_t = W_y \cdot h_t + b_y$ to each</span>

<span style="font-size: 14px;">4. Return (outputs, $h_T$, $C_T$)</span>

<span style="font-size: 14px;">After the loop, $h$ holds $h_T$ (the last hidden state) and $C$ holds $C_T$ (the last cell state). The function returns both alongside the projected outputs, because downstream tasks often need the final states. In sequence-to-sequence models, the encoder's final $(h_T, C_T)$ initializes the decoder.</span>

---

## <span style="font-size: 16px;">Output Projection</span>

<span style="font-size: 14px;">The output projection transforms each hidden state $h_t \in \mathbb{R}^H$ to $y_t \in \mathbb{R}^O$ via $y_t = W_y \cdot h_t + b_y$. This decouples the hidden dimension $H$ from the output dimension $O$. Without it, the network output dimension would be locked to the hidden size.</span>

<span style="font-size: 14px;">In language modeling, $O$ equals the vocabulary size and $y_t$ produces logits for softmax. In regression, $O$ might be 1. In sequence tagging, $O$ equals the number of tags. The projection uses the same shared $W_y$ and $b_y$ at every time step.</span>

<span style="font-size: 14px;">A critical point: the projection applies to $h_t$, not $C_t$. The cell state is unbounded and represents raw internal memory. The hidden state $h_t = o_t \odot \tanh(C_t)$ is the gated, bounded version the cell has chosen to expose. Projecting $C_t$ directly would bypass the output gate, defeating the LSTM's gating mechanism.</span>

---

## <span style="font-size: 16px;">Maintaining Two States</span>

<span style="font-size: 14px;">The LSTM's defining feature is its dual-state architecture. Both $h_t$ and $C_t$ propagate through time, but they serve different roles.</span>

<span style="font-size: 14px;">**Cell state $C_t$** is the long-term memory. It flows through time with only element-wise modifications: the forget gate scales it ($f_t \odot C_{t-1}$) and the input gate adds to it ($i_t \odot \tilde{C}_t$). This additive update creates the "constant error carousel": during backpropagation through time, the gradient of $C_t$ with respect to $C_{t-1}$ is $f_t$, and when $f_t \approx 1$, gradients pass through without decay. This is how the LSTM bridges time lags of 1,000+ steps.</span>

<span style="font-size: 14px;">**Hidden state $h_t$** is the short-term output. It is derived via $h_t = o_t \odot \tanh(C_t)$, meaning it is a filtered, bounded view of the memory. The output gate controls which dimensions are exposed. This allows the cell to store information in $C_t$ that is not yet relevant to output. In language modeling, a subject noun stored in the cell state may not need to influence output until the verb appears many steps later.</span>

<span style="font-size: 14px;">Both states must be initialized before the loop. The standard choice is $h_0 = \mathbf{0}$ and $C_0 = \mathbf{0}$. Zero initialization for $C_0$ means no initial memory. Zero initialization for $h_0$ means the first step's gate computations depend only on $x_1$. Some implementations allow learned initial states, but zeros are the default.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Hochreiter and Schmidhuber published "Long Short-Term Memory" in Neural Computation in 1997. The paper was motivated by the vanishing gradient problem: as gradients are backpropagated through many time steps, they shrink to zero or explode to infinity. This makes standard RNNs unable to learn dependencies spanning more than roughly 10-20 steps.</span>

<span style="font-size: 14px;">The key innovation was the constant error carousel (CEC). By designing the cell state update as $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$, gradient flow through $C$ can remain at magnitude 1 when the forget gate is close to 1. The paper demonstrated that LSTM could learn tasks where the relevant input and point of use were separated by over 1,000 time steps.</span>

<span style="font-size: 14px;">The LSTM became the dominant sequence model for two decades. In speech recognition, it powered Google's voice search starting in 2012. In machine translation, LSTM-based sequence-to-sequence models with attention (Bahdanau et al., 2014; Sutskever et al., 2014) were state of the art before Transformers. In time series forecasting, LSTMs remain competitive. The complete LSTM network, with its unrolled cell and output projection, is the standard form used in all of these systems.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider an LSTM with $D = 2$, $H = 2$, $O = 2$, processing $T = 3$ steps. Initialize $h_0 = [0, 0]$, $C_0 = [0, 0]$.</span>

<span style="font-size: 14px;">**Inputs:**</span>

$$
x_1 = \begin{pmatrix} 1.0 \\ 0.5 \end{pmatrix}, \quad x_2 = \begin{pmatrix} -0.3 \\ 0.8 \end{pmatrix}, \quad x_3 = \begin{pmatrix} 0.2 \\ -0.4 \end{pmatrix}
$$

<span style="font-size: 14px;">**Weights** (all gates share the same weights for brevity):</span>

$$
W = \begin{pmatrix} 0.1 & 0.2 & 0.3 & -0.1 \\ -0.2 & 0.1 & 0.1 & 0.4 \end{pmatrix}, \quad b = \begin{pmatrix} 0 \\ 0 \end{pmatrix}, \quad W_y = \begin{pmatrix} 0.5 & -0.3 \\ 0.2 & 0.7 \end{pmatrix}, \quad b_y = \begin{pmatrix} 0.1 \\ -0.1 \end{pmatrix}
$$

<span style="font-size: 14px;">**Step 1 ($t = 1$).** $[h_0, x_1] = [0, 0, 1.0, 0.5]$. Pre-activation: $[0.25, 0.30]$.</span>

<span style="font-size: 14px;">$f_1 = i_1 = o_1 = \sigma([0.25, 0.30]) = [0.5622, 0.5744]$, $\tilde{C}_1 = \tanh([0.25, 0.30]) = [0.2449, 0.2913]$.</span>

<span style="font-size: 14px;">$C_1 = [0, 0] + [0.5622 \times 0.2449, 0.5744 \times 0.2913] = [0.1377, 0.1674]$.</span>

<span style="font-size: 14px;">$h_1 = [0.5622 \times \tanh(0.1377), 0.5744 \times \tanh(0.1674)] = [0.0770, 0.0954]$.</span>

<span style="font-size: 14px;">$y_1 = [0.5(0.0770) - 0.3(0.0954) + 0.1, 0.2(0.0770) + 0.7(0.0954) - 0.1] = [0.1099, -0.0174]$.</span>

<span style="font-size: 14px;">**Step 2 ($t = 2$).** $[h_1, x_2] = [0.0770, 0.0954, -0.3, 0.8]$. Pre-activation: $[-0.1432, 0.2841]$.</span>

<span style="font-size: 14px;">$f_2 = i_2 = o_2 = [0.4643, 0.5706]$, $\tilde{C}_2 = [-0.1424, 0.2768]$.</span>

<span style="font-size: 14px;">$C_2 = [0.4643(0.1377) + 0.4643(-0.1424), 0.5706(0.1674) + 0.5706(0.2768)] = [-0.0022, 0.2536]$.</span>

<span style="font-size: 14px;">$h_2 = [0.4643(-0.0022), 0.5706(0.2487)] = [-0.0010, 0.1419]$.</span>

<span style="font-size: 14px;">$y_2 = [0.5(-0.0010) - 0.3(0.1419) + 0.1, 0.2(-0.0010) + 0.7(0.1419) - 0.1] = [0.0562, -0.0009]$.</span>

<span style="font-size: 14px;">**Step 3 ($t = 3$).** $[h_2, x_3] = [-0.0010, 0.1419, 0.2, -0.4]$. Pre-activation: $[0.1283, -0.1256]$.</span>

<span style="font-size: 14px;">$f_3 = i_3 = o_3 = [0.5320, 0.4686]$, $\tilde{C}_3 = [0.1277, -0.1250]$.</span>

<span style="font-size: 14px;">$C_3 = [0.5320(-0.0022) + 0.5320(0.1277), 0.4686(0.2536) + 0.4686(-0.1250)] = [0.0668, 0.0603]$.</span>

<span style="font-size: 14px;">$h_3 = [0.5320(0.0667), 0.4686(0.0603)] = [0.0355, 0.0283]$.</span>

<span style="font-size: 14px;">$y_3 = [0.5(0.0355) - 0.3(0.0283) + 0.1, 0.2(0.0355) + 0.7(0.0283) - 0.1] = [0.1093, -0.0727]$.</span>

<span style="font-size: 14px;">**Final return:** outputs $= [[0.1099, -0.0174], [0.0562, -0.0009], [0.1093, -0.0727]]$, $h_{\text{last}} = [0.0355, 0.0283]$, $C_{\text{last}} = [0.0668, 0.0603]$. The hidden state and cell state are different vectors with different magnitudes, confirming they carry distinct information.</span>

---

## <span style="font-size: 16px;">Bidirectional and Stacked LSTMs</span>

<span style="font-size: 14px;">The single-direction, single-layer LSTM above is the basic building block. Two extensions are standard.</span>

<span style="font-size: 14px;">**Bidirectional LSTM.** Two separate LSTMs process the same sequence: one forward ($t = 1$ to $T$) and one backward ($t = T$ to $1$). Each has independent weights. At each step, hidden states are concatenated: $h_t^{\text{bi}} = [h_t^{\text{fwd}}, h_t^{\text{bwd}}] \in \mathbb{R}^{2H}$. The output projection must use $W_y \in \mathbb{R}^{O \times 2H}$. Bidirectional LSTMs apply when the full sequence is available at inference, such as named entity recognition. They cannot be used in autoregressive generation where future tokens are unknown.</span>

<span style="font-size: 14px;">**Stacked LSTM.** Multiple LSTM layers are stacked vertically. The outputs $[h_1^{(l)}, \ldots, h_T^{(l)}]$ of layer $l$ become the inputs to layer $l+1$. Each layer has its own gate weights. The input dimension of layer $l+1$ equals $H^{(l)}$. Stacking 2-4 layers is common. The output projection applies only after the final layer. Dropout between layers (not within the recurrence) regularizes deep stacks.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting to propagate $C$ alongside $h$.** The most common mistake is updating $h$ at each step but not carrying $C$ forward. Both states must be passed from step $t$ to step $t+1$. If $C$ is not updated, the forget and input gates operate on stale or zero cell states, destroying the long-term memory mechanism. The constant error carousel requires a continuous chain of $C$ values.</span>

* <span style="font-size: 14px;">**Wrong initialization.** Both $h_0$ and $C_0$ must be initialized. Leaving them uninitialized produces nondeterministic outputs. The standard is zero vectors of shape $(H,)$ or $(B, H)$ for batched inputs. Using ones or other non-zero values changes the first step's gate activations unpredictably.</span>

* <span style="font-size: 14px;">**Confusing $h_{\text{last}}$ with $C_{\text{last}}$.** The function returns both. $h_T$ is the filtered output at the final step. $C_T$ is the raw internal memory. These are different vectors with different magnitudes and meanings. Swapping them when initializing a decoder corrupts the sequence-to-sequence pipeline.</span>

* <span style="font-size: 14px;">**Applying output projection to $C_t$ instead of $h_t$.** The projection $y_t = W_y \cdot h_t + b_y$ must use the hidden state. The cell state is unbounded and unfiltered. The hidden state has been gated by $o_t$ and squashed by $\tanh$, making it the appropriate representation for downstream use.</span>

* <span style="font-size: 14px;">**Dimension mismatch in the output projection.** $W_y$ must have shape $(O, H)$, not $(O, H+D)$ or $(H, O)$. The projection input is $h_t \in \mathbb{R}^H$, not the concatenation $[h_t, x_t]$. Using the wrong shape either crashes or silently produces incorrect outputs.</span>

* <span style="font-size: 14px;">**Not collecting outputs at each step.** The network must return a projected output for every time step, not just the last. Returning only $y_T$ discards intermediate outputs and makes the network useless for sequence labeling tasks where predictions at each position are needed.</span>

---
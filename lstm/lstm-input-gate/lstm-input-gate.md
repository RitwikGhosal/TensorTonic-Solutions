# <span style="font-size: 20px;">Input Gate and Candidate Memory</span>

<span style="font-size: 14px;">The input gate and candidate memory form the "write" mechanism of the LSTM cell. While the forget gate decides what to erase from memory, the input gate and candidate together decide what new information to store. They are two separate computations: the input gate produces a vector in $[0, 1]$ controlling **how much** to write, and the candidate produces a vector in $[-1, 1]$ proposing **what** to write. Their element-wise product determines the actual update added to the cell state.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">The input gate and candidate memory are a **two-part mechanism** that together determine the new information written into the LSTM cell state at each time step. The mechanism is split into two components because "what to write" and "how much to write" require different mathematical properties:</span>

* <span style="font-size: 14px;">**Input gate $i_t$:** A sigmoid-activated vector in $[0, 1]$. Each element acts as a valve controlling the magnitude of the update for that cell state dimension. A value near 1 means "write fully," near 0 means "block this update."</span>
* <span style="font-size: 14px;">**Candidate memory $\tilde{C}_t$:** A tanh-activated vector in $[-1, 1]$. Each element proposes a new value to add to the cell state. Positive values push the cell state up, negative values push it down.</span>

<span style="font-size: 14px;">Both components receive the **same concatenated input** $[h_{t-1}, x_t]$ but use separate weight matrices and biases. The gate learns to detect **when** information is relevant, while the candidate learns to encode **what** that information is. Their element-wise product $i_t \odot \tilde{C}_t$ produces the update vector added to the cell state after the forget gate has done its work.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">**Step 1 - Concatenate inputs.** The previous hidden state $h_{t-1} \in \mathbb{R}^{h}$ and current input $x_t \in \mathbb{R}^{d}$ are concatenated:</span>

$$
[h_{t-1}, x_t] \in \mathbb{R}^{h + d}
$$

<span style="font-size: 14px;">This concatenation is shared across all LSTM gates. The combined vector carries both recurrent context and fresh input information.</span>

<span style="font-size: 14px;">**Step 2 - Input gate.** The concatenated vector is transformed by a learned weight matrix and bias, then passed through sigmoid:</span>

$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
$$

<span style="font-size: 14px;">where $W_i \in \mathbb{R}^{h \times (h+d)}$ and $b_i \in \mathbb{R}^{h}$. The sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$ squashes each element to $(0, 1)$, making the output interpretable as independent "write intensity" controls.</span>

<span style="font-size: 14px;">**Step 3 - Candidate memory.** The same concatenated vector is transformed by a different weight matrix and bias, then passed through tanh:</span>

$$
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
$$

<span style="font-size: 14px;">where $W_C \in \mathbb{R}^{h \times (h+d)}$ and $b_C \in \mathbb{R}^{h}$. The tanh maps each element to $(-1, 1)$, meaning the candidate can propose both increases and decreases to the cell state.</span>

<span style="font-size: 14px;">**Step 4 - Gated update.** The input gate and candidate combine via element-wise multiplication:</span>

$$
i_t \odot \tilde{C}_t \in \mathbb{R}^{h}
$$

<span style="font-size: 14px;">Each dimension of the candidate is independently scaled by the corresponding gate value. This product is added to the cell state in the full update equation $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$.</span>

---

## <span style="font-size: 16px;">Why Two Separate Components</span>

<span style="font-size: 14px;">A natural question is: why not use a single computation to produce the cell state update? The separation exists because **magnitude control and content generation are fundamentally different tasks** that benefit from different activations and learned representations.</span>

<span style="font-size: 14px;">**The gate controls magnitude.** The input gate $i_t$ operates in $[0, 1]$ and determines the **scale** of the update per dimension. It answers "how much should I modify this dimension of memory right now?" During training, the gate learns to recognize contexts where memory should be updated (gate near 1) versus contexts where the cell state should be left alone (gate near 0).</span>

<span style="font-size: 14px;">**The candidate controls content.** The candidate $\tilde{C}_t$ operates in $[-1, 1]$ and determines the **direction and content** of the update. It encodes the actual semantic information from the current input and context into a form suitable for long-term storage.</span>

<span style="font-size: 14px;">**The element-wise product combines both.** The product $i_t \odot \tilde{C}_t$ allows fine-grained control: one dimension might have a strong candidate value of $0.95$ but a gate value of only $0.1$, meaning the model has encoded useful information but decided now is not the time to write it. Another dimension might have a moderate candidate of $0.4$ but a fully open gate of $0.98$.</span>

<span style="font-size: 14px;">If a single computation produced the update directly (say, a single tanh layer), the model would have no mechanism to selectively suppress individual dimensions. It could only learn to produce small values, which conflates "this information is not relevant now" with "this information has small magnitude." The gate-candidate separation disentangles relevance from content.</span>

---

## <span style="font-size: 16px;">Why Tanh for the Candidate</span>

<span style="font-size: 14px;">**Bounded output in $[-1, 1]$.** The tanh function prevents the candidate from proposing unbounded updates that could cause the cell state to explode. Combined with the gate's $[0, 1]$ range, the maximum possible update to any cell state dimension per time step is bounded between $-1$ and $+1$.</span>

<span style="font-size: 14px;">**Zero-centered output.** Unlike sigmoid which outputs in $(0, 1)$ with a mean around $0.5$, tanh is centered at zero. This is essential because the candidate needs to both **increase** and **decrease** the cell state. If the candidate used sigmoid (always positive), the cell state could only grow over time. Zero-centered outputs let the model naturally push cell state dimensions in either direction.</span>

<span style="font-size: 14px;">**Stronger gradients near zero.** The derivative of tanh at zero is 1 (compared to 0.25 for sigmoid). Since many pre-activation values cluster around zero during training, tanh provides stronger gradient signal for the candidate's weight updates, helping the model learn useful representations faster.</span>

<span style="font-size: 14px;">**Contrast with sigmoid for gates.** Gates use sigmoid because they need to be in $[0, 1]$ to function as multiplicative switches. A gate value of 0 means "block completely" and 1 means "pass completely." Tanh values of $-1$ and $+1$ have no such interpretation for a gate. The functions are chosen to match the mathematical role: sigmoid for gating (scaling), tanh for content (representation).</span>

---

## <span style="font-size: 16px;">Paper Context: Hochreiter and Schmidhuber (1997)</span>

<span style="font-size: 14px;">The input gate was introduced in the original LSTM paper, "Long Short-Term Memory" by Sepp Hochreiter and Jurgen Schmidhuber (1997). The paper describes the input gate's role with a specific framing: **"The input gate protects the memory contents from perturbation by irrelevant inputs."** This frames the gate as a **protective mechanism**, not merely a write controller.</span>

<span style="font-size: 14px;">The core problem being solved was the **vanishing gradient problem** in recurrent neural networks. Standard RNNs could not learn long-range dependencies because gradient signals decayed exponentially through time. The LSTM's solution was the **constant error carousel (CEC)**: a cell state updated additively rather than multiplicatively, allowing gradients to flow unchanged across many time steps.</span>

<span style="font-size: 14px;">The input gate is essential to making the CEC work. Without it, every input at every time step would perturb the cell state, drowning out stored long-term information. The gate acts as a bouncer: it learns which inputs carry information worth storing and blocks everything else. The paper's use of "protection" emphasizes that the default behavior should be to **not write**. The cell state is precious, and the input gate must actively decide to open before any modification occurs.</span>

<span style="font-size: 14px;">The candidate memory (called "cell input" in the original paper) represents the transformation of current input into a form suitable for storage. The original paper used a squashing function $g$ for this role, which in modern implementations is tanh.</span>

<span style="font-size: 14px;">Notably, the original 1997 LSTM did **not** include a forget gate. The cell state could only be written to (via input gate) and read from (via output gate), but old information could never be explicitly erased. The forget gate was added by Gers et al. (2000). In the original architecture, the input gate was even more critical: once information entered the cell state, it stayed indefinitely, so the gate had to be highly selective because there was no mechanism to undo a bad write.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider an LSTM with hidden size $h = 2$ and input size $d = 2$.</span>

<span style="font-size: 14px;">**Given values:**</span>

$$
h_{t-1} = \begin{pmatrix} 0.5 \\ -0.3 \end{pmatrix}, \quad x_t = \begin{pmatrix} 0.8 \\ 0.1 \end{pmatrix}
$$

$$
W_i = \begin{pmatrix} 0.4 & -0.2 & 0.3 & 0.1 \\ 0.1 & 0.5 & -0.1 & 0.6 \end{pmatrix}, \quad b_i = \begin{pmatrix} 0.1 \\ -0.2 \end{pmatrix}
$$

$$
W_C = \begin{pmatrix} -0.3 & 0.7 & 0.2 & -0.4 \\ 0.6 & -0.1 & 0.5 & 0.3 \end{pmatrix}, \quad b_C = \begin{pmatrix} 0.0 \\ 0.1 \end{pmatrix}
$$

<span style="font-size: 14px;">**Step 1 - Concatenate:** $[h_{t-1}, x_t] = (0.5, -0.3, 0.8, 0.1)^T$</span>

<span style="font-size: 14px;">**Step 2 - Input gate pre-activation:**</span>

<span style="font-size: 14px;">Dim 1: $(0.4)(0.5) + (-0.2)(-0.3) + (0.3)(0.8) + (0.1)(0.1) + 0.1 = 0.2 + 0.06 + 0.24 + 0.01 + 0.1 = 0.61$</span>

<span style="font-size: 14px;">Dim 2: $(0.1)(0.5) + (0.5)(-0.3) + (-0.1)(0.8) + (0.6)(0.1) + (-0.2) = 0.05 - 0.15 - 0.08 + 0.06 - 0.2 = -0.32$</span>

<span style="font-size: 14px;">**Step 3 - Apply sigmoid:**</span>

$$
i_t = \begin{pmatrix} \sigma(0.61) \\ \sigma(-0.32) \end{pmatrix} = \begin{pmatrix} 0.648 \\ 0.421 \end{pmatrix}
$$

<span style="font-size: 14px;">Dimension 1 allows about 65% of the candidate through. Dimension 2 allows about 42%.</span>

<span style="font-size: 14px;">**Step 4 - Candidate pre-activation:**</span>

<span style="font-size: 14px;">Dim 1: $(-0.3)(0.5) + (0.7)(-0.3) + (0.2)(0.8) + (-0.4)(0.1) + 0.0 = -0.15 - 0.21 + 0.16 - 0.04 = -0.24$</span>

<span style="font-size: 14px;">Dim 2: $(0.6)(0.5) + (-0.1)(-0.3) + (0.5)(0.8) + (0.3)(0.1) + 0.1 = 0.3 + 0.03 + 0.4 + 0.03 + 0.1 = 0.86$</span>

<span style="font-size: 14px;">**Step 5 - Apply tanh:**</span>

$$
\tilde{C}_t = \begin{pmatrix} \tanh(-0.24) \\ \tanh(0.86) \end{pmatrix} = \begin{pmatrix} -0.235 \\ 0.697 \end{pmatrix}
$$

<span style="font-size: 14px;">The candidate proposes a **negative** update for dimension 1 and a **positive** update for dimension 2. This illustrates how tanh's zero-centered output naturally allows both directions.</span>

<span style="font-size: 14px;">**Step 6 - Element-wise product $i_t \odot \tilde{C}_t$:**</span>

$$
i_t \odot \tilde{C}_t = \begin{pmatrix} 0.648 \times (-0.235) \\ 0.421 \times 0.697 \end{pmatrix} = \begin{pmatrix} -0.152 \\ 0.293 \end{pmatrix}
$$

<span style="font-size: 14px;">**Key observations:**</span>

* <span style="font-size: 14px;">**Dimension 1:** The candidate proposed $-0.235$ but the gate allowed 65% through, resulting in $-0.152$. The cell state will decrease by this amount.</span>
* <span style="font-size: 14px;">**Dimension 2:** The candidate proposed $0.697$ but the gate allowed only 42% through, resulting in $0.293$. Despite a stronger candidate, the lower gate value significantly reduced the update.</span>
* <span style="font-size: 14px;">**Independence:** The gate and candidate are independent. Dimension 2 has a strong candidate but weak gate; dimension 1 has a weaker candidate but stronger gate. Content and relevance are separate judgments.</span>

---

## <span style="font-size: 16px;">Connection to GRU</span>

<span style="font-size: 14px;">The Gated Recurrent Unit (GRU), introduced by Cho et al. (2014), has its own candidate computation that differs from the LSTM candidate in an important way.</span>

<span style="font-size: 14px;">**LSTM candidate** uses the **raw** hidden state:</span>

$$
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
$$

<span style="font-size: 14px;">**GRU candidate** uses a **reset-gated** hidden state:</span>

$$
\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t] + b)
$$

<span style="font-size: 14px;">where $r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$ is the reset gate. Before the hidden state enters the candidate computation, each dimension is scaled by the reset gate. When $r_t$ is near 0, the candidate ignores previous state and computes purely from current input, allowing the GRU to "start fresh."</span>

* <span style="font-size: 14px;">**LSTM:** Relies on the forget gate to manage old information and the input gate to control writing. The candidate always sees the full hidden state.</span>
* <span style="font-size: 14px;">**GRU:** Uses the reset gate to filter the hidden state before it enters the candidate. This makes the candidate conditionally independent of history, something the LSTM candidate cannot do.</span>

<span style="font-size: 14px;">The GRU also merges forget and input gates into a single update gate $z_t$, using the complementary relationship $z_t$ and $(1 - z_t)$: whatever fraction of old state is kept, the complement is used for the new candidate. The LSTM treats forgetting and writing as independent decisions with separate gates, which is more expressive but uses more parameters.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Using sigmoid instead of tanh for the candidate.** If the candidate uses sigmoid, its output is in $(0, 1)$ and can never produce a negative update. The cell state can only increase, destroying the model's ability to push dimensions downward. Tanh's $(-1, 1)$ range is essential for bidirectional updates.</span>

* <span style="font-size: 14px;">**Sharing weight matrices between $i_t$ and $\tilde{C}_t$.** The input gate and candidate must have **separate** weight matrices ($W_i$ and $W_C$). Sharing makes gate and candidate deterministic functions of each other, eliminating independent control. The model loses the ability to say "I see important content but now is not the time to write it."</span>

* <span style="font-size: 14px;">**Forgetting these are TWO separate computations returning a tuple.** A common error is treating the input gate as a single operation. You must compute **both** $i_t$ and $\tilde{C}_t$ as separate tensors, then combine them with element-wise multiplication. If your implementation fuses them into a single linear layer followed by one activation, the mechanism is broken.</span>

* <span style="font-size: 14px;">**Incorrect concatenation order.** The convention $[h_{t-1}, x_t]$ means hidden state comes first, then input. If $W_i$ is shaped $h \times (h + d)$, the first $h$ columns multiply $h_{t-1}$ and the last $d$ columns multiply $x_t$. Reversing concatenation without transposing the weight matrix produces wrong results.</span>

* <span style="font-size: 14px;">**Confusing $\tilde{C}_t$ with $C_t$.** The candidate $\tilde{C}_t$ (with tilde) is the **proposed** update. The actual cell state $C_t$ (without tilde) is the result of $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$. These are different tensors at different stages. The input gate problem only asks you to compute $i_t$ and $\tilde{C}_t$, not the final $C_t$.</span>

* <span style="font-size: 14px;">**Applying the gate before the activation.** The correct order is: linear transform, then activation, for each component separately. If you apply the gate to pre-activation values or compute $\sigma(\tanh(W \cdot [h, x]))$, the math is completely different and the mechanism breaks.</span>

---
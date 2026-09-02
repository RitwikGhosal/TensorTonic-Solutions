# <span style="font-size: 20px;">Output Gate and Hidden State</span>

<span style="font-size: 14px;">The output gate is the final gating mechanism in the LSTM cell, introduced in "Long Short-Term Memory" (Hochreiter & Schmidhuber, 1997). It controls which parts of the already-updated cell state $C_t$ are exposed as the hidden state $h_t$, the vector that the rest of the network actually sees. Without the output gate, every piece of information stored in the cell state would be broadcast to downstream layers and the next time step's gate computations, regardless of whether it is relevant right now.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">The output gate and hidden state computation is the last stage of the LSTM cell's forward pass. By this point, the forget gate has decided what to erase from the previous cell state $C_{t-1}$, the input gate has decided what new information to write, and the cell state has been updated to $C_t$. The output gate now determines what fraction of $C_t$ to reveal as the hidden state $h_t$.</span>

<span style="font-size: 14px;">The hidden state $h_t$ serves two roles simultaneously. First, it is passed to whatever layer sits on top of the LSTM -- a classification head, a dense layer, or another recurrent layer in a stacked architecture. Second, it is fed back into the LSTM at the next time step as $h_{t-1}$, where it participates in computing all four gates. The output gate acts as a selective filter between the cell's internal memory and everything external to the cell.</span>

<span style="font-size: 14px;">Hochreiter and Schmidhuber describe this as "protecting other units from currently irrelevant memory contents stored in the cell." The cell state might store information that will be critical ten time steps later but is not useful for the current prediction. The output gate learns to keep that information hidden inside $C_t$ until it becomes relevant.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">Let $x_t \in \mathbb{R}^D$ be the input at time step $t$, $h_{t-1} \in \mathbb{R}^H$ be the previous hidden state, and $C_t \in \mathbb{R}^H$ be the already-updated cell state. The concatenation $[h_{t-1}, x_t] \in \mathbb{R}^{H+D}$ is formed by stacking these two vectors.</span>

<span style="font-size: 14px;">**Equation 1 -- Output gate.** Determines which dimensions of the cell state to expose:</span>

$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
$$

<span style="font-size: 14px;">Here $W_o \in \mathbb{R}^{H \times (H+D)}$ is the output gate weight matrix and $b_o \in \mathbb{R}^H$ is the bias. The sigmoid $\sigma$ squashes each element to $(0, 1)$. A value near $1$ means "expose this dimension," while a value near $0$ means "keep this dimension private inside the cell."</span>

<span style="font-size: 14px;">**Equation 2 -- Hidden state.** Combines the gate with the squashed cell state:</span>

$$
h_t = o_t \odot \tanh(C_t)
$$

<span style="font-size: 14px;">The symbol $\odot$ denotes element-wise (Hadamard) multiplication. The tanh squashes $C_t$ from its potentially unbounded range into $[-1, 1]$, and $o_t$ scales each dimension independently. The resulting $h_t \in \mathbb{R}^H$ is bounded to $[-1, 1]$ because both factors are bounded: $o_t \in (0, 1)$ and $\tanh(C_t) \in (-1, 1)$.</span>

---

## <span style="font-size: 16px;">Why Gate the Output</span>

<span style="font-size: 14px;">The cell state $C_t$ is the LSTM's long-term memory. It can accumulate information over many time steps because the forget gate allows values to persist with minimal modification. But not everything stored in long-term memory is relevant at every single time step. The output gate provides the mechanism to selectively expose only the portions of $C_t$ that matter right now.</span>

<span style="font-size: 14px;">Consider a language model processing "The cat, which had been sleeping on the mat for several hours, finally woke up and stretched." The cell state might store the subject "cat" across the entire subordinate clause. During "which had been sleeping on the mat for several hours," the subject information is stored but not immediately needed for next-word prediction. The output gate keeps those dimensions suppressed. When "finally" arrives and the model needs to predict "woke," the output gate opens those dimensions to expose the stored subject.</span>

<span style="font-size: 14px;">Without the output gate, the hidden state would always be a direct function of the entire cell state. Gate computations at the next time step would be influenced by all stored information, not just the relevant subset. This creates noise in the recurrent signal and makes it harder for the network to learn which patterns to attend to. The output gate adds a learnable bottleneck that filters this signal.</span>

---

## <span style="font-size: 16px;">Why Tanh on the Cell State</span>

<span style="font-size: 14px;">The cell state $C_t$ has no activation function bounding its values. It is updated through an additive process: $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$, where the forget gate $f_t$ scales the old state and the input gate $i_t$ adds new content. Because this is a weighted sum rather than a squashed activation, $C_t$ can grow to arbitrary magnitudes over many time steps. This is intentional -- it is what allows the LSTM to maintain stable gradients through long sequences.</span>

<span style="font-size: 14px;">However, exposing unbounded values as the hidden state would cause problems. Downstream layers receiving $h_t$ would face inputs of unpredictable scale. Gradient magnitudes would vary wildly depending on how large $C_t$ has grown. The tanh maps $C_t$ into $[-1, 1]$ before gating, ensuring $h_t$ remains stable regardless of how many time steps have elapsed or how much information has accumulated.</span>

<span style="font-size: 14px;">The tanh also introduces a useful nonlinearity. Two cell state values that are both very large and positive -- say $C_t = 5$ and $C_t = 50$ -- both map to $\tanh$ values near $1$. The network treats them as "strongly positive" without distinguishing exact magnitudes. This saturation acts as soft clipping that prevents extreme values in one dimension from dominating the hidden state.</span>

---

## <span style="font-size: 16px;">Hidden State vs Cell State</span>

<span style="font-size: 14px;">The LSTM maintains two state vectors at every time step, and understanding their distinct roles is essential.</span>

<span style="font-size: 14px;">**Cell state** $C_t \in \mathbb{R}^H$ is the internal memory. It flows through time with additive updates, shielded by the forget and input gates. Values in $C_t$ can persist for many time steps with minimal degradation because the forget gate can pass them through nearly unchanged ($f_t \approx 1$). The cell state is never directly seen by any layer outside the LSTM cell.</span>

<span style="font-size: 14px;">**Hidden state** $h_t \in \mathbb{R}^H$ is the external interface. It is the only vector that leaves the LSTM cell. It serves two purposes: (1) it is passed to the next layer for predictions or further computation, and (2) it is fed back into the same cell at the next time step as $h_{t-1}$, participating in all four gate computations. The hidden state is a filtered, bounded version of the cell state: $h_t = o_t \odot \tanh(C_t)$.</span>

<span style="font-size: 14px;">This separation is one of the key architectural innovations of the LSTM. In a vanilla RNN, there is only one state vector $h_t$ that must simultaneously serve as long-term memory and short-term output. The LSTM splits these responsibilities: $C_t$ handles long-term storage with additive updates for gradient stability, while $h_t$ handles short-term communication with bounded values for numerical stability. The output gate is the bridge between them.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Hochreiter and Schmidhuber introduced the LSTM in their 1997 paper "Long Short-Term Memory," published in Neural Computation. The central problem the paper addresses is the **vanishing gradient problem**: in standard RNNs, error signals flowing backward through time either decay exponentially (vanishing) or explode exponentially (exploding), making it impossible to learn long-range dependencies.</span>

<span style="font-size: 14px;">The paper's solution is the **Constant Error Carousel (CEC)** -- the cell state's additive update mechanism that allows gradients to flow unchanged through time. The gates (forget, input, output) are the learned controllers that regulate what enters, persists in, and exits this carousel. The output gate was designed to solve what the authors call the "output weight conflict": without it, the cell's output connections would need to simultaneously serve the current prediction and future gate computations.</span>

<span style="font-size: 14px;">The original 1997 architecture did not include the forget gate -- that was added by Gers, Schmidhuber, and Cummins in 2000. The original cell state update was purely additive: $C_t = C_{t-1} + i_t \odot \tilde{C}_t$. The output gate, however, was part of the original design from the start, reflecting its fundamental importance. The paper states that the output gate "protects other units from currently irrelevant memory contents stored in the cell," establishing the principle that internal memory and external communication should be decoupled.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider an LSTM with $H = 3$ (hidden size) and $D = 2$ (input size). Assume the forget gate, input gate, and candidate have already executed, producing the updated cell state $C_t$. The output gate now computes $o_t$ and $h_t$.</span>

<span style="font-size: 14px;">**Given values:**</span>

$$
h_{t-1} = \begin{pmatrix} 0.2 \\ -0.5 \\ 0.1 \end{pmatrix}, \quad x_t = \begin{pmatrix} 0.8 \\ -0.3 \end{pmatrix}, \quad C_t = \begin{pmatrix} 1.5 \\ -0.7 \\ 3.2 \end{pmatrix}
$$

<span style="font-size: 14px;">**Weights and bias:**</span>

$$
W_o = \begin{pmatrix} 0.3 & -0.1 & 0.4 & 0.2 & -0.5 \\ 0.1 & 0.6 & -0.2 & 0.3 & 0.1 \\ -0.4 & 0.2 & 0.5 & -0.1 & 0.3 \end{pmatrix}, \quad b_o = \begin{pmatrix} 0.1 \\ -0.2 \\ 0.0 \end{pmatrix}
$$

<span style="font-size: 14px;">**Step 1: Concatenate.** $[h_{t-1}, x_t] = [0.2, -0.5, 0.1, 0.8, -0.3]$</span>

<span style="font-size: 14px;">**Step 2: Compute the pre-activation.** $W_o \cdot [h_{t-1}, x_t] + b_o$:</span>

<span style="font-size: 14px;">Row 1: $0.3(0.2) + (-0.1)(-0.5) + 0.4(0.1) + 0.2(0.8) + (-0.5)(-0.3) + 0.1$</span>

<span style="font-size: 14px;">$= 0.06 + 0.05 + 0.04 + 0.16 + 0.15 + 0.1 = 0.56$</span>

<span style="font-size: 14px;">Row 2: $0.1(0.2) + 0.6(-0.5) + (-0.2)(0.1) + 0.3(0.8) + 0.1(-0.3) + (-0.2)$</span>

<span style="font-size: 14px;">$= 0.02 - 0.30 - 0.02 + 0.24 - 0.03 - 0.2 = -0.29$</span>

<span style="font-size: 14px;">Row 3: $(-0.4)(0.2) + 0.2(-0.5) + 0.5(0.1) + (-0.1)(0.8) + 0.3(-0.3) + 0.0$</span>

<span style="font-size: 14px;">$= -0.08 - 0.10 + 0.05 - 0.08 - 0.09 + 0.0 = -0.30$</span>

<span style="font-size: 14px;">**Step 3: Apply sigmoid to get $o_t$.**</span>

<span style="font-size: 14px;">$o_t = [\sigma(0.56), \sigma(-0.29), \sigma(-0.30)] = [0.6365, 0.4280, 0.4256]$</span>

<span style="font-size: 14px;">**Step 4: Apply tanh to $C_t$.**</span>

<span style="font-size: 14px;">$\tanh(C_t) = [\tanh(1.5), \tanh(-0.7), \tanh(3.2)] = [0.9051, -0.6044, 0.9967]$</span>

<span style="font-size: 14px;">Note that $C_t[2] = 3.2$ maps to $0.9967$ -- nearly saturated at $1$. This illustrates the squashing effect: a large cell state value is compressed into the bounded range.</span>

<span style="font-size: 14px;">**Step 5: Element-wise multiply to get $h_t$.**</span>

<span style="font-size: 14px;">$h_t[0] = 0.6365 \times 0.9051 = 0.5762$</span>

<span style="font-size: 14px;">$h_t[1] = 0.4280 \times (-0.6044) = -0.2587$</span>

<span style="font-size: 14px;">$h_t[2] = 0.4256 \times 0.9967 = 0.4242$</span>

<span style="font-size: 14px;">**Final result:** $h_t = [0.5762, -0.2587, 0.4242]$</span>

<span style="font-size: 14px;">Dimension 0 has a high output gate ($0.64$) and a large positive cell state ($1.5$), so the hidden state strongly reflects that memory. Dimension 2 has a lower output gate ($0.43$) despite an even larger cell state ($3.2$) -- the gate suppresses most of that stored information. The cell "knows" something in dimension 2 but chooses not to reveal it at this time step.</span>

---

## <span style="font-size: 16px;">Connection to GRU</span>

<span style="font-size: 14px;">The Gated Recurrent Unit (Cho et al., 2014) is a simplified recurrent architecture that eliminates the output gate entirely. In the GRU, there is no separate cell state -- the hidden state $h_t$ is the memory. The final hidden state is computed by linear interpolation between $h_{t-1}$ and a candidate $\tilde{h}_t$, with no gating applied after the interpolation.</span>

<span style="font-size: 14px;">This means the GRU has no mechanism to store information privately. Whatever the GRU remembers is always fully exposed as its output. In the LSTM, the cell state can hold information that the output gate suppresses for many time steps, only revealing it when relevant. The GRU cannot do this -- its memory is always transparent.</span>

<span style="font-size: 14px;">This simplification reduces the GRU from four gate computations (forget, input, candidate, output) to three (reset, update, candidate). The parameter savings are exactly $\frac{1}{4}$: one fewer weight matrix of shape $(H, H+D)$ and one fewer bias of shape $(H,)$. Empirically, this trade-off matters most on tasks requiring complex internal state tracking, where the LSTM's ability to hide information behind the output gate gives it an advantage. On simpler sequence tasks, the GRU performs comparably with fewer parameters.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Using $C_{t-1}$ instead of $C_t$ in the hidden state equation.** The hidden state is $h_t = o_t \odot \tanh(C_t)$, where $C_t$ is the already-updated cell state including contributions from both the forget gate and the input gate at the current step. Using $C_{t-1}$ means the hidden state does not reflect new information written at this step. This is subtle because $o_t$ itself is computed from $h_{t-1}$ and $x_t$ -- it does not depend on $C_t$ -- so the gate values are correct even when the wrong cell state is used.</span>

* <span style="font-size: 14px;">**Forgetting to apply tanh to the cell state.** Writing $h_t = o_t \odot C_t$ instead of $h_t = o_t \odot \tanh(C_t)$ removes the squashing nonlinearity. Since $o_t \in (0, 1)$ but $C_t$ can be arbitrarily large, the hidden state becomes unbounded. Downstream layers will receive inputs of unpredictable scale, gradients will destabilize, and training will diverge.</span>

* <span style="font-size: 14px;">**Confusing the output gate with the network's output.** The output gate $o_t$ is not the LSTM's prediction or the network's final output. It is an internal gating mechanism that produces $h_t$. The actual output (a class probability, a next-token logit) is computed by passing $h_t$ through additional layers. Naming variables carelessly leads to confusion about what the LSTM returns.</span>

* <span style="font-size: 14px;">**Computing the output gate from the new hidden state.** The output gate must be computed from $h_{t-1}$ (the previous hidden state), not from $h_t$ (which does not exist yet). Writing $o_t = \sigma(W_o \cdot [h_t, x_t] + b_o)$ creates a circular dependency: $h_t$ depends on $o_t$, which depends on $h_t$. All four LSTM gates take $[h_{t-1}, x_t]$ as input, never the current hidden state.</span>

* <span style="font-size: 14px;">**Using matrix multiplication instead of element-wise product.** The operation $o_t \odot \tanh(C_t)$ is an element-wise (Hadamard) product, not a matrix multiplication. Both operands have shape $(H,)$ for a single sample or $(N, H)$ for a batch. Element-wise multiplication produces the same shape. Matrix multiplication collapses the result to a scalar, destroying all dimensional information.</span>

* <span style="font-size: 14px;">**Assuming the output gate is always near 1.** A common misconception is that the output gate "mostly passes things through" and is less important than the forget or input gates. In practice, trained LSTMs use the output gate aggressively. On language modeling tasks, dimensions frequently have $o_t < 0.1$ at time steps where stored information is not yet needed. The output gate is as critical as the other gates for managing information flow.</span>

---
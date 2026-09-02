# <span style="font-size: 20px;">Complete LSTM Cell</span>

<span style="font-size: 14px;">The Long Short-Term Memory (LSTM) cell is a recurrent neural network unit introduced by Hochreiter and Schmidhuber (1997) in "Long Short-Term Memory." It processes an input vector and a previous hidden state through four parallel gates to update a cell state and produce a new hidden state. The LSTM was designed to solve the vanishing gradient problem that prevents standard RNNs from learning long-range dependencies.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">The complete LSTM cell takes three inputs at each time step: the current input $x_t \in \mathbb{R}^D$, the previous hidden state $h_{t-1} \in \mathbb{R}^H$, and the previous cell state $C_{t-1} \in \mathbb{R}^H$. It produces two outputs: a new hidden state $h_t \in \mathbb{R}^H$ and a new cell state $C_t \in \mathbb{R}^H$. The hidden state is the externally visible output used by downstream layers, while the cell state is internal memory that flows across time steps with minimal transformation.</span>

<span style="font-size: 14px;">The cell operates through four learned gates. The **forget gate** decides which parts of the old cell state to discard. The **input gate** decides which parts to update. The **candidate cell state** proposes new content. The **output gate** decides which parts of the updated cell state to expose as the hidden state. These use four weight matrices and four bias vectors.</span>

<span style="font-size: 14px;">As stated in the original paper, "the memory cell is the central component of the LSTM architecture." The cell state acts as a conveyor belt: information flows along it unchanged across many time steps, with gates controlling what gets added or removed. This creates a direct gradient path through time that avoids the exponential decay plaguing vanilla RNNs.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">Let $x_t \in \mathbb{R}^D$ be the input, $h_{t-1} \in \mathbb{R}^H$ the previous hidden state, and $C_{t-1} \in \mathbb{R}^H$ the previous cell state. The concatenation $[h_{t-1}, x_t] \in \mathbb{R}^{H+D}$ is shared across all four gate computations.</span>

<span style="font-size: 14px;">**Equation 1 -- Forget gate.** Determines which components of the old cell state to retain:</span>

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

<span style="font-size: 14px;">Here $W_f \in \mathbb{R}^{H \times (H+D)}$ and $b_f \in \mathbb{R}^H$. The sigmoid squashes each element to $(0, 1)$. When $f_t \approx 1$, the dimension is preserved; when $f_t \approx 0$, it is erased.</span>

<span style="font-size: 14px;">**Equation 2 -- Input gate.** Controls how much of the candidate to write into memory:</span>

$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
$$

<span style="font-size: 14px;">Here $W_i \in \mathbb{R}^{H \times (H+D)}$ and $b_i \in \mathbb{R}^H$. Unlike the GRU's update gate, the forget and input gates are independent and do not sum to $1$.</span>

<span style="font-size: 14px;">**Equation 3 -- Candidate cell state.** Proposes new content to write:</span>

$$
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
$$

<span style="font-size: 14px;">Here $W_C \in \mathbb{R}^{H \times (H+D)}$ and $b_C \in \mathbb{R}^H$. The tanh bounds the candidate to $(-1, 1)$. The input gate scales this element-wise to control what gets written.</span>

<span style="font-size: 14px;">**Equation 4 -- Output gate.** Controls which parts of the cell state to expose:</span>

$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
$$

<span style="font-size: 14px;">Here $W_o \in \mathbb{R}^{H \times (H+D)}$ and $b_o \in \mathbb{R}^H$. This gate lets the LSTM store information in $C_t$ that is not yet reflected in its output.</span>

<span style="font-size: 14px;">**Equation 5 -- Cell state update.** Combines forgetting and writing:</span>

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

<span style="font-size: 14px;">The element-wise product $f_t \odot C_{t-1}$ erases old content, and $i_t \odot \tilde{C}_t$ writes new content. This additive update is the core of gradient preservation: during backpropagation, the gradient flows with a multiplicative factor of $f_t$, which can stay close to $1$.</span>

<span style="font-size: 14px;">**Equation 6 -- Hidden state.** Filters the cell state through the output gate:</span>

$$
h_t = o_t \odot \tanh(C_t)
$$

<span style="font-size: 14px;">The tanh squashes $C_t$ to $(-1, 1)$ before the output gate filters it. Note $C_t$ here is from Equation 5, not $C_{t-1}$. The hidden state $h_t$ is both the cell's output and the recurrent input for the next step.</span>

---

## <span style="font-size: 16px;">The Four Gates</span>

<span style="font-size: 14px;">**Forget gate ($f_t$).** Outputs a value in $(0, 1)$ per dimension, deciding how much of $C_{t-1}$ to keep.</span>

<span style="font-size: 14px;">**Input gate ($i_t$).** Outputs a value in $(0, 1)$ per dimension, deciding how much of $\tilde{C}_t$ to write.</span>

<span style="font-size: 14px;">**Candidate ($\tilde{C}_t$).** Proposes new values in $(-1, 1)$ via a tanh-activated transform on $[h_{t-1}, x_t]$.</span>

<span style="font-size: 14px;">**Output gate ($o_t$).** Outputs a value in $(0, 1)$ per dimension, deciding which parts of $C_t$ to expose as $h_t$.</span>

---

## <span style="font-size: 16px;">The Data Flow</span>

### <span style="font-size: 14px;">Step 1: Concatenation and Parallel Gate Computation</span>

<span style="font-size: 14px;">The input $x_t$ and previous hidden state $h_{t-1}$ are concatenated into dimension $H + D$. This is shared across all four gate computations. Implementations typically stack four weight matrices into a single $(4H, H+D)$ matrix, compute one matmul, split into four chunks, and apply sigmoid to three and tanh to the candidate.</span>

### <span style="font-size: 14px;">Step 2: Cell State Update</span>

<span style="font-size: 14px;">The forget gate $f_t$ multiplies element-wise with $C_{t-1}$, erasing selected dimensions. The input gate $i_t$ multiplies element-wise with $\tilde{C}_t$, scaling the new content. These are added to produce $C_t$. The update is purely element-wise with no matrix multiplication or activation. This linear path is what Hochreiter and Schmidhuber called the "constant error carousel."</span>

### <span style="font-size: 14px;">Step 3: Hidden State Extraction</span>

<span style="font-size: 14px;">The updated $C_t$ passes through tanh, then the output gate filters it element-wise to produce $h_t$. This serves dual roles: the cell's output for downstream layers and the recurrent input for the next time step.</span>

---

## <span style="font-size: 16px;">Why Separate Cell and Hidden State</span>

<span style="font-size: 14px;">The cell state $C_t$ is long-term memory with nearly linear dynamics. Because $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ is additive, gradients flow backward with minimal attenuation when the forget gate is close to $1$. This is the constant error carousel: the cell state carries information across hundreds of time steps without the exponential gradient decay of vanilla RNNs.</span>

<span style="font-size: 14px;">The hidden state $h_t = o_t \odot \tanh(C_t)$ is a filtered view for current use. The output gate lets the LSTM store information privately in $C_t$ until it is needed. In language modeling, the cell might store subject-verb agreement in $C_t$ while suppressing it until the relevant verb appears many tokens later.</span>

<span style="font-size: 14px;">The GRU folds cell and hidden state into one vector, so everything it remembers is always visible. The LSTM's two-state design provides more representational capacity at the cost of additional parameters.</span>

---

## <span style="font-size: 16px;">Parameter Count</span>

<span style="font-size: 14px;">The LSTM has four weight matrices of shape $(H, H + D)$ and four bias vectors of shape $(H,)$.</span>

* <span style="font-size: 14px;">**Forget gate:** $W_f$ has $H(H + D)$ weights, $b_f$ has $H$ biases</span>
* <span style="font-size: 14px;">**Input gate:** $W_i$ has $H(H + D)$ weights, $b_i$ has $H$ biases</span>
* <span style="font-size: 14px;">**Candidate:** $W_C$ has $H(H + D)$ weights, $b_C$ has $H$ biases</span>
* <span style="font-size: 14px;">**Output gate:** $W_o$ has $H(H + D)$ weights, $b_o$ has $H$ biases</span>

<span style="font-size: 14px;">**Total LSTM parameters:**</span>

$$
4H(H + D) + 4H
$$

<span style="font-size: 14px;">The GRU has three gates, totaling $3H(H + D) + 3H$. The LSTM uses $\frac{4}{3}$ of the GRU's parameters. With $H = 256$, $D = 128$: LSTM has $4 \times 256 \times 384 + 1{,}024 = 394{,}240$; GRU has $3 \times 256 \times 384 + 768 = 295{,}680$. The LSTM needs roughly 98,500 extra parameters per layer, gaining the output gate and private cell state.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Hochreiter and Schmidhuber (1997) published "Long Short-Term Memory" in Neural Computation. The paper addressed the vanishing gradient problem that Hochreiter analyzed in his 1991 diploma thesis and Bengio et al. (1994) independently documented. Standard RNNs multiply gradients by the recurrent weight matrix at each step during BPTT. When the largest singular value is below $1$, gradients decay exponentially; above $1$, they explode.</span>

<span style="font-size: 14px;">The solution was the constant error carousel: a self-loop with weight $1.0$ on the cell state. Gradients flow through this path multiplied by $f_t$ rather than a learned weight matrix. The input and output gates protect the carousel from irrelevant inputs and prevent stored information from disturbing unrelated computations.</span>

<span style="font-size: 14px;">The original 1997 LSTM lacked the forget gate, added by Gers, Schmidhuber, and Cummins (2000). Peephole connections came from Gers and Schmidhuber (2000). Greff et al. (2017) found that the forget gate and output activation are the most critical components, while peephole connections provide negligible benefit on most tasks.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">LSTM with $H = 2$, $D = 2$ at time step $t$:</span>

$$
x_t = \begin{pmatrix} 0.5 \\ -0.3 \end{pmatrix}, \quad h_{t-1} = \begin{pmatrix} 0.1 \\ -0.2 \end{pmatrix}, \quad C_{t-1} = \begin{pmatrix} 0.4 \\ -0.6 \end{pmatrix}
$$

<span style="font-size: 14px;">Concatenation: $[h_{t-1}, x_t] = [0.1, -0.2, 0.5, -0.3]$. Weights:</span>

$$
W_f = \begin{pmatrix} 0.3 & -0.1 & 0.2 & 0.4 \\ 0.1 & 0.5 & -0.3 & 0.2 \end{pmatrix}, \quad b_f = \begin{pmatrix} 0.1 \\ -0.1 \end{pmatrix}
$$

$$
W_i = \begin{pmatrix} 0.2 & 0.3 & -0.1 & 0.5 \\ -0.2 & 0.1 & 0.4 & -0.3 \end{pmatrix}, \quad b_i = \begin{pmatrix} 0.0 \\ 0.1 \end{pmatrix}
$$

$$
W_C = \begin{pmatrix} -0.1 & 0.4 & 0.3 & -0.2 \\ 0.2 & -0.3 & 0.1 & 0.5 \end{pmatrix}, \quad b_C = \begin{pmatrix} 0.0 \\ 0.1 \end{pmatrix}
$$

$$
W_o = \begin{pmatrix} 0.4 & -0.2 & 0.1 & 0.3 \\ -0.1 & 0.3 & 0.2 & -0.4 \end{pmatrix}, \quad b_o = \begin{pmatrix} 0.05 \\ -0.05 \end{pmatrix}
$$

<span style="font-size: 14px;">**Step 1: Forget gate.** $W_f [h,x] + b_f$: Row 1 = $0.03 + 0.02 + 0.10 - 0.12 + 0.10 = 0.13$. Row 2 = $0.01 - 0.10 - 0.15 - 0.06 - 0.10 = -0.40$. $f_t = [\sigma(0.13), \sigma(-0.40)] = [0.5325, 0.4013]$.</span>

<span style="font-size: 14px;">**Step 2: Input gate.** $W_i [h,x] + b_i$: Row 1 = $0.02 - 0.06 - 0.05 - 0.15 = -0.24$. Row 2 = $-0.02 - 0.02 + 0.20 + 0.09 + 0.10 = 0.35$. $i_t = [\sigma(-0.24), \sigma(0.35)] = [0.4403, 0.5866]$.</span>

<span style="font-size: 14px;">**Step 3: Candidate.** $W_C [h,x] + b_C$: Row 1 = $-0.01 - 0.08 + 0.15 + 0.06 = 0.12$. Row 2 = $0.02 + 0.06 + 0.05 - 0.15 + 0.10 = 0.08$. $\tilde{C}_t = [\tanh(0.12), \tanh(0.08)] = [0.1194, 0.0798]$.</span>

<span style="font-size: 14px;">**Step 4: Output gate.** $W_o [h,x] + b_o$: Row 1 = $0.04 + 0.04 + 0.05 - 0.09 + 0.05 = 0.09$. Row 2 = $-0.01 - 0.06 + 0.10 + 0.12 - 0.05 = 0.10$. $o_t = [\sigma(0.09), \sigma(0.10)] = [0.5225, 0.5250]$.</span>

<span style="font-size: 14px;">**Step 5: Cell update.** $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$: $C_t[0] = 0.5325 \times 0.4 + 0.4403 \times 0.1194 = 0.2130 + 0.0526 = 0.2656$. $C_t[1] = 0.4013 \times (-0.6) + 0.5866 \times 0.0798 = -0.2408 + 0.0468 = -0.1940$.</span>

<span style="font-size: 14px;">**Step 6: Hidden state.** $\tanh(C_t) = [0.2595, -0.1916]$. $h_t = o_t \odot \tanh(C_t) = [0.5225 \times 0.2595, \ 0.5250 \times (-0.1916)] = [0.1356, -0.1006]$.</span>

<span style="font-size: 14px;">Final: $h_t = [0.1356, -0.1006]$, $C_t = [0.2656, -0.1940]$. Cell dimension 0 shrank from $0.4$ to $0.27$ (forget gate $0.53$ partially erased it, input gate $0.44$ wrote a small candidate). Dimension 1 moved from $-0.6$ toward zero (forget gate $0.40$ erased most of the old value). Both output gates near $0.52$ passed roughly half the tanh-squashed cell state to $h_t$.</span>

---

## <span style="font-size: 16px;">LSTM vs GRU</span>

<span style="font-size: 14px;">**Gates.** LSTM has four (forget, input, candidate, output); GRU has three (reset, update, candidate). The GRU's update gate couples forgetting and input via $z_t + (1 - z_t) = 1$. The LSTM's forget and input gates are independent, so $f_t + i_t$ can exceed $1$, allowing the cell to simultaneously retain and add information.</span>

<span style="font-size: 14px;">**Cell state.** The LSTM maintains a separate $C_t$ shielded by the output gate, enabling private internal state. The GRU exposes its hidden state directly.</span>

<span style="font-size: 14px;">**Parameters.** LSTM: $4H(H + D) + 4H$. GRU: $3H(H + D) + 3H$. Ratio is $\frac{4}{3}$. For $H = 512$, $D = 256$: roughly 786K vs 589K per layer.</span>

<span style="font-size: 14px;">**Performance.** Comparable on most benchmarks. LSTM has a slight edge on tasks requiring precise long-term state tracking. GRU trains faster and generalizes better on smaller datasets.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Wrong gate order in the cell update.** The correct formula is $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$. Swapping to $i_t \odot C_{t-1} + f_t \odot \tilde{C}_t$ inverts gate semantics. The network may converge but learned weights will have inverted meanings.</span>

* <span style="font-size: 14px;">**Applying tanh to the cell state update.** The update must be $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ without tanh. Tanh is only applied in $h_t = o_t \odot \tanh(C_t)$. Wrapping the cell update in tanh destroys the linear dynamics of the constant error carousel and reintroduces vanishing gradients.</span>

* <span style="font-size: 14px;">**Forgetting to return both $h_t$ and $C_t$.** The LSTM must return $(h_t, C_t)$ because both are needed at the next step. Returning only $h_t$ loses the cell state and the LSTM degenerates into a gated network without long-term memory.</span>

* <span style="font-size: 14px;">**Using $C_{t-1}$ instead of $C_t$ for the output.** The hidden state uses the updated cell state: $h_t = o_t \odot \tanh(C_t)$. Using $C_{t-1}$ means the hidden state misses information just written by the input gate, a subtle off-by-one that silently degrades performance.</span>

* <span style="font-size: 14px;">**Inconsistent concatenation order.** All four affine transforms must use $[h_{t-1}, x_t]$. If one gate uses $[x_t, h_{t-1}]$, its weight matrix applies to the wrong inputs. No shape error occurs but results are incorrect.</span>

* <span style="font-size: 14px;">**Element-wise vs matrix multiplication.** The operations $f_t \odot C_{t-1}$, $i_t \odot \tilde{C}_t$, and $o_t \odot \tanh(C_t)$ are Hadamard products, not matmuls. Both operands have shape $(H,)$ and produce $(H,)$.</span>

---
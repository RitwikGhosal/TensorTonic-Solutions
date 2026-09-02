# <span style="font-size: 20px;">Cell State Update</span>

<span style="font-size: 14px;">The cell state update is the core memory mechanism of the Long Short-Term Memory (LSTM) network, introduced by Hochreiter & Schmidhuber (1997) in "Long Short-Term Memory." At each time step, the cell state $C_t$ is computed by combining a selective erasure of the previous cell state $C_{t-1}$ with a selective write of new candidate content. The forget gate $f_t$ controls erasure; the input gate $i_t$ controls writing. No activation function is applied to $C_t$ itself, making the cell state a linear self-loop that preserves gradient flow across arbitrarily many time steps.</span>

<span style="font-size: 14px;">This is the equation that makes the LSTM work. Without it, recurrent networks suffer from vanishing gradients and cannot learn long-range dependencies. The cell state update solves this by creating what the original paper calls the "constant error carousel" -- a path through which error signals flow backward without being squashed or amplified.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">The cell state $C_t$ is the LSTM's internal memory vector. At every time step $t$, it is updated by two independent operations applied element-wise: the previous cell state $C_{t-1}$ is multiplied by the forget gate $f_t$ to selectively erase dimensions, and the input gate $i_t$ multiplied by the candidate cell state $\tilde{C}_t$ adds new information. The result is $C_t$.</span>

<span style="font-size: 14px;">The cell state is distinct from the hidden state $h_t$. The hidden state is what the LSTM exposes to the outside world. The cell state is internal -- it is only read out through the output gate, which computes $h_t = o_t \odot \tanh(C_t)$. This separation means the cell state can store information that is not yet relevant for output, holding it in reserve until the output gate opens.</span>

<span style="font-size: 14px;">Every operation on the cell state is element-wise. Each dimension of $C_t$ is updated independently: dimension $j$ can be fully erased while dimension $k$ is fully preserved, and dimension $m$ can receive strong new content while dimension $n$ receives none. This per-dimension independence gives the LSTM fine-grained control over what to remember and what to write.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">The cell state update depends on four quantities computed earlier in the LSTM pipeline. Let $x_t \in \mathbb{R}^D$ be the input and $h_{t-1} \in \mathbb{R}^H$ be the previous hidden state.</span>

<span style="font-size: 14px;">**Forget gate** (what to erase):</span>

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \in (0, 1)^H
$$

<span style="font-size: 14px;">**Input gate** (how much to write):</span>

$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \in (0, 1)^H
$$

<span style="font-size: 14px;">**Candidate cell state** (what to write):</span>

$$
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \in (-1, 1)^H
$$

<span style="font-size: 14px;">**Cell state update** (the central equation):</span>

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

<span style="font-size: 14px;">Here $\odot$ denotes element-wise (Hadamard) multiplication. The forget gate values lie in $(0, 1)$, so $f_t \odot C_{t-1}$ scales each dimension of $C_{t-1}$ between full erasure ($f_t[j] \approx 0$) and full preservation ($f_t[j] \approx 1$). The input gate similarly scales the candidate. No activation is applied after the addition -- $C_t$ is raw, unbounded, and linear.</span>

---

## <span style="font-size: 16px;">The Constant Error Carousel</span>

<span style="font-size: 14px;">The constant error carousel (CEC) is the key insight of the original 1997 paper. It refers to the cell state's linear self-connection: $C_t = f_t \odot C_{t-1} + \ldots$. When the forget gate is close to 1, this reduces to approximately $C_t \approx C_{t-1}$. The cell state copies itself forward through time with minimal modification.</span>

<span style="font-size: 14px;">In a vanilla RNN, $h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$. During backpropagation through time, the gradient over $k$ steps involves $\prod_{s=1}^{k} W_h^T \cdot \text{diag}(\tanh'(\cdot))$. Since $|\tanh'(x)| \leq 1$ and the weight matrix compounds at each step, gradients vanish exponentially. Hochreiter (1991) proved that gradients in vanilla RNNs decay exponentially, making it impossible to learn dependencies beyond roughly 10-20 steps.</span>

<span style="font-size: 14px;">The CEC breaks this. The partial derivative of $C_t$ with respect to $C_{t-1}$ is simply $f_t$ (element-wise). No weight matrix, no activation derivative. If $f_t \approx 1$, the gradient passes through unchanged. Over $k$ time steps, the gradient is $\prod_{s=1}^{k} f_{t-s+1}$. When forget gates are consistently near 1, this product stays close to 1 even for large $k$.</span>

<span style="font-size: 14px;">The paper states: "The constant error carousel: the cell's linear self-connection ensures that error can flow back through arbitrary time steps without vanishing or exploding." This single design decision -- a linear self-loop with a learned gating coefficient -- is what made LSTMs the dominant sequence model for two decades.</span>

---

## <span style="font-size: 16px;">Why No Activation on $C_t$</span>

<span style="font-size: 14px;">The cell state update deliberately avoids applying any activation function to $C_t$ after the addition. This is not an oversight -- it is the entire point of the architecture.</span>

<span style="font-size: 14px;">If we applied tanh to $C_t$, the update would become $C_t = \tanh(f_t \odot C_{t-1} + i_t \odot \tilde{C}_t)$. The derivative of tanh is $1 - \tanh^2(x)$, which lies in $(0, 1]$. During backpropagation, each time step would multiply the gradient by this derivative, reintroducing the vanishing gradient problem the LSTM was designed to solve. The CEC requires the cell state recurrence to be linear in $C_{t-1}$.</span>

<span style="font-size: 14px;">Additionally, tanh would bound $C_t$ to $(-1, 1)$. Without activation, $C_t$ is unbounded -- it can grow to large positive or negative values. This is by design. The cell state is read out through $h_t = o_t \odot \tanh(C_t)$, where the tanh squashes $C_t$ for the output. The unbounded internal representation lets the cell accumulate information (counting, running sums) without saturating, while the output path ensures $h_t$ remains bounded.</span>

---

## <span style="font-size: 16px;">The Two-Part Update</span>

<span style="font-size: 14px;">The cell state update $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ has two additive components, each serving a distinct role. They are independent: the forget and input gates are computed from separate weight matrices and can take any combination of values.</span>

<span style="font-size: 14px;">**Part 1 -- Selective erasure: $f_t \odot C_{t-1}$.** The forget gate $f_t \in (0, 1)^H$ multiplies each dimension of $C_{t-1}$ independently. When $f_t[j] \approx 0$, dimension $j$ is erased. When $f_t[j] \approx 1$, dimension $j$ is preserved. For example, when processing language and encountering a sentence boundary, the forget gate might erase dimensions storing the previous sentence's subject.</span>

<span style="font-size: 14px;">**Part 2 -- Selective writing: $i_t \odot \tilde{C}_t$.** The input gate $i_t \in (0, 1)^H$ scales the candidate $\tilde{C}_t \in (-1, 1)^H$ per dimension. When $i_t[j] \approx 0$, nothing is written. When $i_t[j] \approx 1$, the full candidate value is added. The candidate proposes what to write; the input gate decides how much actually gets written.</span>

<span style="font-size: 14px;">The independence of forget and input gates distinguishes the LSTM from the GRU. In the GRU, a single update gate $z_t$ controls both: $h_t = z_t \odot h_{t-1} + (1 - z_t) \odot \tilde{h}_t$. The coefficients must sum to 1. The LSTM has no such constraint: $f_t$ and $i_t$ can both be near 1 (cell grows), both near 0 (cell shrinks toward zero), or any other combination.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Hochreiter & Schmidhuber published "Long Short-Term Memory" in Neural Computation in 1997. The paper addressed the fundamental problem identified in Hochreiter's 1991 diploma thesis and Bengio et al.'s 1994 analysis: gradient signals in recurrent networks vanish or explode exponentially with sequence length.</span>

<span style="font-size: 14px;">The core contribution was the constant error carousel -- the idea that a recurrent unit should have a linear self-connection so that error flows back through time without decay. The original 1997 architecture had input and output gates; the forget gate was added by Gers, Schmidhuber & Cummins in 2000. The cell state update equation is the direct implementation of the CEC idea.</span>

<span style="font-size: 14px;">The paper demonstrated that LSTMs could learn tasks requiring memory over 1000+ time steps, where vanilla RNNs failed at lags beyond 10-20 steps. This was revolutionary. The LSTM went on to become the dominant sequence architecture from roughly 2013 to 2017, powering breakthroughs in machine translation (Sutskever et al. 2014), speech recognition (Graves et al. 2013), and language modeling.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider an LSTM with hidden size $H = 4$. At time step $t$, the gates and candidate have been computed. We focus on the cell state update: $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$.</span>

<span style="font-size: 14px;">**Given values:**</span>

$$
C_{t-1} = \begin{pmatrix} 1.5 \\ -0.8 \\ 0.3 \\ 2.1 \end{pmatrix}, \quad f_t = \begin{pmatrix} 0.9 \\ 0.1 \\ 0.95 \\ 0.7 \end{pmatrix}, \quad \tilde{C}_t = \begin{pmatrix} 0.4 \\ 0.7 \\ -0.5 \\ 0.2 \end{pmatrix}, \quad i_t = \begin{pmatrix} 0.2 \\ 0.85 \\ 0.1 \\ 0.6 \end{pmatrix}
$$

<span style="font-size: 14px;">**Step 1: Selective erasure.** $f_t \odot C_{t-1}$, element by element:</span>

<span style="font-size: 14px;">Dim 0: $0.9 \times 1.5 = 1.35$ (high $f$, old value mostly preserved)</span>

<span style="font-size: 14px;">Dim 1: $0.1 \times (-0.8) = -0.08$ (low $f$, old value nearly erased)</span>

<span style="font-size: 14px;">Dim 2: $0.95 \times 0.3 = 0.285$ (very high $f$, almost fully kept)</span>

<span style="font-size: 14px;">Dim 3: $0.7 \times 2.1 = 1.47$ (moderate $f$, partial retention)</span>

<span style="font-size: 14px;">**Step 2: Selective writing.** $i_t \odot \tilde{C}_t$, element by element:</span>

<span style="font-size: 14px;">Dim 0: $0.2 \times 0.4 = 0.08$ (low $i$, little new content)</span>

<span style="font-size: 14px;">Dim 1: $0.85 \times 0.7 = 0.595$ (high $i$, strong write)</span>

<span style="font-size: 14px;">Dim 2: $0.1 \times (-0.5) = -0.05$ (low $i$, minimal write)</span>

<span style="font-size: 14px;">Dim 3: $0.6 \times 0.2 = 0.12$ (moderate $i$, partial write)</span>

<span style="font-size: 14px;">**Step 3: Sum.** $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$:</span>

<span style="font-size: 14px;">Dim 0: $1.35 + 0.08 = 1.43$ (preserved old memory, barely modified)</span>

<span style="font-size: 14px;">Dim 1: $-0.08 + 0.595 = 0.515$ (erased old, wrote new -- sign flipped from $-0.8$ to $+0.515$)</span>

<span style="font-size: 14px;">Dim 2: $0.285 - 0.05 = 0.235$ (preserved old, negligible new write)</span>

<span style="font-size: 14px;">Dim 3: $1.47 + 0.12 = 1.59$ (partial retention plus moderate addition)</span>

<span style="font-size: 14px;">**Interpretation.** Dim 0 is a "remember" dimension (high $f$, low $i$): $1.5 \to 1.43$. Dim 1 is a "replace" dimension (low $f$, high $i$): $-0.8 \to 0.515$. Dim 2 is pure preservation (high $f$, low $i$): $0.3 \to 0.235$. Dim 3 shows the additive nature: old content scaled down by $f = 0.7$, then new content layered on top. Note that $C_t$ values like 1.43 and 1.59 exceed $[-1, 1]$ -- this is expected since no activation bounds the cell state.</span>

---

## <span style="font-size: 16px;">Gradient Flow Analysis</span>

<span style="font-size: 14px;">From $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$, the partial derivative with respect to $C_{t-1}$ is:</span>

$$
\frac{\partial C_t}{\partial C_{t-1}} = \text{diag}(f_t)
$$

<span style="font-size: 14px;">This is a diagonal matrix whose entries are the forget gate values. No weight matrix, no activation derivative -- just the gate. Over $k$ time steps:</span>

$$
\frac{\partial C_t}{\partial C_{t-k}} = \prod_{s=1}^{k} \text{diag}(f_{t-s+1}) = \text{diag}\left(\prod_{s=1}^{k} f_{t-s+1}\right)
$$

<span style="font-size: 14px;">For each dimension $j$, the gradient magnitude is $\prod_{s=1}^{k} f_{t-s+1}[j]$:</span>

* <span style="font-size: 14px;">**$f[j] \approx 1$ at every step:** Product stays near 1. Gradient preserved. This is the CEC.</span>
* <span style="font-size: 14px;">**$f[j] \approx 0.9$ at every step:** After 10 steps, $0.9^{10} \approx 0.349$. After 50 steps, $0.9^{50} \approx 0.005$. Slower decay than vanilla RNNs, but still decays.</span>
* <span style="font-size: 14px;">**$f[j] \approx 0$ at some step:** Gradient for dimension $j$ is killed. This is intentional -- when the forget gate erases a dimension, it also blocks gradient flow through it.</span>

<span style="font-size: 14px;">Compare to a vanilla RNN where the gradient over $k$ steps is $\prod_{s=1}^{k} W_h^T \cdot \text{diag}(\tanh'(\cdot))$ -- repeated multiplication by a full weight matrix causing exponential growth or decay. The LSTM replaces this with multiplication by a diagonal matrix of gate values, which is vastly more stable.</span>

<span style="font-size: 14px;">There is a subtlety: the full LSTM gradient is more complex because $f_t$ depends on $h_{t-1}$, which depends on $C_{t-1}$ through the output gate. But the direct path through $\text{diag}(f_t)$ is the dominant gradient pathway and is what prevents vanishing gradients.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Applying tanh or sigmoid to $C_t$.** Writing $C_t = \tanh(f_t \odot C_{t-1} + i_t \odot \tilde{C}_t)$ destroys the constant error carousel. The gradient now includes $\tanh'(\cdot)$ at every step, reintroducing vanishing gradients. The cell state must remain activation-free.</span>

* <span style="font-size: 14px;">**Using matrix multiplication instead of element-wise product.** The operation $f_t \odot C_{t-1}$ is a Hadamard product, not a dot product or matrix multiply. Both $f_t$ and $C_{t-1}$ have shape $(H,)$ and produce shape $(H,)$. A dot product would collapse to a scalar.</span>

* <span style="font-size: 14px;">**Confusing $C_t$ with $h_t$.** The cell state is internal memory; the hidden state $h_t = o_t \odot \tanh(C_t)$ is the output. Returning $C_t$ as the LSTM output or using $h_t$ in place of $C_t$ in the recurrence ($C_t = f_t \odot h_{t-1} + \ldots$) breaks the architecture because $h_t$ has been squashed by tanh and gated by $o_t$.</span>

* <span style="font-size: 14px;">**Forgetting that $C_t$ can exceed $[-1, 1]$.** Since no activation is applied, $C_t$ values can grow large. Clipping $C_t$ to $[-1, 1]$ or assuming bounded values introduces incorrect behavior. The cell state is deliberately unbounded.</span>

* <span style="font-size: 14px;">**Wrong operation order.** The correct order is: (1) multiply $f_t$ with $C_{t-1}$, (2) multiply $i_t$ with $\tilde{C}_t$, (3) add. Computing $f_t + i_t$ first and then multiplying by some combined state is algebraically different and incorrect.</span>

* <span style="font-size: 14px;">**Assuming $f_t + i_t = 1$.** Unlike the GRU's update gate where coefficients sum to 1, the LSTM's forget and input gates are independent sigmoid outputs. They can both be near 1, both near 0, or any combination. There is no summation constraint.</span>
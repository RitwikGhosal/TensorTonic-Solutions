# <span style="font-size: 20px;">Forward Sequence Pass</span>

<span style="font-size: 14px;">The forward sequence pass unrolls a Vanilla RNN cell across $T$ timesteps. At each step $t$, the cell consumes the current input $x_t$ and the previous hidden state $h_{t-1}$, producing $h_t = \tanh(x_t W_{xh}^T + h_{t-1} W_{hh}^T + b_h)$. The pass collects every hidden state into a tensor of shape $(B, T, H)$ and returns $h_T$ separately with shape $(B, H)$.</span>

<span style="font-size: 14px;">This is the core operation that makes a recurrent network recurrent. A single RNN cell processes one timestep. The forward sequence pass is the loop that applies that cell repeatedly, threading the hidden state from one step to the next, turning a static cell into a sequence processor.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">Unrolling an RNN means applying the same cell once per timestep, feeding the output hidden state of step $t$ as input to step $t+1$. The recurrent graph is "unfolded" into a feedforward chain of $T$ identical cells sharing parameters.</span>

<span style="font-size: 14px;">The procedure takes three inputs:</span>

* <span style="font-size: 14px;">**Input sequence $X$:** Shape $(B, T, D)$ where $B$ is batch size, $T$ is timesteps, $D$ is input dimension. Each slice $x_t = X[:, t, :]$ is one timestep's input.</span>
* <span style="font-size: 14px;">**Initial hidden state $h_0$:** Shape $(B, H)$. Often zeros, but can be learned or carried from a previous segment.</span>
* <span style="font-size: 14px;">**Shared parameters:** $W_{xh} \in \mathbb{R}^{H \times D}$, $W_{hh} \in \mathbb{R}^{H \times H}$, $b_h \in \mathbb{R}^{H}$. Identical at every timestep.</span>

<span style="font-size: 14px;">It produces two outputs:</span>

* <span style="font-size: 14px;">**All hidden states:** Shape $(B, T, H)$ containing $h_1, h_2, \ldots, h_T$ stacked along the time axis. Each $h_t$ encodes information about the sequence up to step $t$.</span>
* <span style="font-size: 14px;">**Final hidden state $h_T$:** Shape $(B, H)$, summarizing the entire sequence. This is the last slice of the hidden states tensor, returned separately as a convenience.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">The forward pass applies the same recurrence $T$ times:</span>

$$
h_t = \tanh(x_t W_{xh}^T + h_{t-1} W_{hh}^T + b_h) \quad \text{for } t = 1, 2, \ldots, T
$$

<span style="font-size: 14px;">The three additive terms inside $\tanh$:</span>

* <span style="font-size: 14px;">**Input contribution $x_t W_{xh}^T$:** Projects the current input from $\mathbb{R}^D$ to $\mathbb{R}^H$, mapping input features into hidden space.</span>
* <span style="font-size: 14px;">**Recurrent contribution $h_{t-1} W_{hh}^T$:** Projects the previous hidden state from $\mathbb{R}^H$ to $\mathbb{R}^H$. This is the memory pathway -- it carries information from all previous timesteps.</span>
* <span style="font-size: 14px;">**Bias $b_h$:** Learnable offset in $\mathbb{R}^H$, broadcast across the batch.</span>

<span style="font-size: 14px;">The $\tanh$ squashes each element to $[-1, 1]$, bounding hidden state magnitude. Without it, repeated matrix multiplications through time would cause activations to explode or vanish even faster than they already tend to.</span>

<span style="font-size: 14px;">After the loop:</span>

$$
\text{hidden\_states} = \text{stack}([h_1, h_2, \ldots, h_T], \text{dim}=1) \in \mathbb{R}^{B \times T \times H}
$$

$$
h_{\text{final}} = h_T \in \mathbb{R}^{B \times H}
$$

---

## <span style="font-size: 16px;">The Unrolling Loop</span>

<span style="font-size: 14px;">The algorithm is a simple for-loop:</span>

<span style="font-size: 14px;">1. Set $h_{\text{prev}} = h_0$.</span>
<span style="font-size: 14px;">2. Create an empty list for hidden states.</span>
<span style="font-size: 14px;">3. For $t = 1$ to $T$:</span>
<span style="font-size: 14px;">$\quad$ a. Extract $x_t = X[:, t-1, :]$ (0-indexed).</span>
<span style="font-size: 14px;">$\quad$ b. Compute $h_t = \tanh(x_t W_{xh}^T + h_{\text{prev}} W_{hh}^T + b_h)$.</span>
<span style="font-size: 14px;">$\quad$ c. Append $h_t$ to the list.</span>
<span style="font-size: 14px;">$\quad$ d. Set $h_{\text{prev}} = h_t$.</span>
<span style="font-size: 14px;">4. Stack the list into shape $(B, T, H)$.</span>
<span style="font-size: 14px;">5. Return (stacked hidden states, $h_T$).</span>

<span style="font-size: 14px;">The critical detail is step 3d: the hidden state from the current step becomes the input to the next. This creates the sequential chain -- each $h_t$ is a function of $x_t$ and $h_{t-1}$, which itself depends on $x_{t-1}$ and $h_{t-2}$, all the way back to $h_0$. The entire sequence history is compressed into the hidden state at each step.</span>

---

## <span style="font-size: 16px;">Weight Sharing</span>

<span style="font-size: 14px;">A defining feature of RNNs is that the same $W_{xh}$, $W_{hh}$, and $b_h$ are reused at every timestep. This is weight sharing across time.</span>

* <span style="font-size: 14px;">**Parameter efficiency:** Without sharing, a sequence of length $T$ needs $T$ separate weight sets. Sharing keeps the parameter count independent of sequence length.</span>
* <span style="font-size: 14px;">**Generalization across positions:** The network learns one transformation that works at any position. A pattern learned at position 5 applies at position 50 without additional training.</span>
* <span style="font-size: 14px;">**Variable-length inputs:** The same cell handles any sequence length at inference, even lengths unseen during training.</span>

<span style="font-size: 14px;">During BPTT, gradients from every timestep flow to the same parameters. The total gradient is the sum of contributions from all $T$ steps, analogous to how a convolutional filter accumulates gradients from every spatial position.</span>

---

## <span style="font-size: 16px;">Sequential Dependency</span>

<span style="font-size: 14px;">The recurrence $h_t = f(h_{t-1}, x_t)$ creates a strict sequential dependency: step $t$ cannot begin until step $t-1$ completes.</span>

<span style="font-size: 14px;">**Why it cannot be parallelized across time:** To compute $h_3$, you need $h_2$. To get $h_2$, you need $h_1$. The chain is inherently serial. Even on a GPU with thousands of cores, the $T$ steps execute one after another. Parallelism is limited to the batch dimension -- all $B$ sequences process simultaneously, but within each sequence, timesteps are serial.</span>

<span style="font-size: 14px;">**Contrast with Transformers:** Self-attention computes all pairwise token interactions in one matrix multiplication -- no recurrence, all positions in parallel. A Transformer processes 1000 tokens in one parallel step; an RNN needs 1000 sequential steps. This is the main reason Transformers replaced RNNs for most sequence tasks.</span>

<span style="font-size: 14px;">**The tradeoff:** RNNs use $O(1)$ memory per step (just the hidden state), while Transformers use $O(T^2)$ for the attention matrix. For very long sequences this matters, but in practice the wall-clock penalty of serial execution dominates.</span>

---

## <span style="font-size: 16px;">Why Collect All Hidden States</span>

<span style="font-size: 14px;">Returning only $h_T$ might seem sufficient since it summarizes the whole sequence. But intermediate states are essential for many tasks:</span>

* <span style="font-size: 14px;">**Sequence labeling (many-to-many):** Tasks like POS tagging require a prediction at every position. Each $h_t$ feeds an output layer to produce the label for step $t$.</span>
* <span style="font-size: 14px;">**Attention mechanisms:** In seq2seq models, the decoder attends over all encoder states $[h_1, \ldots, h_T]$ to decide which input positions matter for each output step.</span>
* <span style="font-size: 14px;">**Bidirectional RNNs:** Forward and backward passes produce separate hidden states at each position; both are needed to concatenate into bidirectional representations.</span>
* <span style="font-size: 14px;">**Classification alternatives:** Even for many-to-one tasks, mean-pooling or max-pooling across all $h_t$ can outperform using $h_T$ alone.</span>

<span style="font-size: 14px;">Returning both the full tensor and $h_T$ separately follows PyTorch's `nn.RNN` convention, which returns `(output, h_n)`.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">The concept of unrolling a recurrent network through time originates with Jeffrey Elman's 1990 paper "Finding Structure in Time." Elman introduced the Simple Recurrent Network (SRN), maintaining a "context layer" that feeds back to the hidden layer at each timestep -- exactly the $h_{t-1} \rightarrow h_t$ recurrence implemented here.</span>

<span style="font-size: 14px;">The paper's key insight, captured in the quote "The network acts as a mapping from an input sequence to an output sequence, with the hidden units providing a memory of recent context," is that the hidden state serves as compressed memory. At each step, the network decides what to keep from the past (via $W_{hh}$) and how to integrate new input (via $W_{xh}$). The word "recent" is critical -- vanilla RNNs struggle with long-range dependencies because information degrades through repeated nonlinear transformations.</span>

<span style="font-size: 14px;">Elman's work predates the vanishing gradient analysis by Bengio et al. (1994) and the LSTM by Hochreiter and Schmidhuber (1997). The forward sequence pass here is the simplest temporal unrolling -- no gates, no cell state, just tanh at each step. Understanding it is essential before studying gated architectures designed to fix its limitations.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Trace $T = 3$ steps with $B = 1$, $D = 2$, $H = 3$.</span>

<span style="font-size: 14px;">**Parameters:**</span>

$$
W_{xh} = \begin{bmatrix} 0.5 & -0.3 \\ 0.2 & 0.4 \\ -0.1 & 0.6 \end{bmatrix}, \quad W_{hh} = \begin{bmatrix} 0.1 & -0.2 & 0.3 \\ 0.4 & 0.1 & -0.1 \\ -0.3 & 0.5 & 0.2 \end{bmatrix}, \quad b_h = \begin{bmatrix} 0.0 \\ 0.1 \\ -0.1 \end{bmatrix}
$$

<span style="font-size: 14px;">**Inputs:** $x_1 = [1.0, 0.5]$, $x_2 = [-0.5, 1.0]$, $x_3 = [0.8, -0.2]$. **Initial state:** $h_0 = [0, 0, 0]$.</span>

<span style="font-size: 14px;">**Step 1 ($h_0 \rightarrow h_1$):**</span>

<span style="font-size: 14px;">$x_1 W_{xh}^T = [0.5 - 0.15,\; 0.2 + 0.2,\; -0.1 + 0.3] = [0.35, 0.40, 0.20]$. Recurrent term is zero ($h_0 = 0$). Pre-activation: $[0.35, 0.50, 0.10]$ (after adding $b_h$). $h_1 = \tanh([0.35, 0.50, 0.10]) = [0.336, 0.462, 0.100]$.</span>

<span style="font-size: 14px;">**Step 2 ($h_1 \rightarrow h_2$):**</span>

<span style="font-size: 14px;">$x_2 W_{xh}^T = [-0.25 - 0.30,\; -0.10 + 0.40,\; 0.05 + 0.60] = [-0.55, 0.30, 0.65]$. $h_1 W_{hh}^T = [-0.029, 0.171, 0.150]$. Pre-activation: $[-0.579, 0.571, 0.700]$. $h_2 = \tanh([-0.579, 0.571, 0.700]) = [-0.522, 0.516, 0.604]$.</span>

<span style="font-size: 14px;">**Step 3 ($h_2 \rightarrow h_3$):**</span>

<span style="font-size: 14px;">$x_3 W_{xh}^T = [0.40 + 0.06,\; 0.16 - 0.08,\; -0.08 - 0.12] = [0.46, 0.08, -0.20]$. $h_2 W_{hh}^T = [0.026, -0.218, 0.535]$. Pre-activation: $[0.486, -0.038, 0.235]$. $h_3 = \tanh([0.486, -0.038, 0.235]) = [0.451, -0.038, 0.231]$.</span>

<span style="font-size: 14px;">**Output:** hidden_states shape $(1, 3, 3)$: $[[0.336, 0.462, 0.100],\; [-0.522, 0.516, 0.604],\; [0.451, -0.038, 0.231]]$. $h_{\text{final}} = [0.451, -0.038, 0.231]$.</span>

<span style="font-size: 14px;">**Observations:** At step 1, $h_0 = 0$ so only the input matters. By step 2, the recurrent term mixes $x_1$ into the processing of $x_2$. By step 3, $h_2$ carries influence from both $x_1$ and $x_2$, so $h_3$ reflects the entire sequence. Notice how $h_3$ is not simply a function of $x_3$ -- the value $-0.038$ in the second component emerged from the recurrent mixing of all three inputs.</span>

---

## <span style="font-size: 16px;">Truncated BPTT</span>

<span style="font-size: 14px;">The forward pass always unrolls all $T$ steps. But backpropagation through time (BPTT) can be truncated to $k$ steps -- a training-time optimization that does not change forward computation but affects learning.</span>

<span style="font-size: 14px;">**Full BPTT:** Gradients at step $t$ propagate through every prior hidden state to $h_0$, requiring $O(T)$ time and memory for all $T$ hidden states.</span>

<span style="font-size: 14px;">**Truncated BPTT (to $k$ steps):** Gradients at step $t$ only flow back through $h_t, h_{t-1}, \ldots, h_{t-k+1}$, then detach.</span>

* <span style="font-size: 14px;">**Memory savings:** Only $k$ states stored for backward, not all $T$.</span>
* <span style="font-size: 14px;">**Faster computation:** Each backward traverses $k$ steps instead of $T$.</span>
* <span style="font-size: 14px;">**Biased gradients:** Dependencies longer than $k$ steps cannot be learned, since gradient information beyond $k$ is discarded.</span>

<span style="font-size: 14px;">A common implementation splits the sequence into chunks of length $k$, runs forward within each chunk, computes gradients, then carries the final hidden state (detached from the graph) as $h_0$ for the next chunk. The forward pass still sees the full sequence; only the backward pass is truncated.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

<span style="font-size: 14px;">**1. Wrong loop direction.**</span>

<span style="font-size: 14px;">Iterating from $t = T$ to $t = 1$ reverses temporal order. The hidden state at "step 1" would encode $x_T$, not $x_1$. The forward pass must go in chronological order. (Backward passes in bidirectional RNNs deliberately reverse, but use separate parameters.)</span>

<span style="font-size: 14px;">**2. Forgetting to collect intermediate states.**</span>

<span style="font-size: 14px;">Only tracking $h_{\text{prev}}$ and $h_{\text{current}}$ without appending each $h_t$ to a list yields only the final state. Downstream tasks needing per-position representations (labeling, attention) will fail.</span>

<span style="font-size: 14px;">**3. Using the wrong hidden state for the next step.**</span>

<span style="font-size: 14px;">Forgetting $h_{\text{prev}} = h_t$ after computing $h_t$ causes the same initial state to be reused every step. The network loses all recurrence and degenerates into a position-independent feedforward map.</span>

<span style="font-size: 14px;">**4. Shape mismatch when stacking.**</span>

<span style="font-size: 14px;">Each $h_t$ has shape $(B, H)$. Stacking $T$ of them should give $(B, T, H)$ via `torch.stack(list, dim=1)`. Stacking along dim=0 produces $(T, B, H)$, which is wrong for the batch-first convention required here.</span>

<span style="font-size: 14px;">**5. Not returning $h_{\text{final}}$ separately.**</span>

<span style="font-size: 14px;">The problem requires both outputs. Returning only the stacked tensor and expecting the caller to slice `[:, -1, :]` violates the interface.</span>

<span style="font-size: 14px;">**6. Including $h_0$ in the output.**</span>

<span style="font-size: 14px;">Collecting $h_0$ alongside $h_1, \ldots, h_T$ produces shape $(B, T+1, H)$ instead of $(B, T, H)$. Only the states computed by the cell belong in the output.</span>
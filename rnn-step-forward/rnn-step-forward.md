## Why Sequences Need a Different Architecture

Standard neural networks take a fixed-size input and produce a fixed-size output. They have no concept of **order**. Feed them the words "dog bites man" or "man bites dog" and they see the same bag of words.

But many problems are inherently sequential:

- **Language**: word order determines meaning
- **Audio**: a waveform is a stream of samples over time
- **Finance**: stock prices form a time series
- **Sensors**: readings arrive one after another

In all of these, **earlier elements influence later ones**, and the sequence length can vary. Recurrent Neural Networks (RNNs) handle this by maintaining a **hidden state** that gets updated at every time step, acting as a running memory of what the network has seen so far.

---

## The Hidden State

The hidden state $h_t$ is a vector of size $H$ that summarizes everything the network has processed up to time $t$. Think of it as the network's "working memory."

- At $t = 0$, the network reads the first input and produces $h_0$.
- At $t = 1$, it reads the second input **and** $h_0$, producing $h_1$.
- At $t = 2$, it reads the third input **and** $h_1$, producing $h_2$.
- ...and so on.

Each hidden state carries forward a compressed representation of the entire history. The network never looks at raw past inputs directly. It only sees the current input and the previous hidden state.

The initial hidden state $h_{-1}$ is typically set to all zeros.

---

## The Tanh RNN Cell

The update rule for the simplest RNN cell is:

$$
h_t = \tanh(x_t \, W_x + h_{t-1} \, W_h + b)
$$

This has three parts that get **added together** before $\tanh$ is applied:

**1. Input contribution:** $x_t W_x$

- $x_t$ is the current input, shape $(D,)$
- $W_x$ is the input weight matrix, shape $(D, H)$
- The product is shape $(H,)$: it projects the input into the hidden space
- This is how the network "reads" the current time step

**2. Recurrent contribution:** $h_{t-1} W_h$

- $h_{t-1}$ is the previous hidden state, shape $(H,)$
- $W_h$ is the hidden weight matrix, shape $(H, H)$
- The product is shape $(H,)$: it transforms the previous memory
- This is the **recurrent connection**, the piece that gives the network memory
- $W_h$ is square because the hidden state stays the same size across time steps

**3. Bias:** $b$

- Shape $(H,)$, a learnable offset added to the pre-activation

After summing all three, the $\tanh$ function squashes each element into $[-1, 1]$. The result is the new hidden state $h_t$.

---

## Why Tanh?

- **Zero-centered**: outputs in $[-1, 1]$, so hidden states can be positive or negative. This leads to better gradient behavior than sigmoid (which outputs only positive values).
- **Bounded**: prevents the hidden state from growing unboundedly over many time steps.
- **Smooth**: differentiable everywhere, which is needed for gradient-based training.

The saturation at the extremes (output barely changes for very large or very small inputs) is both a feature (bounding) and a weakness (vanishing gradients), which we discuss below.

---

## Unrolling Through Time

To process a full sequence $x_0, x_1, x_2, \ldots$, the same cell is applied repeatedly:

$$
h_0 = \tanh(x_0 W_x + h_{-1} W_h + b)
$$

$$
h_1 = \tanh(x_1 W_x + h_0 W_h + b)
$$

$$
h_2 = \tanh(x_2 W_x + h_1 W_h + b)
$$

Two important things to notice:

- **Weight sharing**: The same $W_x$, $W_h$, and $b$ are used at every step. This means the number of parameters does not grow with sequence length. A sequence of length 10 and a sequence of length 10,000 use the same weights.
- **Chain of dependencies**: $h_2$ depends on $h_1$, which depends on $h_0$, which depends on $h_{-1}$. Information flows forward through this chain. Training requires sending gradients backward through it, which is called **Backpropagation Through Time (BPTT)**.

---

## The Vanishing Gradient Problem

During BPTT, the gradient at each step gets multiplied by:

- The derivative of $\tanh$ (at most 1, often much less)
- The weight matrix $W_h$

After many steps, these repeated multiplications cause the gradient to **shrink exponentially**. The result:

- The network can learn short-range patterns (a few steps back) easily
- It struggles with long-range dependencies (information from 50 or 100 steps ago)
- Early parts of the sequence get almost no gradient signal

This is why the plain tanh RNN has been largely replaced by gated architectures.

---

## LSTM and GRU: Solving the Problem

**LSTM** (Long Short-Term Memory) adds a separate **cell state** $c_t$ that flows through time with minimal modification, controlled by three gates:

- **Forget gate**: decides what to erase from the cell state
- **Input gate**: decides what new information to write
- **Output gate**: decides what to expose as the hidden state

The key insight: the cell state can carry information unchanged across many steps when the forget gate stays close to 1, letting gradients flow without shrinking.

**GRU** (Gated Recurrent Unit) is a simpler variant with two gates:

- **Reset gate**: controls how much of the previous state to forget
- **Update gate**: controls the mix between old state and new candidate

Both architectures build on the same core idea as the tanh RNN (combine input + previous state, apply nonlinearity) but add gating mechanisms that let gradients survive over long sequences.

---

## Where This Shows Up

- **Natural language processing**: RNNs were the dominant architecture for machine translation, text generation, and sentiment analysis before Transformers took over
- **Speech recognition**: Processing audio frame by frame, maintaining state across the utterance
- **Time series forecasting**: Predicting future values from historical patterns
- **Reservoir computing**: The tanh RNN is used with fixed (untrained) recurrent weights; only the output layer is learned
- **Building block**: The "linear combination then nonlinearity" pattern inside the tanh cell is the same atom used inside LSTM gates, GRU gates, and many other architectures
## Why Training Starts Badly

At the very beginning of training, everything is random:

- **Weights are random**: initialized from a random distribution (Xavier, He, etc.)
- **Activations are random**: since inputs pass through random weights, the activations are essentially noise
- **Gradients are unreliable**: computed from random activations, these gradients have high variance and may point in misleading directions
- **Optimizer state is empty**: Adam's moment estimates ($m$ and $v$) are all zeros, making the first few updates poorly calibrated

If you apply a full-strength learning rate to these unreliable gradients, the results can be catastrophic:

- A large, noisy gradient can push weights far into a bad region
- Once the model is in a bad region, it may take thousands of steps to recover
- In extreme cases, the loss diverges to infinity or becomes NaN
- Even if training eventually stabilizes, the early damage can lead to a suboptimal final result

---

## Warmup: Starting Gently

**Warmup** addresses this by starting with a very small (or zero) learning rate and gradually increasing it over the first several hundred or thousand steps.

The linear warmup formula for step $t < W$ (warmup steps):

$$
\eta(t) = \eta_0 \cdot \frac{t}{W}
$$

- At $t = 0$: $\eta = 0$ (no update at all)
- At $t = W/2$: $\eta = \eta_0/2$ (half the target rate)
- At $t = W$: $\eta = \eta_0$ (full target rate, warmup complete)

During warmup, several things stabilize:

- **Adam's second moment** ($v_t$): needs several steps to become a reliable estimate of gradient magnitudes. With the learning rate near zero, the early mismatch does not cause problems.
- **Batch normalization statistics**: running mean and variance need many batches to stabilize. Small learning rates prevent the model from changing too fast while these statistics are unreliable.
- **Gradient scale**: the typical magnitude of gradients settles into a stable range as the random initialization is "washed out" by the first few updates.
- **Activation distributions**: activations move from random noise toward meaningful patterns, making subsequent gradient signals more informative.

---

## Who Needs Warmup (and How Much)

Different settings need different amounts of warmup:

**Transformer models**:
- Warmup is essentially required. Without it, training often diverges in the first 100-1000 steps.
- Typical warmup: 500-4000 steps
- The "Attention Is All You Need" paper introduced warmup specifically for Transformers

**Large batch training**:
- Larger batches amplify gradient noise in the early phase
- Rule of thumb: warmup steps should increase with batch size
- GPT-3 used 375 million tokens of warmup

**Adam/AdamW**:
- The second moment estimates ($v_t$) are initialized to zero and need time to calibrate
- Without warmup, the first few Adam updates can have wildly wrong magnitudes
- Warmup gives $v_t$ time to become reliable

**Fine-tuning pretrained models**:
- Shorter warmup (100-500 steps) is usually sufficient
- The pretrained weights are already in a good region, so early gradients are more reliable

**Small models with SGD**:
- May not need warmup at all
- Simple MLPs and small CNNs often train fine without it

---

## Why Decay After Warmup

After warmup, the learning rate is at its peak value $\eta_0$. If it stayed there for the rest of training:

- The optimizer keeps taking **large steps** throughout training
- Near the end, when the model is close to a minimum, these large steps cause it to **bounce around** the minimum instead of settling in
- Training loss oscillates and never reaches its lowest possible value
- The model fails to "fine-tune" its parameters

Decaying the learning rate over time fixes this:

- Large rate early = fast exploration
- Decreasing rate = gradual transition from exploration to exploitation
- Small rate at the end = precise convergence

---

## The Complete Three-Phase Schedule

**Phase 1: Warmup** (step $t < W$)

Learning rate ramps up linearly from 0 to $\eta_0$:

$$
\eta(t) = \eta_0 \cdot \frac{t}{W}
$$

Example with $W = 1000$, $\eta_0 = 0.001$:
- Step 0: $\eta = 0$
- Step 250: $\eta = 0.00025$
- Step 500: $\eta = 0.0005$
- Step 1000: $\eta = 0.001$ (warmup complete)

**Phase 2: Linear decay** ($W \leq t \leq T$)

Learning rate decreases linearly from $\eta_0$ to 0:

$$
\eta(t) = \eta_0 \cdot \frac{T - t}{T - W}
$$

Example with $W = 1000$, $T = 10000$, $\eta_0 = 0.001$:
- Step 1000: $\eta = 0.001$ (just finished warmup)
- Step 3250: $\eta = 0.00075$
- Step 5500: $\eta = 0.0005$
- Step 7750: $\eta = 0.00025$
- Step 10000: $\eta = 0$ (end of training)

**Phase 3: Post-training** ($t > T$)

Learning rate stays fixed at 0 (or $\eta_{\min}$ if specified).

---

## The Shape

If you plot $\eta$ vs. step, the schedule looks like a **triangle** (or a ramp up followed by a ramp down):

- Rises linearly during warmup
- Peaks at step $W$
- Falls linearly during decay
- Flat at 0 after step $T$

The peak is always at the warmup boundary $t = W$. The warmup is typically much shorter than the total training, so the triangle is asymmetric: a short rise followed by a long decline.

---

## Edge Cases

**Zero warmup** ($W = 0$):
- The schedule starts at $\eta_0$ and decays linearly to 0
- No warmup phase at all
- This is just linear decay

**Warmup equals total steps** ($W = T$):
- The schedule ramps up from 0 to $\eta_0$ and immediately starts decaying
- The peak lasts only one step
- Not a useful configuration in practice

**Step beyond total steps** ($t > T$):
- The learning rate is clamped at 0 (or $\eta_{\min}$)
- No negative learning rates

---

## Warmup + Linear Decay vs. Other Schedules

- **Warmup + linear decay**: triangular shape. Simple, effective. The default in Hugging Face Transformers (`get_linear_schedule_with_warmup`).
- **Warmup + cosine decay**: warmup followed by a cosine curve. Slightly smoother. Often marginally better than linear decay. Used in many LLM training recipes.
- **Step decay (no warmup)**: learning rate drops by a fixed factor (e.g., 10x) at specific epochs. The traditional approach for CNNs trained with SGD. No warmup phase.
- **Constant (no schedule)**: $\eta$ never changes. Simple but suboptimal for most tasks.
- **Exponential decay**: $\eta$ is multiplied by a constant factor each step. Similar to linear in practice but with a different curve shape.

---

## Where This Schedule Shows Up

- **BERT pretraining**: the original BERT paper used exactly this: linear warmup + linear decay. This is what popularized the approach.
- **BERT fine-tuning**: the standard recipe for fine-tuning BERT on downstream tasks uses warmup (5-10% of steps) + linear decay.
- **Hugging Face Transformers**: the default scheduler is `get_linear_schedule_with_warmup`, which implements exactly this three-phase schedule.
- **GPT-2/GPT-3 training**: used warmup + cosine decay (a close relative).
- **Any Transformer training**: warmup + some form of decay has become the standard practice. If you are training or fine-tuning a Transformer and not using warmup, you are likely leaving performance on the table.
- **General best practice**: even outside of Transformers, warmup + decay is a safe default for any deep learning training run.
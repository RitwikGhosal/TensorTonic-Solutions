## What Are Discounted Returns?

The discounted return (also called discounted cumulative reward) is the total reward an agent receives from a given time step, with future rewards weighted less than immediate rewards.

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

where $\gamma \in [0, 1]$ is the **discount factor**.

---

## Why Discount Future Rewards?

**1. Mathematical convenience:**

Discounting ensures the sum converges for infinite horizons. Without discounting, returns could be infinite.

**2. Uncertainty about the future:**

Future rewards are less certain. Discounting reflects this uncertainty.

**3. Preference for immediacy:**

In many domains, getting reward sooner is preferable (time value of money, risk, etc.).

**4. Finite effective horizon:**

Discounting creates a "soft" horizon beyond which rewards barely matter.

---

## The Discount Factor $\gamma$

The discount factor $\gamma$ determines how much we value future rewards:

**$\gamma = 0$ (myopic):**

$$
G_t = R_{t+1}
$$
Only immediate reward matters. No planning ahead.

**$\gamma = 1$ (no discounting):**

$$
G_t = R_{t+1} + R_{t+2} + R_{t+3} + ...
$$
All rewards equally important. Only works for finite episodes.

**$\gamma \in (0, 1)$ (typical):**

Common values: 0.9, 0.95, 0.99. Balances short and long-term.

---

## Understanding $\gamma$ Values

**$\gamma = 0.9$:**
- Reward 10 steps ahead: $0.9^{10} \approx 0.35$ of immediate value
- Reward 20 steps ahead: $0.9^{20} \approx 0.12$ of immediate value
- Effective horizon: ~10 steps

**$\gamma = 0.99$:**
- Reward 10 steps ahead: $0.99^{10} \approx 0.90$ of immediate value
- Reward 100 steps ahead: $0.99^{100} \approx 0.37$ of immediate value
- Effective horizon: ~100 steps

Higher $\gamma$ means longer planning horizon.

---

## Recursive Definition

The return satisfies a recursive relationship:

$$
G_t = R_{t+1} + \gamma G_{t+1}
$$

This is crucial for efficient computation. Instead of summing all future rewards, we can compute returns backward from the end of an episode.

---

## Computing Returns: Worked Example

**Episode:** 5 timesteps with rewards

- $t=0$: $R_1 = 1$
- $t=1$: $R_2 = 2$
- $t=2$: $R_3 = 3$
- $t=3$: $R_4 = 4$
- $t=4$: $R_5 = 5$ (terminal)

**Discount factor:** $\gamma = 0.9$

---

**Direct computation for $G_0$:**

$$
G_0 = R_1 + \gamma R_2 + \gamma^2 R_3 + \gamma^3 R_4 + \gamma^4 R_5
$$

$$
= 1 + 0.9(2) + 0.81(3) + 0.729(4) + 0.6561(5)
$$

$$
= 1 + 1.8 + 2.43 + 2.916 + 3.2805 = 11.4265
$$

---

**Recursive computation (backward):**

$G_4 = R_5 = 5$

$G_3 = R_4 + \gamma G_4 = 4 + 0.9(5) = 4 + 4.5 = 8.5$

$G_2 = R_3 + \gamma G_3 = 3 + 0.9(8.5) = 3 + 7.65 = 10.65$

$G_1 = R_2 + \gamma G_2 = 2 + 0.9(10.65) = 2 + 9.585 = 11.585$

$G_0 = R_1 + \gamma G_1 = 1 + 0.9(11.585) = 1 + 10.4265 = 11.4265$

Same answer, computed efficiently in one backward pass.

---

## Algorithm for Computing All Returns

**Input:** Rewards $[R_1, R_2, ..., R_T]$, discount factor $\gamma$

**Output:** Returns $[G_0, G_1, ..., G_{T-1}]$

**Procedure:**

1. Initialize $G = 0$
2. For $t = T-1, T-2, ..., 0$:
   - $G = R_{t+1} + \gamma \cdot G$
   - Store $G_t = G$
3. Return list of returns

**Time complexity:** $O(T)$

**Space complexity:** $O(T)$ for storing returns

---

## Handling Terminal States

At episode end, the return is simply the final reward:

$$
G_{T-1} = R_T
$$

There are no future rewards beyond termination.

**For continuing (non-episodic) tasks:**

The return is an infinite sum. In practice, we truncate at some horizon or use bootstrapping (TD methods).

---

## n-Step Returns

Instead of the full return, we can use partial returns with bootstrapping:

**1-step return (TD target):**

$$
G_t^{(1)} = R_{t+1} + \gamma V(S_{t+1})
$$

**2-step return:**

$$
G_t^{(2)} = R_{t+1} + \gamma R_{t+2} + \gamma^2 V(S_{t+2})
$$

**n-step return:**

$$
G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n V(S_{t+n})
$$

These blend actual rewards with value estimates.

---

## Returns in Different RL Methods

**Monte Carlo methods:**
- Use complete returns $G_t$
- Wait until episode ends

**Temporal Difference (TD):**
- Use 1-step return: $R_{t+1} + \gamma V(S_{t+1})$
- Update after each step

**n-step TD:**
- Use n-step return
- Balance between MC and TD

**$\lambda$-return:**
- Weighted average of all n-step returns
- Used in TD($\lambda$) and GAE

---

## Properties of Discounted Returns

**Boundedness:**

If rewards are bounded by $R_{max}$, then:

$$
|G_t| \leq \sum_{k=0}^{\infty} \gamma^k R_{max} = \frac{R_{max}}{1 - \gamma}
$$

**Contraction:**

The discount factor makes the return a contraction mapping, which guarantees convergence of value iteration.

**Additivity:**

Returns can be decomposed:

$$
G_t = R_{t+1} + \gamma G_{t+1}
$$

---

## Effective Horizon

The effective horizon is approximately:

$$
H_{eff} \approx \frac{1}{1 - \gamma}
$$

This is the timescale over which rewards significantly contribute.

**Examples:**
- $\gamma = 0.9$: $H_{eff} \approx 10$ steps
- $\gamma = 0.99$: $H_{eff} \approx 100$ steps
- $\gamma = 0.999$: $H_{eff} \approx 1000$ steps

---

## Choosing the Discount Factor

**Task-dependent considerations:**

- Short episodes: Lower $\gamma$ (0.9-0.95) often works
- Long episodes: Higher $\gamma$ (0.99-0.999) needed
- Sparse rewards: Higher $\gamma$ to propagate reward signal

**Trade-offs:**

- Higher $\gamma$: Better long-term planning, slower learning, potential instability
- Lower $\gamma$: Faster learning, may miss long-term patterns

---

## Undiscounted Returns ($\gamma = 1$)

For finite episodic tasks, undiscounted returns are sometimes used:

$$
G_t = \sum_{k=t}^{T-1} R_{k+1}
$$

**When appropriate:**
- Episodes always terminate
- All rewards equally important
- No preference for sooner vs later

**Caution:** Can cause numerical issues in long episodes.

---

## Return Normalization

Returns can vary widely in magnitude. Normalization helps:

**Standardization:**

$$
G_t^{norm} = \frac{G_t - \mu_G}{\sigma_G}
$$

**Benefits:**
- Consistent learning rate across tasks
- Prevents exploding gradients
- Faster, more stable learning

---

## Returns vs Rewards vs Value

**Reward $R_t$:**

Immediate signal received at time $t$. Given by the environment.

**Return $G_t$:**

Cumulative discounted reward from time $t$ onward. Computed from rewards.

**Value $V(s)$:**

Expected return from state $s$. Learned estimate of $E[G_t | S_t = s]$.

---

## Computing Returns in Practice

**For policy gradient methods (PPO, REINFORCE):**

1. Collect complete episode trajectory
2. Compute returns backward: $G_t = R_{t+1} + \gamma G_{t+1}$
3. Use returns or advantages for gradient update

**For value-based methods (DQN):**

1. Use TD target: $R + \gamma \max_a Q(s', a)$
2. No need for complete returns

---

## Vectorized Computation

For efficiency, compute returns for batch of trajectories:

**Padding:** Pad shorter trajectories to same length

**Masking:** Use mask to handle different episode lengths

**Reverse cumsum:** Some frameworks provide efficient operations:

$$
G = \text{reverse\_cumsum}(\gamma^k \cdot R)
$$

---

## Common Mistakes

**1. Forgetting to handle terminal states:**

At terminal state, there is no $V(s')$ or $G_{t+1}$.

**2. Wrong discount application:**

$G_t = R_{t+1} + \gamma G_{t+1}$, not $G_t = \gamma R_{t+1} + \gamma G_{t+1}$

**3. Computing forward instead of backward:**

Forward computation is $O(T^2)$; backward is $O(T)$.

**4. Numerical overflow with large returns:**

Use appropriate scaling or lower $\gamma$ for very long episodes.
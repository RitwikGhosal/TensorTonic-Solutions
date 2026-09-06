## What Is Epsilon-Greedy?

Epsilon-greedy is an **action selection strategy** that balances exploration and exploitation in reinforcement learning. It selects the best-known action most of the time, but occasionally takes a random action to discover potentially better options.

The parameter $\epsilon$ controls the exploration rate.

---

## The Exploration-Exploitation Dilemma

In reinforcement learning, an agent faces a fundamental tradeoff:

**Exploitation:** Choose actions that have yielded high rewards in the past. This maximizes short-term reward based on current knowledge.

**Exploration:** Choose actions that have not been tried much. This gathers information that might lead to higher long-term rewards.

Pure exploitation may miss better actions. Pure exploration wastes time on bad actions. Good strategies balance both.

---

## The Epsilon-Greedy Policy

With probability $1 - \epsilon$: Choose the **greedy action** (highest estimated value)

With probability $\epsilon$: Choose a **random action** uniformly

Mathematically:

$$
\pi(a|s) = \begin{cases} 1 - \epsilon + \frac{\epsilon}{|A|} & \text{if } a = \arg\max_{a'} Q(s, a') \\ \frac{\epsilon}{|A|} & \text{otherwise} \end{cases}
$$

where $|A|$ is the number of available actions.

---

## Understanding the Probabilities

For a state with 4 possible actions and $\epsilon = 0.1$:

**Greedy action probability:**

$$
P(a^*) = 1 - \epsilon + \frac{\epsilon}{|A|} = 1 - 0.1 + \frac{0.1}{4} = 0.9 + 0.025 = 0.925
$$

**Non-greedy action probability (each):**

$$
P(a \neq a^*) = \frac{\epsilon}{|A|} = \frac{0.1}{4} = 0.025
$$

**Verification:**

$$
0.925 + 3 \times 0.025 = 0.925 + 0.075 = 1.0 \checkmark
$$

---

## Algorithm

**Input:** Q-values $Q(s, a)$, exploration rate $\epsilon$, current state $s$

**Output:** Action $a$

1. Generate random number $r \sim \text{Uniform}(0, 1)$

2. If $r < \epsilon$:
   - Return random action from available actions

3. Else:
   - Return $\arg\max_a Q(s, a)$

---

## Worked Example

**Setup:**
- State $s$ with 3 actions: A, B, C
- Q-values: $Q(s, A) = 2.5$, $Q(s, B) = 3.0$, $Q(s, C) = 1.8$
- $\epsilon = 0.2$

**Greedy action:** B (highest Q-value of 3.0)

**Action probabilities:**
- $P(B) = 1 - 0.2 + 0.2/3 = 0.8 + 0.067 = 0.867$
- $P(A) = 0.2/3 = 0.067$
- $P(C) = 0.2/3 = 0.067$

**If random number $r = 0.15$:**

Since $0.15 < 0.2$ (epsilon), explore: pick random action

**If random number $r = 0.75$:**

Since $0.75 \geq 0.2$, exploit: pick action B

---

## Choosing Epsilon

**High $\epsilon$ (e.g., 0.5):**
- More exploration
- Slower convergence to optimal policy
- Better for early learning
- Useful in highly stochastic environments

**Low $\epsilon$ (e.g., 0.01):**
- More exploitation
- Faster convergence if estimates are good
- May get stuck in local optima
- Good when Q-values are well-estimated

**Typical values:** $\epsilon \in [0.01, 0.3]$

---

## Epsilon Decay

A common strategy is to start with high exploration and gradually reduce it:

**Linear decay:**

$$
\epsilon_t = \max(\epsilon_{min}, \epsilon_0 - \text{decay\_rate} \times t)
$$

**Exponential decay:**

$$
\epsilon_t = \max(\epsilon_{min}, \epsilon_0 \times \text{decay}^t)
$$

**Inverse decay:**

$$
\epsilon_t = \frac{1}{1 + \text{decay\_rate} \times t}
$$

This allows broad exploration initially, then fine-tuning of the policy.

---

## Epsilon Decay Example

**Setup:**
- Initial: $\epsilon_0 = 1.0$ (100% exploration)
- Final: $\epsilon_{min} = 0.01$
- Decay rate: 0.995 (exponential)

**After 100 episodes:**

$$
\epsilon_{100} = \max(0.01, 1.0 \times 0.995^{100}) = \max(0.01, 0.606) = 0.606
$$

**After 500 episodes:**

$$
\epsilon_{500} = \max(0.01, 1.0 \times 0.995^{500}) = \max(0.01, 0.082) = 0.082
$$

**After 1000 episodes:**

$$
\epsilon_{1000} = \max(0.01, 1.0 \times 0.995^{1000}) = \max(0.01, 0.0067) = 0.01
$$

---

## Handling Ties in Greedy Selection

When multiple actions have the same maximum Q-value:

**Option 1:** Random selection among tied actions
- Most common approach
- Provides implicit exploration

**Option 2:** Fixed selection (e.g., first action)
- Deterministic
- May cause bias

**Option 3:** Ordered preference
- Use secondary criteria (e.g., action index)

---

## Epsilon-Greedy in Q-Learning

Q-learning uses epsilon-greedy for action selection during learning:

**While learning:**
1. Observe state $s$
2. Select action $a$ using epsilon-greedy from $Q(s, \cdot)$
3. Execute $a$, observe reward $r$ and next state $s'$
4. Update: $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$
5. $s \leftarrow s'$

The epsilon-greedy exploration ensures all state-action pairs are visited.

---

## Epsilon-Greedy in SARSA

SARSA also uses epsilon-greedy, but for both selection and update:

1. Observe state $s$
2. Select action $a$ using epsilon-greedy
3. Execute $a$, observe reward $r$ and next state $s'$
4. Select next action $a'$ using epsilon-greedy from $Q(s', \cdot)$
5. Update: $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma Q(s',a') - Q(s,a)]$
6. $s \leftarrow s'$, $a \leftarrow a'$

SARSA learns the value of the epsilon-greedy policy itself.

---

## Properties of Epsilon-Greedy

**1. GLIE (Greedy in the Limit with Infinite Exploration):**

With decaying $\epsilon$ where $\sum_t \epsilon_t = \infty$ and $\lim_{t \to \infty} \epsilon_t = 0$, epsilon-greedy satisfies GLIE conditions for convergence.

**2. All actions have non-zero probability:**

Every action can be selected, ensuring exploration of the entire state-action space.

**3. Simple and effective:**

Easy to implement and works well in practice.

---

## Comparison with Other Exploration Strategies

**Greedy ($\epsilon = 0$):**
- No exploration
- Exploits current knowledge completely
- May get stuck on suboptimal actions

**Random ($\epsilon = 1$):**
- Pure exploration
- No exploitation of learned values
- Very slow learning

**Softmax (Boltzmann):**

$$
\pi(a|s) = \frac{e^{Q(s,a)/\tau}}{\sum_{a'} e^{Q(s,a')/\tau}}
$$
- Action probability proportional to Q-value
- Temperature $\tau$ controls exploration
- Considers relative Q-values, not just max

---

## Softmax vs Epsilon-Greedy

**Epsilon-greedy:**
- All non-greedy actions equally likely
- Simple to implement and tune
- Does not consider Q-value differences

**Softmax:**
- Better actions more likely to be chosen
- More nuanced exploration
- Temperature parameter can be harder to tune

**Example:** Q-values [10.0, 9.9, 1.0]

Epsilon-greedy: Actions 2 and 3 equally likely when exploring

Softmax: Action 2 much more likely than action 3

---

## Upper Confidence Bound (UCB)

Another exploration strategy that considers uncertainty:

$$
a_t = \arg\max_a \left[ Q(a) + c\sqrt{\frac{\ln t}{N(a)}} \right]
$$

where $N(a)$ is the count of times action $a$ was selected.

UCB favors actions with high Q-values OR high uncertainty (low visit count).

---

## Epsilon-Greedy in Deep RL

In deep reinforcement learning (e.g., DQN):

- Q-values come from a neural network: $Q(s, a; \theta)$
- Epsilon-greedy still used for exploration
- Common to use linear epsilon decay over millions of frames
- Example: DQN uses $\epsilon$ from 1.0 to 0.1 over 1M frames

---

## Implementing Epsilon-Greedy Correctly

**Common mistakes:**

1. Forgetting to decay epsilon
2. Not handling action ties
3. Using epsilon for evaluation (should use greedy)
4. Decaying too fast (insufficient exploration)

**Best practices:**

1. Start with high epsilon (0.5 to 1.0)
2. Decay gradually based on environment complexity
3. Keep minimum epsilon > 0 for robustness
4. Use greedy policy for final evaluation

---

## Evaluation vs Training

**During training:**
- Use epsilon-greedy for exploration
- Learn from exploratory actions

**During evaluation:**
- Use greedy policy ($\epsilon = 0$)
- Report performance of learned policy
- No exploration needed when testing

This separation is important for fair evaluation of learned policies.
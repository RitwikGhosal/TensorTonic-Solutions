## What Is a Bigram?

A **bigram** is a sequence of two consecutive elements. In NLP, bigrams are pairs of consecutive words (or characters):

**Word bigrams in "I love machine learning":**
- (I, love)
- (love, machine)
- (machine, learning)

**Character bigrams in "hello":**
- (h, e)
- (e, l)
- (l, l)
- (l, o)

Bigrams capture local context: what word or character tends to follow another.

---

## The Bigram Language Model

A bigram language model estimates the probability of a word given only the previous word:

$$
P(w_n | w_1, w_2, ..., w_{n-1}) \approx P(w_n | w_{n-1})
$$

This is the **Markov assumption**: the next word depends only on the current word, not the entire history.

The probability of a full sentence is the product of bigram probabilities:

$$
P(w_1, w_2, ..., w_n) = P(w_1) \times P(w_2|w_1) \times P(w_3|w_2) \times ... \times P(w_n|w_{n-1})
$$

---

## Computing Bigram Probabilities

Bigram probabilities are estimated from a corpus using **maximum likelihood estimation**:

$$
P(w_2 | w_1) = \frac{\text{Count}(w_1, w_2)}{\text{Count}(w_1)}
$$

This is the number of times the bigram $(w_1, w_2)$ appears, divided by the number of times $w_1$ appears.

---

## A Worked Example

**Corpus:** "the cat sat on the mat the cat slept"

**Step 1: Count all bigrams**

- (the, cat): 2
- (cat, sat): 1
- (sat, on): 1
- (on, the): 1
- (the, mat): 1
- (mat, the): 1
- (cat, slept): 1

**Step 2: Count all unigrams (single words)**

- the: 3
- cat: 2
- sat: 1
- on: 1
- mat: 1
- slept: 1

**Step 3: Compute bigram probabilities**

$P(\text{cat} | \text{the}) = \frac{\text{Count(the, cat)}}{\text{Count(the)}} = \frac{2}{3} \approx 0.667$

$P(\text{mat} | \text{the}) = \frac{\text{Count(the, mat)}}{\text{Count(the)}} = \frac{1}{3} \approx 0.333$

$P(\text{sat} | \text{cat}) = \frac{\text{Count(cat, sat)}}{\text{Count(cat)}} = \frac{1}{2} = 0.5$

$P(\text{slept} | \text{cat}) = \frac{\text{Count(cat, slept)}}{\text{Count(cat)}} = \frac{1}{2} = 0.5$

---

## Start and End Tokens

Real text has beginnings and endings. We use special tokens:

- **<s>** or **<START>**: marks the beginning of a sentence
- **</s>** or **<END>**: marks the end of a sentence

**Sentence:** "the cat sat"

**With special tokens:** "<s> the cat sat </s>"

**Bigrams:**
- (<s>, the)
- (the, cat)
- (cat, sat)
- (sat, </s>)

This lets the model learn:
- Which words typically start sentences: $P(\text{the} | \text{<s>})$
- Which words typically end sentences: $P(\text{</s>} | \text{sat})$

---

## The Zero Probability Problem

What if we encounter a bigram that never appeared in training?

**Training corpus:** "the cat sat"

**New sentence:** "the dog sat"

The bigram (the, dog) has count 0, so:

$$
P(\text{dog} | \text{the}) = \frac{0}{\text{Count(the)}} = 0
$$

This is catastrophic. The probability of the entire sentence becomes 0, even though "the dog sat" is perfectly valid English.

---

## Smoothing Techniques

**Laplace (Add-One) Smoothing:**

Add 1 to all bigram counts:

$$
P(w_2 | w_1) = \frac{\text{Count}(w_1, w_2) + 1}{\text{Count}(w_1) + V}
$$

where $V$ is the vocabulary size.

This ensures no probability is ever exactly zero.

**Add-k Smoothing:**

Add a small constant $k$ (e.g., 0.01) instead of 1:

$$
P(w_2 | w_1) = \frac{\text{Count}(w_1, w_2) + k}{\text{Count}(w_1) + k \times V}
$$

**Backoff and Interpolation:**

If a bigram is unseen, "back off" to the unigram probability, or interpolate between bigram and unigram estimates.

---

## Building the Probability Matrix

For a vocabulary of size $V$, bigram probabilities form a $V \times V$ matrix where:

- Row $i$ corresponds to the first word $w_1$
- Column $j$ corresponds to the second word $w_2$
- Entry $(i, j)$ contains $P(w_2 = j | w_1 = i)$

Each row sums to 1 (it is a probability distribution over possible next words).

---

## Using Bigram Probabilities

**Text generation:**
1. Start with <s>
2. Sample next word from $P(w | \text{<s>})$
3. Sample next word from $P(w | \text{previous word})$
4. Repeat until </s> is sampled

**Sentence scoring:**
- Compute $P(\text{sentence}) = \prod P(w_i | w_{i-1})$
- Higher probability means the sentence is more likely under the model

**Perplexity:**
- Perplexity measures how "surprised" the model is by test data
- Lower perplexity means better model fit

---

## Limitations of Bigrams

**Limited context:**
Bigrams only see one word of history. They cannot capture:
- "The cat that I saw yesterday sat" (long-range dependency)
- Subject-verb agreement across clauses

**Data sparsity:**
Even with smoothing, rare word combinations are poorly estimated.

**No semantic understanding:**
Bigrams are purely statistical. They do not understand meaning.

Despite these limitations, bigram models are:
- Fast to train and query
- Useful baselines
- Components in larger systems (spelling correction, speech recognition)
- Educational for understanding language modeling fundamentals
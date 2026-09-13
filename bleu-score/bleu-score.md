## What Is BLEU?

BLEU (Bilingual Evaluation Understudy) is the standard automatic metric for evaluating machine translation quality. It measures how similar a candidate translation is to one or more reference translations.

The core idea: count how many n-grams (sequences of n consecutive words) in the candidate also appear in the reference. More matching n-grams means a better translation.

---

## Why BLEU Exists

Before BLEU, evaluating translation quality required expensive human judgments. BLEU provided a fast, automatic metric that correlates reasonably well with human assessments.

BLEU is now used beyond translation:
- Image captioning
- Text summarization
- Any text generation task with reference outputs

---

## The Components of BLEU

BLEU combines three elements:

**1. N-gram precision:** What fraction of candidate n-grams appear in the reference?

**2. Modified precision (clipping):** Prevents gaming by repeating words.

**3. Brevity penalty:** Penalizes translations that are too short.

---

## Modified N-gram Precision

Simple precision would count how many candidate n-grams appear in the reference:

$$
p_n = \frac{\text{matching n-grams}}{\text{total candidate n-grams}}
$$

But this is gameable. If the candidate is just "the the the the", it might match many references since "the" is common.

**Modified precision clips counts:**

For each n-gram, count its occurrences in the candidate, but cap at its maximum count in any reference:

$$
p_n = \frac{\sum_{\text{n-gram}} \min(C_{\text{n-gram}}, R_{\text{n-gram}})}{\sum_{\text{n-gram}} C_{\text{n-gram}}}
$$

Where:
- $C_{\text{n-gram}}$ is the count in the candidate
- $R_{\text{n-gram}}$ is the max count in any reference

---

## A Detailed Example

**Reference:** "the cat sat on the mat"
**Candidate:** "the the the cat mat"

**Unigram counts:**

Candidate counts: the=3, cat=1, mat=1 (total: 5)
Reference counts: the=2, cat=1, sat=1, on=1, mat=1

**Modified precision for unigrams:**
- "the": min(3, 2) = 2
- "cat": min(1, 1) = 1
- "mat": min(1, 1) = 1

Clipped matches: 2 + 1 + 1 = 4
Total candidate unigrams: 5

$p_1 = 4/5 = 0.8$

Without clipping, we would count all three "the"s, getting 5/5 = 1.0, which is misleading.

---

## Brevity Penalty

A short translation can have high precision by only including words it is confident about. To counter this, BLEU adds a brevity penalty (BP):

$$
BP = \begin{cases} 1 & \text{if } c \geq r \\ e^{1 - r/c} & \text{if } c < r \end{cases}
$$

Where:
- $c$ is the candidate length
- $r$ is the reference length (or closest reference length if multiple)

**Example:**
- Reference length: 10
- Candidate length: 5
- BP = $e^{1 - 10/5} = e^{-1} = 0.368$

The score is multiplied by 0.368, heavily penalizing the too-short translation.

If the candidate is longer than the reference, BP = 1 (no penalty). BLEU does not explicitly penalize verbosity, though low precision on longer outputs naturally reduces the score.

---

## Combining Into the Final Score

BLEU uses the geometric mean of precisions across different n-gram orders:

$$
\text{BLEU} = BP \cdot \exp\left(\frac{1}{N} \sum_{n=1}^{N} \log p_n\right)
$$

Where $N$ is the maximum n-gram order (typically 4, written as BLEU-4).

The geometric mean ensures that a zero precision at any level makes the entire score zero. This prevents translations that only get unigrams right but fail on longer phrases.

---

## Handling Zero Counts

If any $p_n = 0$ (no matching n-grams of that order), the log is undefined (negative infinity).

**Smoothing techniques:**
- Add a small epsilon to zero counts
- Add-one smoothing: add 1 to both numerator and denominator
- Simply report BLEU as 0 if any precision is 0

For single sentences, smoothing is important since zero n-gram matches are common.

---

## Corpus-Level vs. Sentence-Level

**Corpus-level BLEU:**
- Aggregate n-gram counts across all sentences
- More stable and meaningful
- This is the standard way to report BLEU

**Sentence-level BLEU:**
- Compute BLEU for each sentence separately
- High variance, often zero
- Requires smoothing to be useful

---

## Interpreting BLEU Scores

BLEU scores range from 0 to 1 (often reported as 0 to 100):

**BLEU = 0.60-1.00:** Very high quality, near human level
**BLEU = 0.40-0.60:** Understandable, good quality
**BLEU = 0.20-0.40:** The gist is clear but many errors
**BLEU = 0.00-0.20:** Poor quality

These ranges are rough. BLEU scores are most meaningful when comparing systems on the same test set.

---

## Limitations of BLEU

**Synonyms not rewarded:**
"fast" and "quick" mean the same thing, but BLEU treats them as completely different words.

**Word order flexibility:**
Some languages allow flexible word order. BLEU penalizes valid reorderings.

**No semantic understanding:**
"The cat sat on the mat" and "The mat was sat on by a cat" have different n-grams but the same meaning.

**Gaming:**
Systems can be optimized for BLEU without actually improving translation quality.

---

## Alternatives to BLEU

**METEOR:** Includes synonyms and stemming, correlates better with human judgment.

**ROUGE:** Used for summarization, focuses on recall rather than precision.

**BERTScore:** Uses neural embeddings for semantic similarity.

**Human evaluation:** Still the gold standard, but expensive.

BLEU remains popular due to its simplicity, speed, and widespread adoption despite its limitations.
## What Is Bag of Words?

The **Bag of Words (BoW)** model represents text as a collection of word counts, completely ignoring grammar, word order, and sentence structure. It treats a document as a "bag" (multiset) where only the presence and frequency of words matter.

**Example:**

"The cat sat on the mat" becomes:

- the: 2
- cat: 1
- sat: 1
- on: 1
- mat: 1

The fact that "cat" comes before "sat" is lost. The model only knows which words appear and how often.

---

## Why Ignore Word Order?

This seems like a major limitation, and it is. But BoW has surprising utility:

**1. Simplicity**

The representation is just a vector of counts. No complex parsing, no dependencies, no neural networks needed.

**2. Efficiency**

Building BoW representations is fast. Comparing documents is just computing vector similarity.

**3. Effectiveness for many tasks**

For topic classification, spam detection, and sentiment analysis, word presence is often the strongest signal. "The movie was terrible" and "terrible was the movie" have the same sentiment.

**4. Foundation for better models**

TF-IDF, BM25, and even some neural models build on BoW concepts.

---

## Building the Vocabulary

Before creating BoW vectors, you need a **vocabulary**: the list of all unique words across your corpus.

**Corpus:**
- Doc 1: "I love machine learning"
- Doc 2: "machine learning is great"
- Doc 3: "I love great food"

**Vocabulary (alphabetically sorted):**

[food, great, i, is, learning, love, machine]

Each word gets an index:
- food: 0
- great: 1
- i: 2
- is: 3
- learning: 4
- love: 5
- machine: 6

---

## Creating BoW Vectors

Each document becomes a vector of length $|V|$ (vocabulary size), where position $i$ contains the count of word $i$.

**Doc 1: "I love machine learning"**

- food: 0
- great: 0
- i: 1
- is: 0
- learning: 1
- love: 1
- machine: 1

Vector: [0, 0, 1, 0, 1, 1, 1]

**Doc 2: "machine learning is great"**

Vector: [0, 1, 0, 1, 1, 0, 1]

**Doc 3: "I love great food"**

Vector: [1, 1, 1, 0, 0, 1, 0]

---

## The Document-Term Matrix

Stack all document vectors to create a **document-term matrix**:

$$
\begin{bmatrix}
0 & 0 & 1 & 0 & 1 & 1 & 1 \\
0 & 1 & 0 & 1 & 1 & 0 & 1 \\
1 & 1 & 1 & 0 & 0 & 1 & 0
\end{bmatrix}
$$

- Rows are documents
- Columns are vocabulary terms
- Each cell is the count of that term in that document

This matrix is typically very **sparse** (mostly zeros) because each document uses only a small fraction of the total vocabulary.

---

## Preprocessing for BoW

Raw text needs cleaning before BoW works well:

**1. Lowercasing**

"Machine" and "machine" should be the same word.

**2. Tokenization**

Split text into words. Handle punctuation, contractions, hyphenated words.

"don't" might become ["do", "n't"] or ["dont"] or ["don't"]

**3. Stopword removal**

Remove common words like "the", "is", "a" that appear everywhere and add noise.

**4. Stemming/Lemmatization**

Reduce words to their root form.
- "learning", "learned", "learns" all become "learn"
- Reduces vocabulary size and groups related words

**5. Removing rare words**

Words appearing in only 1-2 documents add dimensions without useful signal.

**6. Removing very common words**

Words appearing in 90%+ of documents (beyond stopwords) may not be discriminative.

---

## Binary vs. Count vs. Frequency

**Binary BoW:**

1 if word is present, 0 if absent. Ignores how many times it appears.

Doc: "the cat sat on the mat"
Binary: the=1, cat=1, sat=1, on=1, mat=1

**Count BoW (standard):**

Raw count of each word.

Count: the=2, cat=1, sat=1, on=1, mat=1

**Term Frequency (normalized):**

Count divided by document length.

TF: the=2/6, cat=1/6, sat=1/6, on=1/6, mat=1/6

Normalization helps when comparing documents of different lengths.

---

## Vocabulary Size Considerations

Real-world vocabularies can be huge:

- English has 170,000+ words in the dictionary
- A large corpus might have 1,000,000+ unique tokens (including misspellings, names, numbers)

**Problems with large vocabularies:**

- High-dimensional vectors (memory, computation)
- Sparse representations (most entries are zero)
- Noise from rare/meaningless tokens

**Solutions:**

- Keep only top N most frequent words (e.g., N = 10,000)
- Remove words appearing in fewer than K documents
- Use subword tokenization (but this moves beyond simple BoW)

---

## Document Similarity with BoW

Once documents are vectors, you can compute similarity:

**Cosine similarity:**

$$
\text{sim}(A, B) = \frac{A \cdot B}{||A|| \times ||B||}
$$

This measures the angle between vectors, ignoring magnitude.

**Example:**

Doc A: [1, 2, 0, 1]
Doc B: [2, 1, 0, 1]

$A \cdot B = 1 \times 2 + 2 \times 1 + 0 \times 0 + 1 \times 1 = 5$

$||A|| = \sqrt{1 + 4 + 0 + 1} = \sqrt{6}$

$||B|| = \sqrt{4 + 1 + 0 + 1} = \sqrt{6}$

$\text{sim}(A, B) = \frac{5}{6} \approx 0.83$

---

## Limitations of BoW

**1. No word order**

"Dog bites man" and "Man bites dog" have identical BoW representations but opposite meanings.

**2. No semantics**

"Happy" and "joyful" are treated as completely different words with no relationship.

**3. High dimensionality**

Large vocabularies create sparse, high-dimensional vectors.

**4. No context**

"Bank" (financial) and "bank" (river) are the same word in BoW.

**5. Out-of-vocabulary words**

Words not seen during vocabulary construction cannot be represented.

---

## Beyond BoW

**N-grams:**

Instead of single words, use pairs (bigrams) or triples (trigrams).
"New York" becomes a single feature instead of "new" and "york" separately.

**TF-IDF:**

Weight terms by how rare they are across the corpus. Reduces the impact of common words.

**Word embeddings:**

Dense vectors (Word2Vec, GloVe) that capture semantic relationships. "King" - "man" + "woman" = "queen"

**Neural models:**

Transformers and RNNs that consider word order and context.

Despite these advances, BoW remains useful as a baseline, for efficiency-critical applications, and as a component in hybrid systems.
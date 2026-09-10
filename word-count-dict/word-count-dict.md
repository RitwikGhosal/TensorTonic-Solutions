## What Is Word Frequency Counting?

Word frequency counting is one of the most fundamental operations in natural language processing. Given a collection of text, it produces a dictionary mapping each unique word to the number of times it appears.

This simple operation is the foundation for:
- Bag-of-words representations
- TF-IDF vectorization
- Vocabulary construction
- Text analysis and statistics

---

## The Basic Algorithm

**Input:** A list of sentences, where each sentence is a list of tokens (words).

**Output:** A dictionary where keys are words and values are counts.

**Steps:**
1. Initialize an empty dictionary
2. For each sentence in the input:
   - For each token in the sentence:
     - If the token is in the dictionary, increment its count
     - Otherwise, add it with count 1
3. Return the dictionary

---

## A Concrete Example

**Input:**
- Sentence 1: ["the", "cat", "sat"]
- Sentence 2: ["the", "cat", "ate"]
- Sentence 3: ["the", "dog", "sat"]

**Process:**

After sentence 1: {"the": 1, "cat": 1, "sat": 1}
After sentence 2: {"the": 2, "cat": 2, "sat": 1, "ate": 1}
After sentence 3: {"the": 3, "cat": 2, "sat": 2, "ate": 1, "dog": 1}

**Output:**
{"the": 3, "cat": 2, "sat": 2, "ate": 1, "dog": 1}

The word "the" appears 3 times across all sentences, "cat" and "sat" appear 2 times each, etc.

---

## Why Tokenization Matters

The input is already tokenized (split into words). In practice, tokenization is a non-trivial preprocessing step that affects word counts:

**Decisions in tokenization:**
- Case sensitivity: Is "The" the same as "the"?
- Punctuation: Is "cat." the same as "cat"?
- Contractions: Is "don't" one token or two ("do" + "n't")?
- Numbers: Should "123" be a token or ignored?

Different tokenization choices lead to different word count dictionaries.

---

## Properties of Word Frequencies

**Zipf's Law:**
Word frequencies follow a power law. The most common word is much more frequent than the second most common, which is much more frequent than the third, and so on.

In English, "the" typically accounts for 5-7% of all words. The top 100 words often cover 50% of all word occurrences.

**Long tail:**
Most words in the vocabulary appear only once or twice. These rare words are called "hapax legomena" (appearing once) or "dis legomena" (appearing twice).

---

## Using Word Counts

**Bag-of-words representation:**
Convert each document to a vector where each dimension is a word and the value is its count. Ignores word order but captures topic information.

**Stop word identification:**
The most frequent words (the, a, is, of) are often stop words with little semantic content. Word counts help identify them for removal.

**Vocabulary construction:**
Build a vocabulary from words appearing at least $k$ times. Rare words might be typos or irrelevant.

**Keyword extraction:**
Words with unusually high frequency in a document (compared to the corpus) may be keywords.

---

## Efficient Implementation

**Using a default dictionary:**
Instead of checking if a key exists, use a data structure that defaults to 0 for missing keys. This simplifies the code and is often faster.

**Counter class:**
Many languages have built-in counter classes optimized for this task:
- Python: collections.Counter
- Java: HashMap with getOrDefault
- JavaScript: Map with manual counting

**Memory considerations:**
For large corpora, the dictionary can consume significant memory. Consider:
- Using integer IDs instead of string keys
- Pruning rare words during counting
- Using probabilistic data structures (Count-Min Sketch) for approximate counts

---

## Extensions

**N-gram counting:**
Instead of single words, count sequences of n consecutive words. Bigrams (n=2) capture word pairs like "New York" or "ice cream".

**Weighted counting:**
Give different weights to words based on position (title vs. body) or importance (TF-IDF weighting).

**Conditional counting:**
Count words that appear in specific contexts (e.g., after a particular word).

**Distributed counting:**
For very large corpora, use MapReduce or similar frameworks to count in parallel.

---

## From Counts to Vectors

The word count dictionary is often converted to a numerical vector for machine learning:

**Vocabulary mapping:**
Assign each unique word an index (e.g., "the" = 0, "cat" = 1, ...).

**Vector creation:**
For each document, create a vector of length |vocabulary| where position $i$ contains the count of word $i$.

**Sparse representation:**
Most documents contain only a small fraction of vocabulary words. Store as sparse vectors (only non-zero entries) to save memory.

---

## Relationship to Other NLP Concepts

**Term Frequency (TF):**
Word count in a single document. Often normalized by document length.

**Document Frequency (DF):**
Number of documents containing a word. Used with TF for TF-IDF.

**Inverse Document Frequency (IDF):**
$\log(N/DF)$ where $N$ is total documents. Weights rare words higher.

**TF-IDF:**
$TF \times IDF$. Balances word frequency in a document against rarity in the corpus.

Word counting is the first step toward all of these more sophisticated representations.
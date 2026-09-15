## What is Text Chunking?

Text chunking divides long documents into smaller, manageable pieces called chunks. This is essential for NLP systems with context limits, retrieval-augmented generation (RAG), and efficient document processing. Good chunking preserves semantic coherence while respecting size constraints.

---

## Why Chunk Text?

**Context window limits**: LLMs have maximum context lengths (e.g., 4K, 8K, 128K tokens). Documents exceeding this must be split.

**Retrieval precision**: Smaller chunks allow more precise matching in semantic search. A relevant paragraph is more useful than a vaguely related chapter.

**Memory efficiency**: Processing entire large documents at once may exceed available memory.

**Parallel processing**: Chunks can be processed independently across multiple workers.

---

## Chunking Strategies

### Fixed-Size Chunking

Split text into chunks of a fixed number of characters or tokens:

$$
\text{num\_chunks} = \lceil \frac{\text{text\_length}}{\text{chunk\_size}} \rceil
$$

**Advantages**:
- Simple to implement
- Predictable output size
- Works for any text

**Disadvantages**:
- May cut mid-sentence or mid-word
- Ignores document structure
- Can break semantic units

---

### Overlapping Chunks

Include some overlap between consecutive chunks to preserve context at boundaries:

$$
\text{chunk}_i = \text{text}[i \times \text{stride} : i \times \text{stride} + \text{chunk\_size}]
$$

Where stride = chunk_size - overlap

**Example**: chunk_size = 100, overlap = 20, stride = 80

- Chunk 0: characters 0-100
- Chunk 1: characters 80-180 (overlaps 80-100 with chunk 0)
- Chunk 2: characters 160-260

**Benefit**: Information at boundaries is not lost; appears in multiple chunks.

---

### Sentence-Based Chunking

Split at sentence boundaries, grouping sentences until size limit:

**Process**:
1. Split text into sentences
2. Accumulate sentences until adding the next would exceed limit
3. Start new chunk with next sentence

**Advantages**:
- Never cuts mid-sentence
- More semantically coherent
- Natural reading units

**Challenges**:
- Sentence detection is non-trivial (abbreviations, titles, etc.)
- Sentence lengths vary; some chunks much smaller than limit

---

### Paragraph-Based Chunking

Use paragraph breaks (double newlines) as chunk boundaries:

**Advantages**:
- Paragraphs are natural semantic units
- Author-intended structure preserved

**Challenges**:
- Paragraphs can be very long or very short
- May need secondary splitting for long paragraphs

---

### Recursive Chunking

Apply a hierarchy of separators:

1. Try to split by paragraph (double newline)
2. If chunks too large, split by single newline
3. If still too large, split by sentence
4. If still too large, split by word/character

**Benefit**: Preserves as much structure as possible while respecting size limits.

---

## Chunk Size Considerations

**Too small**:
- Loss of context within chunks
- More chunks to process and store
- Higher retrieval overhead

**Too large**:
- May exceed model context limits
- Retrieved chunks contain irrelevant information
- Less precise matching

**Typical sizes**:
- Sentence: 50-150 characters
- Paragraph: 200-500 characters
- Document section: 500-2000 characters
- For RAG: 256-1024 tokens commonly used

---

## Worked Example: Fixed-Size with Overlap

**Text**: "The quick brown fox jumps over the lazy dog. Then it runs away quickly."

**Parameters**: chunk_size = 30 characters, overlap = 10

**Chunks**:
- Chunk 0: "The quick brown fox jumps over" (positions 0-30)
- Chunk 1: "jumps over the lazy dog. Then" (positions 20-50)
- Chunk 2: "og. Then it runs away quickly." (positions 40-70)

**Observation**: "jumps over" appears in both chunks 0 and 1, providing continuity.

---

## Token-Based vs Character-Based

**Character-based chunking**:
- Language-agnostic
- Predictable character counts
- May split mid-token (subword)

**Token-based chunking**:
- Aligns with model tokenization
- More accurate for LLM context limits
- Requires tokenizer (language/model specific)

**Relationship**: Characters and tokens are not 1:1. Average ratio varies by language and tokenizer.

---

## Metadata Preservation

Each chunk should retain:

**Source information**: Document ID, filename, URL

**Position information**: Chunk index, character offset, page number

**Structural context**: Section heading, chapter title

**Why important**: After retrieval, need to trace back to original context and combine related chunks.

---

## Handling Special Content

**Tables**: May need to keep entire table in one chunk or use special formatting

**Code blocks**: Avoid splitting in the middle of functions or control structures

**Lists**: Keep list items together when possible

**Headers**: Include section headers with their content

**Images/Figures**: Reference them but they may need separate handling

---

## Chunking for Different Use Cases

**Semantic search/RAG**:
- Medium chunks (256-512 tokens)
- Overlap to capture boundary content
- Semantic boundaries preferred

**Summarization**:
- Larger chunks to preserve context
- Chapter or section level

**Question answering**:
- Smaller, precise chunks
- Paragraph or sentence level

**Classification**:
- May need full document or representative sample
- Consider hierarchical approach

---

## Quality Metrics for Chunking

**Coherence**: Do chunks represent complete thoughts?

**Coverage**: Is all important content captured (considering overlap)?

**Consistency**: Are chunks similar in size and scope?

**Retrieval performance**: Do retrieved chunks contain relevant information?

---

## Where Text Chunking Shows Up

- **RAG Systems**: Breaking documents for vector store indexing and retrieval

- **Document Processing**: Handling PDFs, reports, and long-form content

- **Chatbots**: Managing conversation history within context limits

- **Search Engines**: Indexing web pages and documents

- **Translation Systems**: Breaking text for efficient parallel translation

- **Summarization Pipelines**: Processing documents in manageable pieces

- **Legal/Medical NLP**: Analyzing lengthy contracts or patient records

- **Educational Platforms**: Breaking textbooks into digestible learning units

- **Content Management**: Organizing and serving large document collections

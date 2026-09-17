## What Is Edit Distance?

Edit distance (also called Levenshtein distance) measures the minimum number of single-character operations needed to transform one string into another. It answers the question: "How different are these two strings?"

The three allowed operations are:
- **Insert** a character
- **Delete** a character
- **Replace** a character with another

Each operation has a cost of 1.

---

## Why Edit Distance Matters

**Spell checking:**
When you type "teh", the spell checker computes edit distances to dictionary words. "the" is distance 1 away (swap two characters, or delete and insert), making it a likely correction.

**DNA sequence alignment:**
Biologists compare genetic sequences to find similarities. Edit distance measures how many mutations separate two sequences.

**Fuzzy string matching:**
Search engines and databases use edit distance to find approximate matches. Searching for "Shaksper" should still find "Shakespeare".

**Plagiarism detection:**
Slightly modified text can be detected by comparing edit distances to known sources.

---

## The Dynamic Programming Approach

Computing edit distance by trying all possible operation sequences would be exponentially slow. Instead, we use dynamic programming to build up the solution from smaller subproblems.

Define $dp[i][j]$ as the edit distance between the first $i$ characters of string $s_1$ and the first $j$ characters of string $s_2$.

---

## Base Cases

**Empty string to string of length j:**

$$
dp[0][j] = j
$$
We need $j$ insertions to build $s_2[0..j-1]$ from nothing.

**String of length i to empty string:**

$$
dp[i][0] = i
$$
We need $i$ deletions to reduce $s_1[0..i-1]$ to nothing.

---

## The Recurrence

For each position $(i, j)$, we have two cases:

**Case 1: Characters match ($s_1[i-1] = s_2[j-1]$)**

No operation needed for this character:

$$
dp[i][j] = dp[i-1][j-1]
$$

**Case 2: Characters differ ($s_1[i-1] \neq s_2[j-1]$)**

Take the minimum of three options:

$$
dp[i][j] = 1 + \min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
$$

Where:
- $dp[i-1][j] + 1$: delete from $s_1$ (we matched $i-1$ chars of $s_1$ to $j$ chars of $s_2$, now delete one)
- $dp[i][j-1] + 1$: insert into $s_1$ (we matched $i$ chars of $s_1$ to $j-1$ chars of $s_2$, now insert one)
- $dp[i-1][j-1] + 1$: replace (we matched $i-1$ to $j-1$, now replace the differing character)

---

## A Detailed Example

**s1 = "cat"**
**s2 = "cut"**

**Initialize the DP table:**

The first row represents transforming "" to "", "c", "cu", "cut" (0, 1, 2, 3 insertions).
The first column represents transforming "", "c", "ca", "cat" to "" (0, 1, 2, 3 deletions).

**Fill position (1,1): s1[0]='c', s2[0]='c'**
Characters match, so dp[1][1] = dp[0][0] = 0.

**Fill position (1,2): s1[0]='c', s2[1]='u'**
Characters differ. Options:
- Delete: dp[0][2] + 1 = 2 + 1 = 3
- Insert: dp[1][1] + 1 = 0 + 1 = 1
- Replace: dp[0][1] + 1 = 1 + 1 = 2

Minimum is 1, so dp[1][2] = 1.

**Continue filling...**

**Final table:**

      ""  c  u  t
  ""   0  1  2  3
  c    1  0  1  2
  a    2  1  1  2
  t    3  2  2  1

**Answer:** dp[3][3] = 1

This makes sense: "cat" becomes "cut" by replacing 'a' with 'u' (one operation).

---

## Reconstructing the Operations

The DP table tells us the minimum cost, but not which operations to perform. To find the actual operations:

1. Start at $dp[m][n]$
2. Look at which neighbor gave the minimum value
3. Move to that neighbor and record the operation
4. Repeat until reaching $dp[0][0]$

This backtracking produces the sequence of edits.

---

## Time and Space Complexity

**Time:** $O(mn)$ where $m = |s_1|$ and $n = |s_2|$. We fill an $m \times n$ table, each cell in $O(1)$.

**Space:** $O(mn)$ for the full table.

**Space optimization:** Since each row only depends on the previous row, we can use $O(\min(m, n))$ space by keeping only two rows.

---

## Variations

**Different operation costs:**
Instead of all costs being 1, you might have:
- Insert: cost 1
- Delete: cost 1
- Replace: cost 2

This changes the recurrence to use the actual costs.

**Damerau-Levenshtein distance:**
Adds a fourth operation: transpose two adjacent characters. "ab" to "ba" costs 1 instead of 2.

**Weighted edit distance:**
Different character substitutions have different costs. Replacing 'a' with 'e' might cost less than replacing 'a' with 'z' (they are closer on a keyboard or phonetically).

---

## Edit Distance vs. Other String Metrics

**Hamming distance:**
Only counts substitutions, requires strings of equal length. Edit distance is more general.

**Longest Common Subsequence (LCS):**
Related to edit distance. If LCS has length $L$, then edit distance is $(m - L) + (n - L)$ when only insertions and deletions are allowed.

**Jaccard similarity:**
Compares sets of characters or n-grams, ignoring order. Edit distance respects character positions.
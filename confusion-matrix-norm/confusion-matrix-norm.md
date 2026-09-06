## The Raw Confusion Matrix

A confusion matrix shows how predictions map to actual labels. For a 3-class problem:

$$
\begin{bmatrix}
50 & 5 & 2 \\
3 & 40 & 7 \\
1 & 4 & 30
\end{bmatrix}
$$

Row $i$, column $j$ contains the count of samples with true label $i$ predicted as class $j$. Diagonal elements are correct predictions.

**Problem:** Raw counts are hard to interpret when classes have different sizes. Class A with 57 samples and Class C with 35 samples cannot be compared directly.

---

## Why Normalize?

Normalization converts counts to proportions, making the matrix easier to interpret:

1. **Compare across classes:** See error rates regardless of class size
2. **Identify patterns:** Which classes are confused with which?
3. **Visualize fairly:** Heatmaps are meaningful when values are proportions

---

## Row Normalization (True Label Normalization)

Divide each row by its sum. Each row then sums to 1.

$$
C_{\text{norm}}[i, j] = \frac{C[i, j]}{\sum_k C[i, k]}
$$

**Interpretation:** "Given true label $i$, what fraction is predicted as class $j$?"

**Example:**

Raw row for Class A: [50, 5, 2], sum = 57

Normalized: [50/57, 5/57, 2/57] = [0.877, 0.088, 0.035]

This tells us:
- 87.7% of Class A samples are correctly classified
- 8.8% are misclassified as Class B
- 3.5% are misclassified as Class C

---

## Row Normalization: Full Example

Raw matrix:

$$
\begin{bmatrix}
50 & 5 & 2 \\
3 & 40 & 7 \\
1 & 4 & 30
\end{bmatrix}
$$

Row sums: [57, 50, 35]

Normalized:

$$
\begin{bmatrix}
0.877 & 0.088 & 0.035 \\
0.060 & 0.800 & 0.140 \\
0.029 & 0.114 & 0.857
\end{bmatrix}
$$

**Reading the result:**
- Class A: 87.7% recall (correctly identified)
- Class B: 80.0% recall, often confused with Class C (14%)
- Class C: 85.7% recall

The diagonal shows **recall** (sensitivity) for each class.

---

## Column Normalization (Predicted Label Normalization)

Divide each column by its sum. Each column then sums to 1.

$$
C_{\text{norm}}[i, j] = \frac{C[i, j]}{\sum_k C[k, j]}
$$

**Interpretation:** "Given prediction $j$, what fraction actually belongs to class $i$?"

**Example:**

Raw column for "Predicted A": [50, 3, 1], sum = 54

Normalized: [50/54, 3/54, 1/54] = [0.926, 0.056, 0.019]

This tells us:
- 92.6% of "predicted A" are actually Class A
- 5.6% are actually Class B
- 1.9% are actually Class C

---

## Column Normalization: Full Example

Raw matrix:

$$
\begin{bmatrix}
50 & 5 & 2 \\
3 & 40 & 7 \\
1 & 4 & 30
\end{bmatrix}
$$

Column sums: [54, 49, 39]

Normalized:

$$
\begin{bmatrix}
0.926 & 0.102 & 0.051 \\
0.056 & 0.816 & 0.179 \\
0.019 & 0.082 & 0.769
\end{bmatrix}
$$

The diagonal shows **precision** for each class.

---

## Total Normalization

Divide every element by the total count. The entire matrix sums to 1.

$$
C_{\text{norm}}[i, j] = \frac{C[i, j]}{\sum_{i,j} C[i, j]}
$$

**Interpretation:** "What fraction of all samples has true label $i$ and prediction $j$?"

**Example:**

Total samples: 57 + 50 + 35 = 142

Each cell divided by 142:

$$
\begin{bmatrix}
0.352 & 0.035 & 0.014 \\
0.021 & 0.282 & 0.049 \\
0.007 & 0.028 & 0.211
\end{bmatrix}
$$

Sum of diagonal = 0.352 + 0.282 + 0.211 = 0.845 = **accuracy**

---

## Choosing Normalization Type

**Row normalization (normalize over true labels):**
- Use when you want to see recall per class
- Answers: "How well do we detect each class?"
- Good for: imbalanced datasets, understanding false negatives

**Column normalization (normalize over predictions):**
- Use when you want to see precision per class
- Answers: "How trustworthy are predictions of each class?"
- Good for: understanding false positives

**Total normalization:**
- Use for overall view of error distribution
- Answers: "What proportion of data falls in each cell?"
- Diagonal sum equals accuracy

---

## Implementation Steps

**Row normalization:**
1. Compute sum of each row
2. Divide each element by its row sum
3. Handle rows with sum = 0 (no samples of that class)

**Column normalization:**
1. Compute sum of each column
2. Divide each element by its column sum
3. Handle columns with sum = 0 (class never predicted)

**Total normalization:**
1. Compute total sum of all elements
2. Divide every element by total

---

## Handling Zero Sums

If a row sums to zero (no samples of that class in test set):
- Option 1: Leave as zeros
- Option 2: Leave as NaN to indicate undefined
- Option 3: Exclude that class from analysis

If a column sums to zero (class never predicted):
- Same options apply
- Indicates the model never predicts that class

---

## Visualization Benefits

Normalized confusion matrices are better for heatmaps:

**Raw counts:** Color scale dominated by majority class. Hard to see patterns in minority classes.

**Normalized:** All rows (or columns) on same scale. Patterns visible across all classes.

The diagonal should ideally be dark (high values) and off-diagonal should be light (low values).

---

## Multi-Class Example Interpretation

Normalized confusion matrix (row-wise):

$$
\begin{bmatrix}
0.90 & 0.05 & 0.03 & 0.02 \\
0.08 & 0.85 & 0.04 & 0.03 \\
0.02 & 0.03 & 0.70 & 0.25 \\
0.01 & 0.02 & 0.20 & 0.77
\end{bmatrix}
$$

**Observations:**
- Classes A and B are well-classified (90%, 85%)
- Class C is often confused with Class D (25%)
- Class D is often confused with Class C (20%)
- Classes C and D might be inherently similar or need better features

---

## Connection to Metrics

**From row-normalized matrix:**
- Diagonal = recall (TPR) per class
- Off-diagonal in row $i$ = false negative rates for class $i$

**From column-normalized matrix:**
- Diagonal = precision per class
- Off-diagonal in column $j$ = false discovery rates for prediction $j$

**From total-normalized matrix:**
- Diagonal sum = accuracy
- Can derive all basic metrics
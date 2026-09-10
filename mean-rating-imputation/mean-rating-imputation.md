## What is Mean Rating Imputation?

Mean rating imputation is a technique used in recommender systems to fill in missing entries of a user-item rating matrix. When a user has not rated an item, we estimate what their rating might be based on available information - typically the mean rating given by that user, the mean rating received by that item, or some combination.

---

## The Sparsity Problem in Recommender Systems

Rating matrices are extremely sparse:
- Netflix: ~99% of entries are missing
- Amazon: ~99.9% of entries are missing

**Why so sparse?**
- Millions of users, millions of items
- Each user interacts with only a tiny fraction
- Impossible to rate everything

**Consequences**:
- Cannot directly compare users who rated different items
- Similarity measures fail without overlapping ratings
- Some algorithms require complete matrices

---

## Types of Mean Imputation

### Global Mean

Replace all missing values with the average of all known ratings in the matrix:

$$
\hat{r}_{ui} = \mu = \frac{1}{|\Omega|} \sum_{(u,i) \in \Omega} r_{ui}
$$

Where $\Omega$ is the set of observed ratings.

**Pros**: Simple, single value to compute
**Cons**: Ignores user and item differences completely

---

### User Mean

Replace missing values with the average rating given by that user:

$$
\hat{r}_{ui} = \bar{r}_u = \frac{1}{|I_u|} \sum_{i \in I_u} r_{ui}
$$

Where $I_u$ is the set of items rated by user $u$.

**Intuition**: Some users are "harsh critics" (low average), others are "easy graders" (high average). User mean captures this tendency.

**Pros**: Captures user-specific rating scale
**Cons**: Ignores item-specific quality

---

### Item Mean

Replace missing values with the average rating received by that item:

$$
\hat{r}_{ui} = \bar{r}_i = \frac{1}{|U_i|} \sum_{u \in U_i} r_{ui}
$$

Where $U_i$ is the set of users who rated item $i$.

**Intuition**: Some items are universally loved (high average), others are generally disliked (low average).

**Pros**: Captures item quality
**Cons**: Ignores user preferences

---

### Combined Imputation (User + Item Mean)

A more sophisticated approach combines user and item effects:

$$
\hat{r}_{ui} = \mu + b_u + b_i
$$

Where:
- $\mu$ = global mean
- $b_u = \bar{r}_u - \mu$ = user bias (how much user deviates from global mean)
- $b_i = \bar{r}_i - \mu$ = item bias (how much item deviates from global mean)

This baseline model accounts for both user leniency and item quality.

---

## Worked Example: User Mean Imputation

**Rating matrix** (? indicates missing):

- Alice: rated Movie1=4, Movie2=5, Movie3=?, Movie4=?
- Bob: rated Movie1=2, Movie2=?, Movie3=3, Movie4=?
- Carol: rated Movie1=?, Movie2=4, Movie3=?, Movie4=5

**Step 1 - Calculate user means**:

$$
\bar{r}_{Alice} = \frac{4 + 5}{2} = 4.5
$$

$$
\bar{r}_{Bob} = \frac{2 + 3}{2} = 2.5
$$

$$
\bar{r}_{Carol} = \frac{4 + 5}{2} = 4.5
$$

**Step 2 - Impute missing values**:

- Alice: Movie3 = 4.5, Movie4 = 4.5
- Bob: Movie2 = 2.5, Movie4 = 2.5
- Carol: Movie1 = 4.5, Movie3 = 4.5

**Interpretation**: Alice and Carol are imputed with higher values because they tend to give higher ratings.

---

## Worked Example: Item Mean Imputation

**Same rating matrix**:

**Step 1 - Calculate item means**:

$$
\bar{r}_{Movie1} = \frac{4 + 2}{2} = 3.0
$$

$$
\bar{r}_{Movie2} = \frac{5 + 4}{2} = 4.5
$$

$$
\bar{r}_{Movie3} = \frac{3}{1} = 3.0
$$

$$
\bar{r}_{Movie4} = \frac{5}{1} = 5.0
$$

**Step 2 - Impute missing values**:

- Alice: Movie3 = 3.0, Movie4 = 5.0
- Bob: Movie2 = 4.5, Movie4 = 5.0
- Carol: Movie1 = 3.0, Movie3 = 3.0

**Interpretation**: Movie4 gets high imputed values because it received a high rating from Carol.

---

## Handling Cold Start

**New user (no ratings)**: Cannot compute user mean
- Fall back to global mean
- Or use demographic-based estimates

**New item (no ratings)**: Cannot compute item mean
- Fall back to global mean
- Or use content-based estimates

**Combined approach**: When user or item mean is unavailable, use the available component plus global mean.

---

## Bias Correction

Raw imputation can introduce systematic errors:

**User mean bias**: If users who rated an item are not representative, user mean imputation inherits their biases.

**Item mean bias**: If items a user rated are not representative, item mean imputation misses their true preferences.

**Regularization**: Add a shrinkage term that pulls estimates toward the global mean when few ratings are available:

$$
\hat{r}_{ui} = \frac{n_u \cdot \bar{r}_u + \lambda \cdot \mu}{n_u + \lambda}
$$

Where $n_u$ is the number of ratings by user $u$ and $\lambda$ controls regularization strength.

---

## Imputation vs Prediction

**Imputation purpose**: Fill the matrix for downstream algorithms that need complete data (e.g., SVD, PCA).

**Prediction purpose**: Estimate what rating a user would give to recommend items.

Mean imputation is often a preprocessing step, not the final recommendation model. More sophisticated methods (matrix factorization, collaborative filtering) build on imputed baselines.

---

## Evaluation Considerations

**Do not evaluate on imputed values**: If you impute with user mean, predicting user mean will have zero error by construction.

**Hold-out evaluation**: Split observed ratings into train/test. Impute only training ratings. Evaluate on held-out test ratings.

**Imputation affects model training**: The quality of imputation impacts collaborative filtering algorithms that use the filled matrix.

---

## Where Mean Rating Imputation Shows Up

- **Netflix, Amazon, Spotify**: Recommendation engines handling sparse user-item matrices

- **Baseline Models**: User and item biases form the foundation of more complex models

- **Matrix Factorization Preprocessing**: SVD and ALS methods often center the matrix using mean subtraction

- **Cold Start Handling**: New users/items get global or content-based mean estimates

- **A/B Testing**: Imputed engagement scores for users who did not interact with certain variants

- **Survey Data**: Missing responses imputed with respondent or question means

- **Educational Platforms**: Student performance matrices with missing assessments

- **Healthcare**: Patient-treatment outcome matrices with missing observations

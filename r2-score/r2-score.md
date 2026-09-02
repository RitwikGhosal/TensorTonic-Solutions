## What R-squared Measures

R-squared ($R^2$), also called the **coefficient of determination**, measures how well a regression model explains the variance in the target variable.

**Intuition:** "What proportion of the target's variance is explained by the model?"

An $R^2$ of 0.85 means 85% of the variance in the target is explained by the model, and 15% remains unexplained.

---

## The Formula

$$
R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}
$$

where:

**Residual Sum of Squares (unexplained variance):**

$$
SS_{\text{res}} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

**Total Sum of Squares (total variance):**

$$
SS_{\text{tot}} = \sum_{i=1}^{n} (y_i - \bar{y})^2
$$

- $y_i$ is the actual value
- $\hat{y}_i$ is the predicted value
- $\bar{y}$ is the mean of actual values

---

## Understanding the Components

**$SS_{\text{tot}}$:** The total variance in the data. This is what you would get if you just predicted the mean for everything.

**$SS_{\text{res}}$:** The variance that remains after using the model. This is the squared error of your predictions.

**$R^2$:** The fraction of variance your model explains:

$$
R^2 = 1 - \frac{\text{unexplained variance}}{\text{total variance}} = \frac{\text{explained variance}}{\text{total variance}}
$$

---

## Interpreting R-squared

**$R^2 = 1$:** Perfect fit. The model explains all variance. $SS_{\text{res}} = 0$.

**$R^2 = 0$:** The model is no better than predicting the mean. $SS_{\text{res}} = SS_{\text{tot}}$.

**$R^2 < 0$:** The model is worse than predicting the mean. This happens when predictions are very poor.

**Typical interpretations:**
- $R^2 > 0.9$: Excellent fit
- $0.7 < R^2 < 0.9$: Good fit
- $0.5 < R^2 < 0.7$: Moderate fit
- $R^2 < 0.5$: Weak fit

Context matters. In some fields (physics), $R^2 > 0.99$ is expected. In others (social sciences), $R^2 = 0.3$ might be excellent.

---

## Worked Example

**Actual values:** $y = [3, 5, 7, 9, 11]$

**Predicted values:** $\hat{y} = [2.5, 5.5, 6.5, 9.5, 10]$

**Step 1: Compute mean of actual values**

$\bar{y} = (3 + 5 + 7 + 9 + 11) / 5 = 7$

**Step 2: Compute $SS_{\text{tot}}$**

$(3-7)^2 + (5-7)^2 + (7-7)^2 + (9-7)^2 + (11-7)^2$

$= 16 + 4 + 0 + 4 + 16 = 40$

**Step 3: Compute $SS_{\text{res}}$**

$(3-2.5)^2 + (5-5.5)^2 + (7-6.5)^2 + (9-9.5)^2 + (11-10)^2$

$= 0.25 + 0.25 + 0.25 + 0.25 + 1 = 2$

**Step 4: Compute $R^2$**

$R^2 = 1 - \frac{2}{40} = 1 - 0.05 = 0.95$

The model explains 95% of the variance.

---

## R-squared Can Be Negative

Unlike correlation (bounded -1 to 1), $R^2$ has no lower bound.

If $SS_{\text{res}} > SS_{\text{tot}}$, then $R^2 < 0$.

**When does this happen?**
- Predictions are worse than just predicting the mean
- Often indicates a bug or severely wrong model
- Can happen when evaluating on very different data than training

**Example:**

Actual: [10, 20, 30]

Predicted: [100, 200, 300] (way off)

The residuals are huge compared to predicting the mean, giving negative $R^2$.

---

## R-squared vs. Correlation

For simple linear regression (one feature), $R^2$ equals the square of the Pearson correlation:

$$
R^2 = r^2
$$

For multiple regression, this relationship does not hold directly. $R^2$ can be computed even when there is no single correlation coefficient.

---

## Adjusted R-squared

Plain $R^2$ always increases when you add more features, even if they add noise. **Adjusted $R^2$** penalizes for extra features:

$$
R^2_{\text{adj}} = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}
$$

where:
- $n$ is the number of samples
- $p$ is the number of features

Adjusted $R^2$ decreases if a new feature does not improve the model enough to justify its inclusion.

---

## R-squared vs. MSE/RMSE

**MSE (Mean Squared Error):**

$$
\text{MSE} = \frac{1}{n} \sum_i (y_i - \hat{y}_i)^2 = \frac{SS_{\text{res}}}{n}
$$

**Relationship:**

$$
R^2 = 1 - \frac{n \times \text{MSE}}{SS_{\text{tot}}}
$$

**Key difference:**
- MSE is in squared units of the target (e.g., dollars²)
- $R^2$ is unitless (proportion of variance)
- MSE depends on the scale; $R^2$ does not

---

## Limitations of R-squared

**Does not indicate prediction quality:**
A high $R^2$ does not mean predictions are accurate in absolute terms. If the target has huge variance, even $R^2 = 0.9$ can have large errors.

**Does not detect bias:**
A model can have high $R^2$ but systematically over- or under-predict.

**Sensitive to outliers:**
Both $SS_{\text{tot}}$ and $SS_{\text{res}}$ are based on squared values, amplifying outlier effects.

**Always increases with features:**
More features always increase $R^2$ (use adjusted $R^2$ to counter this).

**Not meaningful for some models:**
For non-linear models, $R^2$ can be misleading. Some practitioners avoid it for neural networks.

---

## When to Use R-squared

**Linear regression:**
$R^2$ is the standard metric for evaluating linear models.

**Comparing models:**
$R^2$ provides a normalized score (unlike MSE which depends on scale).

**Explaining to stakeholders:**
"The model explains 80% of the variance" is intuitive.

---

## When to Use Other Metrics

**Absolute error matters:**
Use MAE or RMSE to understand actual prediction errors.

**Outliers are a concern:**
Use MAE (less sensitive) or robust metrics.

**Comparing across datasets:**
$R^2$ is relative to each dataset's variance. Two datasets with same MSE can have very different $R^2$.

**Model selection:**
Use cross-validated metrics to avoid overfitting.

---

## Computing R-squared

**Step 1:** Compute mean of targets: $\bar{y} = \frac{1}{n} \sum y_i$

**Step 2:** Compute $SS_{\text{tot}} = \sum (y_i - \bar{y})^2$

**Step 3:** Compute $SS_{\text{res}} = \sum (y_i - \hat{y}_i)^2$

**Step 4:** $R^2 = 1 - SS_{\text{res}} / SS_{\text{tot}}$

Handle edge case: if $SS_{\text{tot}} = 0$ (all targets are identical), $R^2$ is undefined.
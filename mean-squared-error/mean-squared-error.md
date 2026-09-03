## What Is Mean Squared Error?

Mean Squared Error (MSE) is the most common loss function for regression problems. It measures the average of the squared differences between predicted values and actual values.

The formula:

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Where:
- $n$ is the number of samples
- $y_i$ is the true value for sample $i$
- $\hat{y}_i$ is the predicted value for sample $i$

---

## Breaking Down the Computation

The MSE calculation has three steps:

**Step 1: Compute the errors (residuals)**
- For each sample, subtract the prediction from the true value
- Error $= y_i - \hat{y}_i$
- This can be positive (underprediction) or negative (overprediction)

**Step 2: Square each error**
- $(y_i - \hat{y}_i)^2$
- Squaring does two things:
  - Makes all errors positive (so they do not cancel out)
  - Penalizes larger errors more heavily

**Step 3: Take the mean**
- Sum all squared errors and divide by $n$
- This gives a single number representing average squared error

---

## A Worked Example

Suppose you have 4 samples:

**Sample 1:**
- True value: 3.0
- Predicted: 2.5
- Error: 0.5
- Squared error: 0.25

**Sample 2:**
- True value: 5.0
- Predicted: 4.8
- Error: 0.2
- Squared error: 0.04

**Sample 3:**
- True value: 2.0
- Predicted: 2.7
- Error: -0.7
- Squared error: 0.49

**Sample 4:**
- True value: 8.0
- Predicted: 7.0
- Error: 1.0
- Squared error: 1.00

Sum of squared errors: $0.25 + 0.04 + 0.49 + 1.00 = 1.78$

MSE: $\frac{1.78}{4} = 0.445$

---

## Why Squaring?

The squaring operation is not arbitrary. It has several important properties:

**1. Eliminates sign**
- Raw errors can be positive or negative
- If you just averaged raw errors, positives and negatives would cancel
- Example: errors of +5 and -5 would give mean error of 0, hiding large mistakes

**2. Penalizes large errors disproportionately**
- An error of 2 contributes $2^2 = 4$ to the loss
- An error of 4 contributes $4^2 = 16$ to the loss
- The larger error contributes 4x more, not 2x more
- This makes the model very sensitive to outliers

**3. Mathematical convenience**
- The squared function is smooth and differentiable everywhere
- The gradient is simple: $\frac{\partial}{\partial \hat{y}} (y - \hat{y})^2 = -2(y - \hat{y})$
- This makes optimization straightforward

---

## The Gradient of MSE

During backpropagation, we need the gradient with respect to each prediction:

$$
\frac{\partial \text{MSE}}{\partial \hat{y}_i} = \frac{2}{n}(\hat{y}_i - y_i)
$$

Key observations:
- The gradient is proportional to the error itself
- If $\hat{y}_i > y_i$ (overprediction): positive gradient, push prediction down
- If $\hat{y}_i < y_i$ (underprediction): negative gradient, push prediction up
- Larger errors produce larger gradients, so the model corrects big mistakes faster

---

## MSE vs. MAE (Mean Absolute Error)

MAE uses absolute value instead of squaring:

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

**MSE characteristics:**
- Heavily penalizes outliers (squared penalty)
- Smooth gradient everywhere
- Gradient magnitude depends on error size
- Optimal prediction is the mean of true values
- Units are squared (e.g., if predicting meters, MSE is in meters squared)

**MAE characteristics:**
- Treats all errors linearly (less sensitive to outliers)
- Non-smooth gradient at zero (derivative undefined at $y = \hat{y}$)
- Constant gradient magnitude (always $\pm 1$)
- Optimal prediction is the median of true values
- Units match the original (e.g., meters)

---

## When MSE Struggles

**Outliers**
- A single huge error can dominate the entire loss
- Example: 99 samples with error 1, one sample with error 100
- MSE $\approx \frac{99 \times 1 + 10000}{100} = 100.99$
- The outlier contributes 100x more than all other samples combined

**Non-Gaussian error distributions**
- MSE implicitly assumes errors are normally distributed
- If the true error distribution is heavy-tailed or skewed, MSE may not be optimal

**Different scales**
- If targets span different magnitudes, large targets dominate the loss
- Consider normalizing targets or using percentage-based errors

---

## RMSE: Root Mean Squared Error

RMSE is simply the square root of MSE:

$$
\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

Why use RMSE?
- Same units as the original target (interpretable)
- Example: if predicting house prices in dollars, RMSE is also in dollars
- MSE would be in dollars squared, which is harder to interpret

The optimization is identical since taking the square root does not change which model minimizes the loss.

---

## Where MSE Is Used

- **Linear regression**: the classic least squares solution minimizes MSE
- **Neural network regression**: default loss for predicting continuous values
- **Time series forecasting**: predicting future numerical values
- **Autoencoders**: reconstruction loss for continuous data
- **Any task** where you predict a continuous number and want to penalize large errors heavily
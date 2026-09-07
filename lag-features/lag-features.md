## What Are Lag Features?

Lag features are values from previous time steps used as features for predicting the current or future time step. They capture temporal dependencies by incorporating historical information into the model.

Given a time series $y_t$, a lag-k feature is simply $y_{t-k}$, the value from k time steps ago.

---

## Why Use Lag Features?

**1. Capture temporal dependencies:**

Many real-world processes depend on their recent history. Today's stock price depends on yesterday's price.

**2. Enable standard ML models:**

Lag features transform time series problems into supervised learning problems that any ML algorithm can handle.

**3. Capture autocorrelation:**

If a time series is correlated with its own past values, lag features make this information available to the model.

**4. Simple and interpretable:**

Easy to understand and explain what information the model is using.

---

## Basic Lag Feature Definition

For a time series with values at times $t = 1, 2, 3, ..., T$:

**Lag-1 feature:** $x_t^{(1)} = y_{t-1}$

**Lag-2 feature:** $x_t^{(2)} = y_{t-2}$

**Lag-k feature:** $x_t^{(k)} = y_{t-k}$

The target is typically $y_t$ (current value) or $y_{t+h}$ (future value, h steps ahead).

---

## Worked Example

**Original time series:** Daily sales

- Day 1: 100
- Day 2: 120
- Day 3: 115
- Day 4: 130
- Day 5: 125
- Day 6: 140

**Creating lag-1 and lag-2 features:**

Day 3:
- Target (sales): 115
- Lag-1 (yesterday): 120
- Lag-2 (2 days ago): 100

Day 4:
- Target (sales): 130
- Lag-1 (yesterday): 115
- Lag-2 (2 days ago): 120

Day 5:
- Target (sales): 125
- Lag-1 (yesterday): 130
- Lag-2 (2 days ago): 115

Day 6:
- Target (sales): 140
- Lag-1 (yesterday): 125
- Lag-2 (2 days ago): 130

---

## Handling Missing Values at Start

Lag features create missing values for early observations:

**Problem:** For lag-2, the first two observations have no valid lag values.

**Solutions:**

1. **Drop rows:** Remove first k observations (lose data)
2. **Fill with zero:** $x_t^{(k)} = 0$ if $t - k < 1$
3. **Fill with mean:** $x_t^{(k)} = \bar{y}$ if $t - k < 1$
4. **Forward fill:** Use the first available value
5. **Mark as missing:** Let the model handle NaN values

---

## Choosing Number of Lags

**Too few lags:**

- Miss important temporal patterns
- Model cannot capture longer-term dependencies

**Too many lags:**

- Curse of dimensionality
- Many lags may be uninformative
- Increased risk of overfitting

**Guidelines:**

1. Use autocorrelation function (ACF) to identify significant lags
2. Start with lags corresponding to known cycles (lag-7 for weekly patterns)
3. Use cross-validation to select optimal number

---

## Lag Selection Using Autocorrelation

The autocorrelation function (ACF) measures correlation between $y_t$ and $y_{t-k}$:

$$
\rho_k = \frac{\text{Cov}(y_t, y_{t-k})}{\text{Var}(y_t)}
$$

**Interpretation:**

- High $|\rho_k|$ suggests lag-k is informative
- Include lags where ACF is significantly different from zero
- Seasonal patterns show spikes at seasonal lags

---

## Seasonal Lag Features

For data with known seasonality, include lags at the seasonal period:

**Daily data with weekly pattern:**

- Lag-7: Same day last week

**Monthly data with yearly pattern:**

- Lag-12: Same month last year

**Hourly data with daily pattern:**

- Lag-24: Same hour yesterday

**Example:** To predict Monday sales, lag-7 (last Monday) may be more predictive than lag-1 (Sunday).

---

## Multiple Lag Features Example

**Predicting daily website traffic:**

**Features for day t:**

- $y_{t-1}$: Yesterday's traffic
- $y_{t-2}$: Traffic 2 days ago
- $y_{t-7}$: Traffic same day last week
- $y_{t-14}$: Traffic same day 2 weeks ago
- $y_{t-365}$: Traffic same day last year

This captures short-term momentum, weekly patterns, and yearly seasonality.

---

## Difference Features (Related Concept)

Instead of raw lag values, use differences:

**First difference:**

$$
\Delta y_t = y_t - y_{t-1}
$$

**Seasonal difference:**

$$
\Delta_s y_t = y_t - y_{t-s}
$$

**Benefits:**

- Removes trends
- Makes non-stationary data stationary
- Captures change rather than level

---

## Rolling Statistics as Alternatives

Instead of single lag values, use statistics over a window:

**Rolling mean (window = 3):**

$$
\bar{y}_t = \frac{y_{t-1} + y_{t-2} + y_{t-3}}{3}
$$

**Rolling standard deviation:**

$$
\sigma_t = \sqrt{\frac{1}{w}\sum_{i=1}^{w}(y_{t-i} - \bar{y}_t)^2}
$$

These smooth out noise while capturing trends.

---

## Lag Features for Multiple Variables

In multivariate time series, create lags for each variable:

**Variables:** Sales ($s_t$), Price ($p_t$), Advertising ($a_t$)

**Features for predicting $s_t$:**

- Sales lags: $s_{t-1}$, $s_{t-2}$, ...
- Price lags: $p_{t-1}$, $p_{t-2}$, ...
- Advertising lags: $a_{t-1}$, $a_{t-2}$, ...

This allows modeling cross-variable dependencies.

---

## Lead Features (Opposite of Lag)

Lead features use future values:

$$
x_t^{(+k)} = y_{t+k}
$$

**Use case:** When predicting something that depends on known future events.

**Example:** Predicting inventory needs given known future orders.

**Warning:** Only use leads for features that are known in advance, never for the target variable.

---

## Lag Features for Classification

Lag features work for categorical targets too:

**Example:** Predicting customer churn

**Features:**

- Purchases last month (lag-1)
- Purchases 2 months ago (lag-2)
- Active last week (lag-1)
- Active 2 weeks ago (lag-2)

The pattern of declining activity may predict future churn.

---

## Entity-Specific Lags

For panel data (multiple entities over time), create lags within each entity:

**Example:** Multiple stores with daily sales

**Incorrect:** Using lag from different store

**Correct:** Use lag from same store

Store A, Day 5:
- Target: Store A sales on Day 5
- Lag-1: Store A sales on Day 4 (not Store B!)

Always respect entity boundaries.

---

## Time-Based vs Index-Based Lags

**Index-based lag:** Previous row in the data

**Time-based lag:** Previous time period

**Difference matters when data has gaps:**

**Data with missing days:**

- Monday: 100
- Tuesday: 120
- Thursday: 130 (Wednesday missing)

Index-based lag-1 of Thursday = 120 (Tuesday's value)

Time-based lag-1 of Thursday = NaN (Wednesday missing)

Choose based on what makes sense for your problem.

---

## Lag Features for Forecasting

**Single-step forecast:** Predict $y_{t+1}$ using $y_t, y_{t-1}, ...$

**Multi-step forecast:** Two approaches:

**1. Recursive forecasting:**

- Predict $y_{t+1}$ from lags
- Use predicted $\hat{y}_{t+1}$ as lag for $y_{t+2}$
- Errors compound

**2. Direct forecasting:**

- Train separate model for each horizon
- More models but no error compounding

---

## Avoiding Data Leakage

**Critical rule:** Lag features must only use past information.

**Common mistake:** Using current or future data in features.

**Correct:**

At time t, only use $y_{t-1}, y_{t-2}, ...$ (strictly past)

**Incorrect:**

Including $y_t$ or any aggregate that includes $y_t$ or later.

---

## Computational Efficiency

For large datasets, efficient lag computation matters:

**Vectorized shift operation:**

Most data libraries have optimized shift functions that are much faster than loops.

**Memory consideration:**

k lag features multiply data size by k. For large k, consider:
- Creating lags on-the-fly during training
- Using sparse representations
- Limiting lags to most important ones

---

## Lag Features in Different Domains

**Finance:**

- Stock returns: Lag prices, volumes, volatility
- Momentum indicators based on price lags

**Retail:**

- Sales forecasting: Lag sales, promotions, inventory
- Seasonal lags crucial for holidays

**Energy:**

- Load forecasting: Lag demand, temperature, time-of-use
- Strong daily and weekly patterns

**Web analytics:**

- Traffic prediction: Lag visits, page views, user counts
- Hourly and daily seasonality

---

## Feature Engineering with Lags

**Lag ratios:**

$$
\text{ratio}_t = \frac{y_t}{y_{t-1}}
$$

**Lag differences:**

$$
\text{diff}_t = y_t - y_{t-1}
$$

**Percentage change:**

$$
\text{pct}_t = \frac{y_t - y_{t-1}}{y_{t-1}} \times 100
$$

**Cumulative sums:**

$$
\text{cumsum}_t = \sum_{i=1}^{t} y_i
$$

---

## Validating Lag Feature Models

**Time series cross-validation:**

1. Train on days 1-100, test on days 101-110
2. Train on days 1-110, test on days 111-120
3. Continue expanding training window

**Never use standard k-fold:** It would leak future information into training.

**Walk-forward validation:** Most realistic for production use.

---

## Common Mistakes

**1. Leaking future information:**

Using $y_t$ to predict $y_t$ or using future values in features.

**2. Ignoring entity boundaries:**

Mixing lags across different entities in panel data.

**3. Wrong handling of missing values:**

Filling with values that leak information.

**4. Too many lags:**

Creating hundreds of lag features without selection.

**5. Forgetting seasonality:**

Missing important seasonal lags (lag-7, lag-365).

---

## Best Practices

**1. Start simple:**

Begin with lag-1 and add complexity as needed.

**2. Include seasonal lags:**

If weekly pattern exists, include lag-7.

**3. Use domain knowledge:**

Know your data's natural cycles.

**4. Validate properly:**

Use time-aware cross-validation.

**5. Monitor performance:**

Track how lag features improve predictions.
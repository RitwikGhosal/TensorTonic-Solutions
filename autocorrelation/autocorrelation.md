## What Is Autocorrelation?

Autocorrelation measures the linear relationship between a time series and a lagged version of itself. It quantifies how much past values influence current values.

$$
\rho_k = \frac{\sum_{t=k+1}^{T} (y_t - \bar{y})(y_{t-k} - \bar{y})}{\sum_{t=1}^{T} (y_t - \bar{y})^2}
$$

where $k$ is the lag and $\bar{y}$ is the mean of the series.

---

## The Formula

For a time series $y_1, y_2, ..., y_T$, the autocorrelation at lag $k$ is:

$$
\rho_k = \frac{\text{Cov}(y_t, y_{t-k})}{\text{Var}(y_t)}
$$

**Range:** $-1 \leq \rho_k \leq 1$

- $\rho_k = 1$: Perfect positive correlation
- $\rho_k = 0$: No linear correlation
- $\rho_k = -1$: Perfect negative correlation

---

## Worked Example

**Time series:** [5, 8, 6, 9, 7, 10, 8, 11]

**Mean:** $\bar{y} = \frac{5+8+6+9+7+10+8+11}{8} = \frac{64}{8} = 8$

**Lag 1 autocorrelation:**

Pairs: (8,5), (6,8), (9,6), (7,9), (10,7), (8,10), (11,8)

$$
\text{Numerator} = (8-8)(5-8) + (6-8)(8-8) + (9-8)(6-8) + (7-8)(9-8)
$$

$$
+ (10-8)(7-8) + (8-8)(10-8) + (11-8)(8-8)
$$

$$
= 0(-3) + (-2)(0) + 1(-2) + (-1)(1) + 2(-1) + 0(2) + 3(0)
$$

$$
= 0 + 0 - 2 - 1 - 2 + 0 + 0 = -5
$$

$$
\text{Denominator} = (5-8)^2 + (8-8)^2 + (6-8)^2 + (9-8)^2 + (7-8)^2
$$

$$
+ (10-8)^2 + (8-8)^2 + (11-8)^2
$$

$$
= 9 + 0 + 4 + 1 + 1 + 4 + 0 + 9 = 28
$$

$$
\rho_1 = \frac{-5}{28} = -0.179
$$

Weak negative correlation at lag 1.

---

## The Autocorrelation Function

The ACF is the set of autocorrelations for all lags:

$$
\text{ACF} = \{\rho_0, \rho_1, \rho_2, ..., \rho_m\}
$$

where $m$ is the maximum lag (typically $m \approx T/4$).

**Note:** $\rho_0 = 1$ always (series perfectly correlates with itself at lag 0).

**ACF plot:** Graph of $\rho_k$ vs $k$. Visual tool for identifying patterns.

---

## Interpreting ACF Patterns

**Random white noise:**

All $\rho_k \approx 0$ for $k > 0$. No autocorrelation.

**Strong trend:**

ACF decays slowly. High positive autocorrelation at many lags.

**Seasonality (period $s$):**

Peaks at lags $s, 2s, 3s, ...$

**Example:** Monthly data with yearly seasonality shows peaks at lags 12, 24, 36.

**Moving average process:**

ACF cuts off sharply after lag $q$.

**Autoregressive process:**

ACF decays exponentially or in damped sinusoidal pattern.

---

## Statistical Significance

Under null hypothesis of no autocorrelation (white noise):

$$
\rho_k \sim N\left(0, \frac{1}{T}\right)
$$

**95% confidence interval:**

$$
\left[-\frac{1.96}{\sqrt{T}}, \frac{1.96}{\sqrt{T}}\right]
$$

**Example:** For $T=100$, bounds are $\pm 0.196$.

Values outside these bounds are statistically significant.

---

## Partial Autocorrelation

PACF measures correlation at lag $k$ after removing effects of intermediate lags.

$$
\phi_{kk} = \text{Corr}(y_t, y_{t-k} | y_{t-1}, ..., y_{t-k+1})
$$

**Interpretation:**

- ACF: Total correlation (direct + indirect)
- PACF: Direct correlation only

**AR(p) process:**

PACF cuts off after lag $p$, ACF decays.

**MA(q) process:**

ACF cuts off after lag $q$, PACF decays.

---

## Autocorrelation in Model Identification

**ARIMA model selection:**

1. Plot ACF and PACF
2. Identify patterns
3. Determine model order

**Examples:**

- ACF tails off, PACF cuts off at lag $p$: Use AR($p$)
- ACF cuts off at lag $q$, PACF tails off: Use MA($q$)
- Both tail off: Use ARMA($p$, $q$)

**Differencing:**

If ACF decays very slowly, apply differencing until it decays quickly.

---

## Ljung-Box Test

Formal test for presence of autocorrelation at multiple lags:

$$
Q = T(T+2) \sum_{k=1}^{m} \frac{\rho_k^2}{T-k}
$$

**Null hypothesis:** $\rho_1 = \rho_2 = ... = \rho_m = 0$

**Distribution:** $Q \sim \chi^2_m$ under null hypothesis.

**Decision:** If $Q$ exceeds critical value, reject null (autocorrelation exists).

**Use case:** Testing residuals after fitting a model. Want to fail to reject (no autocorrelation in residuals).

---

## Autocorrelation in Financial Data

**Returns:**

Daily stock returns typically have $\rho_k \approx 0$ (efficient market hypothesis).

**Volatility:**

Squared returns show strong positive autocorrelation (volatility clustering).

**Trading strategies:**

Negative autocorrelation suggests mean reversion opportunities. Positive autocorrelation suggests momentum strategies.

---

## Seasonal Autocorrelation

For seasonal data with period $s$:

$$
\rho_s, \rho_{2s}, \rho_{3s}, ...
$$

are large and positive.

**Example:** Monthly sales data with yearly pattern.

$s = 12$, so $\rho_{12}, \rho_{24}, \rho_{36}$ are significant.

**Seasonal differencing:**

Take differences at lag $s$: $y_t - y_{t-s}$

This removes seasonal autocorrelation.

---

## Sample vs Population Autocorrelation

**Population:** $\rho_k$ (true parameter)

**Sample:** $\hat{\rho}_k$ (estimate from data)

Sample autocorrelation is biased in small samples, especially at large lags.

**Bias:** $E[\hat{\rho}_k] \neq \rho_k$ when $T$ is small.

**Variance increases with lag:** Higher uncertainty in $\hat{\rho}_k$ for larger $k$.

**Practical rule:** Only interpret lags up to $T/4$ or $T/3$.

---

## Cross-Correlation

Measures correlation between two different time series at various lags:

$$
\rho_{xy}(k) = \frac{\sum_{t=k+1}^{T} (x_t - \bar{x})(y_{t-k} - \bar{y})}{\sqrt{\sum_{t=1}^{T} (x_t - \bar{x})^2 \sum_{t=1}^{T} (y_t - \bar{y})^2}}
$$

**Interpretation:**

Positive $k$: $x$ leads $y$ by $k$ periods.

Negative $k$: $y$ leads $x$ by $|k|$ periods.

**Use case:** Finding lead-lag relationships between economic indicators.

---

## Autocorrelation and Stationarity

**Stationary process:**

ACF decays to zero as lag increases.

**Non-stationary process:**

ACF decays very slowly or not at all.

**Unit root:**

$\rho_1 \approx 1$ and ACF stays near 1 for many lags.

**Solution:** Difference the series until ACF decays properly.

---

## Computational Considerations

**Direct calculation:** $O(T^2)$ for all lags up to $T-1$.

**FFT-based:** $O(T \log T)$ using Fourier transform properties:

$$
\text{ACF} = \mathcal{F}^{-1}\left[|\mathcal{F}[y]|^2\right]
$$

**Practical:** Use built-in functions in statistical libraries.

---

## Autocorrelation in Forecasting

**High positive autocorrelation:**

Past values strongly predict future values. Time series models will perform well.

**Zero autocorrelation:**

Series is unpredictable from its own history. Need external predictors.

**Model validation:**

Check that forecast errors (residuals) have no autocorrelation. If they do, model is missing structure.

---

## Spurious Autocorrelation

**Trend-induced:**

Two independent trending series appear correlated.

**Solution:** Detrend or difference before computing autocorrelation.

**Seasonality-induced:**

Seasonal patterns create artificial autocorrelation.

**Solution:** Apply seasonal adjustment first.

---

## Durbin-Watson Test

Tests for first-order autocorrelation in regression residuals:

$$
DW = \frac{\sum_{t=2}^{T} (e_t - e_{t-1})^2}{\sum_{t=1}^{T} e_t^2}
$$

**Range:** $0 \leq DW \leq 4$

- $DW = 2$: No autocorrelation
- $DW < 2$: Positive autocorrelation
- $DW > 2$: Negative autocorrelation

**Rule of thumb:** $DW \approx 2(1 - \rho_1)$

---

## Autocorrelation in Residual Analysis

After fitting any time series model:

1. Compute residuals $e_t = y_t - \hat{y}_t$
2. Calculate ACF of residuals
3. Check if all $\rho_k \approx 0$

**Good model:** Residuals show no autocorrelation (white noise).

**Poor model:** Residuals have significant autocorrelation (model missed patterns).

**Action:** If residuals are autocorrelated, add more model terms or change model structure.

---

## Autocorrelation and Variance Estimation

In presence of positive autocorrelation:

$$
\text{Var}(\bar{y}) = \frac{\sigma^2}{T} \left[1 + 2\sum_{k=1}^{T-1} \left(1 - \frac{k}{T}\right) \rho_k\right]
$$

Standard error of the mean is larger than $\sigma/\sqrt{T}$.

**Implication:** Autocorrelated data provides less information than independent data.

**Effective sample size:** $T_{\text{eff}} = \frac{T}{1 + 2\sum_{k=1}^{T-1} \rho_k}$

---

## Spectral Density Connection

Autocorrelation and spectral density are Fourier transform pairs:

$$
f(\omega) = \frac{1}{2\pi} \sum_{k=-\infty}^{\infty} \rho_k e^{-i\omega k}
$$

**Interpretation:**

- ACF: Time domain representation
- Spectral density: Frequency domain representation

Both contain same information about the process.
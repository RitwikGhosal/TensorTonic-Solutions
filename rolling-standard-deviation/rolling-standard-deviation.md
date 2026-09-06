## What Is Rolling Standard Deviation?

Rolling standard deviation measures the variability of the most recent $w$ observations in a time series. It quantifies how spread out values are around their local mean within a moving window.

$$
\sigma_t(w) = \sqrt{\frac{1}{w-1} \sum_{i=0}^{w-1} (y_{t-i} - \bar{y}_t)^2}
$$

where $\bar{y}_t = \frac{1}{w} \sum_{i=0}^{w-1} y_{t-i}$ is the rolling mean.

---

## The Formula

For a time series $y_1, y_2, ..., y_T$, the rolling standard deviation at time $t$ with window $w$ is:

$$
\sigma_t = \sqrt{\frac{1}{w-1} \sum_{i=t-w+1}^{t} (y_i - \bar{y}_t)^2}
$$

**Alternative computational form:**

$$
\sigma_t = \sqrt{\frac{1}{w-1} \left(\sum_{i=t-w+1}^{t} y_i^2 - \frac{1}{w}\left(\sum_{i=t-w+1}^{t} y_i\right)^2\right)}
$$

**Sample vs population:** Denominator $w-1$ gives sample standard deviation (Bessel's correction). Use $w$ for population standard deviation.

---

## Worked Example

**Time series:** [10, 12, 15, 11, 13, 16, 14]

**Window size:** $w = 3$

**Period 3:**

Values: {10, 12, 15}

Mean: $\bar{y}_3 = \frac{10+12+15}{3} = 12.33$

$$
\sigma_3 = \sqrt{\frac{(10-12.33)^2 + (12-12.33)^2 + (15-12.33)^2}{2}}
$$

$$
= \sqrt{\frac{5.44 + 0.11 + 7.11}{2}} = \sqrt{\frac{12.66}{2}} = \sqrt{6.33} = 2.52
$$

**Period 4:**

Values: {12, 15, 11}

Mean: $\bar{y}_4 = \frac{12+15+11}{3} = 12.67$

$$
\sigma_4 = \sqrt{\frac{(12-12.67)^2 + (15-12.67)^2 + (11-12.67)^2}{2}}
$$

$$
= \sqrt{\frac{0.45 + 5.43 + 2.79}{2}} = \sqrt{\frac{8.67}{2}} = \sqrt{4.34} = 2.08
$$

**Result:** [NaN, NaN, 2.52, 2.08, 2.08, 2.52, 1.00]

---

## Interpretation

**High rolling std dev:** Data points widely spread, high variability.

**Low rolling std dev:** Data points clustered close together, low variability.

**Increasing trend in std dev:** Volatility increasing (heteroscedasticity).

**Decreasing trend:** Volatility decreasing.

**Constant std dev:** Homoscedastic (constant variance).

---

## Rolling Volatility in Finance

**Historical volatility:**

$$
\sigma_t = \sqrt{\frac{252}{w-1} \sum_{i=t-w+1}^{t} r_i^2}
$$

where $r_i$ are daily returns and 252 annualization factor (trading days per year).

**Application:**

- Risk management
- Option pricing (Black-Scholes uses volatility)
- Portfolio optimization
- VaR calculations

**Example:** 20-day rolling volatility commonly used for short-term risk assessment.

---

## Window Size Selection

**Small window (e.g., $w=5$):**

- Responsive to recent changes
- Captures short-term volatility spikes
- Noisier estimates

**Large window (e.g., $w=100$):**

- Smoother estimates
- Captures long-term volatility trends
- Less responsive to recent changes

**Trade-off:** Responsiveness vs stability.

**Common choices:**

- Intraday: $w=20$ to $w=50$ observations
- Daily financial: $w=20$ (1 month), $w=60$ (3 months), $w=252$ (1 year)
- General: Choose $w$ to match relevant time scale

---

## Computational Efficiency

**Naive approach:**

Recompute mean and variance for each window.

Time: $O(w)$ per position, $O(Tw)$ total.

**Efficient approach (Welford's algorithm):**

Update mean and variance incrementally.

$$
M_t = M_{t-1} + \frac{y_t - y_{t-w}}{w}
$$

$$
S_t = S_{t-1} + (y_t - M_{t-1})(y_t - M_t) - (y_{t-w} - M_{t-1})(y_{t-w} - M_t)
$$

$$
\sigma_t = \sqrt{\frac{S_t}{w-1}}
$$

Time: $O(1)$ per update, $O(T)$ total.

**Advantage:** Constant-time updates, numerically stable.

---

## Bollinger Bands

**Definition:**

Upper band: $\mu_t + k\sigma_t$

Middle band: $\mu_t$ (rolling mean)

Lower band: $\mu_t - k\sigma_t$

where $k=2$ typically.

**Interpretation:**

- Price near upper band: Potentially overbought
- Price near lower band: Potentially oversold
- Band width: Measure of volatility

**Squeeze:** Bands narrow (low volatility), often precedes volatility expansion.

**Expansion:** Bands widen (high volatility), indicates active market.

---

## Z-Score Normalization

**Standardize values using rolling statistics:**

$$
z_t = \frac{y_t - \mu_t}{\sigma_t}
$$

**Interpretation:**

- $z_t = 0$: Value at rolling mean
- $z_t = 2$: Value 2 standard deviations above rolling mean
- $z_t = -2$: Value 2 standard deviations below rolling mean

**Application:** Detect outliers, mean reversion trading signals.

**Example:** Buy when $z_t < -2$ (oversold), sell when $z_t > 2$ (overbought).

---

## Coefficient of Variation

**Relative variability:**

$$
\text{CV}_t = \frac{\sigma_t}{|\mu_t|}
$$

**Interpretation:** Standard deviation as percentage of mean.

**Advantage:** Scale-independent comparison across series.

**Example:**

Series A: $\mu=100$, $\sigma=10$, $\text{CV}=0.1$

Series B: $\mu=10$, $\sigma=2$, $\text{CV}=0.2$

Series B has higher relative variability despite smaller absolute std dev.

---

## Sample vs Population Standard Deviation

**Sample (unbiased estimator):**

$$
s_t = \sqrt{\frac{1}{w-1} \sum_{i=t-w+1}^{t} (y_i - \bar{y}_t)^2}
$$

**Population:**

$$
\sigma_t = \sqrt{\frac{1}{w} \sum_{i=t-w+1}^{t} (y_i - \bar{y}_t)^2}
$$

**Bessel's correction:** $w-1$ corrects bias in small samples.

**Practical:** Use sample std dev for most applications. Population std dev slightly underestimates.

---

## Rolling Variance

**Variance:** Square of standard deviation.

$$
\text{Var}_t = \sigma_t^2 = \frac{1}{w-1} \sum_{i=t-w+1}^{t} (y_i - \bar{y}_t)^2
$$

**Interpretation:** Average squared deviation from mean.

**Use case:** Variance is additive (portfolio variance), std dev is more interpretable (same units as data).

**Relationship:** $\sigma = \sqrt{\text{Var}}$

---

## GARCH vs Rolling Standard Deviation

**Rolling std dev:** Non-parametric, equal weight to observations in window.

**GARCH (Generalized Autoregressive Conditional Heteroscedasticity):**

$$
\sigma_t^2 = \omega + \alpha r_{t-1}^2 + \beta \sigma_{t-1}^2
$$

**Exponential weighting:** Recent observations weighted more heavily.

**Rolling advantages:** Simple, no parameters to estimate.

**GARCH advantages:** Captures volatility clustering, persistence, mean reversion.

**Financial practice:** GARCH often preferred for volatility forecasting.

---

## Outlier Impact

**Std dev sensitive to outliers:**

Single extreme value increases std dev substantially.

**Example:** [10, 12, 11, 13, 100]

Std dev heavily influenced by 100.

**Robust alternatives:**

**Median Absolute Deviation (MAD):**

$$
\text{MAD}_t = \text{median}(|y_i - \text{median}(\{y_{t-w+1}, ..., y_t\})|)
$$

**Interquartile range (IQR):**

$$
\text{IQR}_t = Q3_t - Q1_t
$$

**Use robust measures when outliers are concern.**

---

## Annualized Volatility

**Convert to annual volatility:**

$$
\sigma_{\text{annual}} = \sigma_{\text{period}} \times \sqrt{n}
$$

where $n$ is number of periods per year.

**Examples:**

- Daily to annual: $\sigma_{\text{annual}} = \sigma_{\text{daily}} \times \sqrt{252}$
- Monthly to annual: $\sigma_{\text{annual}} = \sigma_{\text{monthly}} \times \sqrt{12}$

**Assumes independence:** Volatility scales with square root of time.

**Example:** Daily std dev of 1.5%.

Annual: $1.5\% \times \sqrt{252} = 1.5\% \times 15.87 = 23.8\%$

---

## Control Charts

**Statistical process control:**

Monitor process stability using rolling std dev.

**Upper control limit (UCL):**

$$
\text{UCL} = \mu + 3\sigma
$$

**Lower control limit (LCL):**

$$
\text{LCL} = \mu - 3\sigma
$$

**Out of control signal:** Observation exceeds control limits.

**Application:** Manufacturing quality control, service metrics monitoring.

---

## Heteroscedasticity Detection

**Plot rolling std dev over time:**

**Increasing trend:** Variance growing (heteroscedasticity).

**Constant level:** Homoscedasticity (constant variance).

**White test:** Formal test for heteroscedasticity.

**Remedy:** Transform data (log, square root) to stabilize variance.

**Example:** Stock returns show volatility clustering (periods of high/low volatility).

---

## Expanding vs Rolling Window

**Rolling window:** Fixed size $w$, slides over data.

$$
\sigma_t = f(y_{t-w+1}, ..., y_t)
$$

**Expanding window:** Includes all data from start to time $t$.

$$
\sigma_t = f(y_1, ..., y_t)
$$

**Rolling advantages:** Adapts to changing volatility, local estimates.

**Expanding advantages:** Uses all available information, stable for stationary processes.

**Choose based on:** Is volatility changing over time? (Use rolling) Or constant? (Use expanding)

---

## Realized Volatility

**High-frequency data:**

Compute intraday returns, sum squared returns.

$$
\text{RV}_t = \sum_{i=1}^{M} r_{t,i}^2
$$

where $M$ is number of intraday observations.

**Square root:**

$$
\sigma_t = \sqrt{\text{RV}_t}
$$

**Advantage:** More accurate volatility estimates using intraday information.

**Application:** Modern volatility forecasting, option pricing.

---

## Confidence Intervals

**For normally distributed data:**

$$
\sigma_{\text{true}}^2 \in \left[\frac{(w-1)s^2}{\chi^2_{\alpha/2, w-1}}, \frac{(w-1)s^2}{\chi^2_{1-\alpha/2, w-1}}\right]
$$

**Wide intervals for small $w$:** Uncertainty high with few observations.

**Narrow intervals for large $w$:** More precise estimates.

**Practical:** Rolling std dev is point estimate. Confidence intervals quantify uncertainty.

---

## Mean Absolute Deviation Alternative

**MAD:**

$$
\text{MAD}_t = \frac{1}{w} \sum_{i=t-w+1}^{t} |y_i - \bar{y}_t|
$$

**Comparison to std dev:**

- MAD: Less sensitive to outliers
- Std dev: Penalizes large deviations more (squared term)

**Relationship:** For normal distribution, $\text{MAD} \approx 0.8 \sigma$

**Use MAD when:** Outliers are concern and robustness desired.

---

## Downside Deviation

**Semivariance:** Only considers negative deviations.

$$
\sigma_{\text{down},t} = \sqrt{\frac{1}{w} \sum_{i=t-w+1}^{t} \min(y_i - \bar{y}_t, 0)^2}
$$

**Sortino ratio:**

$$
\text{Sortino} = \frac{\bar{r} - r_f}{\sigma_{\text{down}}}
$$

**Interpretation:** Risk-adjusted return focusing on downside risk only.

**Advantage:** More relevant for investors (upside volatility is desirable, downside is risk).

---

## Rolling Covariance and Correlation

**Rolling covariance between two series:**

$$
\text{Cov}_t(x, y) = \frac{1}{w-1} \sum_{i=t-w+1}^{t} (x_i - \bar{x}_t)(y_i - \bar{y}_t)
$$

**Rolling correlation:**

$$
\rho_t = \frac{\text{Cov}_t(x, y)}{\sigma_{x,t} \sigma_{y,t}}
$$

**Application:** Portfolio diversification (correlation changes over time), pairs trading.

---

## Value at Risk

**VaR:** Maximum loss at given confidence level.

**Parametric VaR using rolling std dev:**

$$
\text{VaR}_{\alpha} = \mu_t - z_{\alpha} \sigma_t
$$

**Example:** 95% VaR with $\mu_t=0.1\%$, $\sigma_t=1.5\%$

$$
\text{VaR}_{0.05} = 0.1\% - 1.645 \times 1.5\% = -2.37\%
$$

**Interpretation:** 5% chance of losing more than 2.37% in next period.

---

## Rolling Skewness and Kurtosis

**Skewness:** Measure of asymmetry.

$$
\text{Skew}_t = \frac{1}{w} \sum_{i=t-w+1}^{t} \left(\frac{y_i - \bar{y}_t}{\sigma_t}\right)^3
$$

**Kurtosis:** Measure of tail heaviness.

$$
\text{Kurt}_t = \frac{1}{w} \sum_{i=t-w+1}^{t} \left(\frac{y_i - \bar{y}_t}{\sigma_t}\right)^4
$$

**Application:** Identify changing distribution characteristics over time.

---

## Regime Detection

**Volatility regimes:**

Low volatility: $\sigma_t < \text{threshold}_1$

Medium volatility: $\text{threshold}_1 \leq \sigma_t < \text{threshold}_2$

High volatility: $\sigma_t \geq \text{threshold}_2$

**Application:**

- Adjust trading strategies based on regime
- Risk management (reduce exposure in high volatility)
- Option strategies (sell volatility when high, buy when low)

---

## Exponentially Weighted Moving Std Dev

**EWMA variance:**

$$
\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1-\lambda) r_t^2
$$

**RiskMetrics approach:** $\lambda = 0.94$ for daily data.

**Advantages:**

- All history considered (not just window)
- Recent data weighted more
- Smooth evolution

**Comparison to rolling:** EWMA adapts faster to regime changes.

---

## Forecast Evaluation

**Rolling std dev for forecast intervals:**

$$
\text{Forecast interval} = \hat{y}_{t+h} \pm z_{\alpha/2} \sigma_t
$$

**Assumption:** Forecast error variance equals recent variance.

**Example:** Point forecast 105, rolling std dev 3.

95% interval: $105 \pm 1.96 \times 3 = [99.12, 110.88]$

**Limitation:** Assumes error distribution matches recent behavior.

---

## Multi-Scale Analysis

**Compute rolling std dev at multiple windows:**

- Short-term: $w=10$ (recent volatility)
- Medium-term: $w=50$ (intermediate volatility)
- Long-term: $w=200$ (baseline volatility)

**Compare:** Is short-term volatility above long-term? (Elevated risk)

**Application:** Adaptive risk management, identify volatility spikes relative to baseline.

---

## Practical Considerations

**Minimum window size:** At least 20-30 observations for reliable std dev estimates.

**Numerical stability:** Use two-pass or Welford's algorithm to avoid catastrophic cancellation.

**Missing data:** Handle NaN values (skip or interpolate before computing std dev).

**Frequency:** Match window size to problem timescale (hours, days, months).

**Validation:** Backtest whether rolling std dev accurately represents realized volatility.
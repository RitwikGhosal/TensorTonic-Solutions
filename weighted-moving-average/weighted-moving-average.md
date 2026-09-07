## What Is a Weighted Moving Average?

A weighted moving average (WMA) is the average of the most recent $w$ observations, where each observation is multiplied by a weight. Recent observations typically receive higher weights.

$$
\text{WMA}_t = \frac{\sum_{i=0}^{w-1} w_i \cdot y_{t-i}}{\sum_{i=0}^{w-1} w_i}
$$

where $w_i$ are the weights.

---

## The Formula

For a time series $y_1, y_2, ..., y_T$ with window size $w$:

$$
\text{WMA}_t = \frac{w_0 y_t + w_1 y_{t-1} + ... + w_{w-1} y_{t-w+1}}{w_0 + w_1 + ... + w_{w-1}}
$$

**Normalization:** Divide by sum of weights to ensure proper averaging.

**Common weight scheme (linear descending):**

$$
w_i = w - i
$$

Most recent observation has highest weight ($w$), oldest has lowest weight ($1$).

---

## Worked Example

**Time series:** [10, 12, 15, 14, 16, 18, 17]

**Window size:** $w = 4$

**Linear weights:** [4, 3, 2, 1]

**Period 4:**

$$
\text{WMA}_4 = \frac{4(14) + 3(15) + 2(12) + 1(10)}{4+3+2+1} = \frac{56 + 45 + 24 + 10}{10} = \frac{135}{10} = 13.5
$$

**Period 5:**

$$
\text{WMA}_5 = \frac{4(16) + 3(14) + 2(15) + 1(12)}{10} = \frac{64 + 42 + 30 + 12}{10} = \frac{148}{10} = 14.8
$$

**Period 6:**

$$
\text{WMA}_6 = \frac{4(18) + 3(16) + 2(14) + 1(15)}{10} = \frac{72 + 48 + 28 + 15}{10} = \frac{163}{10} = 16.3
$$

**Period 7:**

$$
\text{WMA}_7 = \frac{4(17) + 3(18) + 2(16) + 1(14)}{10} = \frac{68 + 54 + 32 + 14}{10} = \frac{168}{10} = 16.8
$$

**Result:** [NaN, NaN, NaN, 13.5, 14.8, 16.3, 16.8]

---

## Weight Schemes

**Linear (arithmetic) weights:**

$$
w_i = w - i
$$

Example ($w=5$): [5, 4, 3, 2, 1]

**Exponential weights:**

$$
w_i = \alpha^i
$$

Example ($\alpha=0.8$, $w=5$): [1, 0.8, 0.64, 0.512, 0.410]

**Triangular weights:**

$$
w_i = \max(0, w - i)
$$

Symmetric around center.

**Custom weights:**

Domain-specific weighting based on application requirements.

---

## Comparison to Simple Moving Average

**SMA:** Equal weights $w_i = 1$ for all $i$.

$$
\text{SMA}_t = \frac{1}{w} \sum_{i=0}^{w-1} y_{t-i}
$$

**WMA:** Variable weights, typically declining.

**Advantage of WMA:**

More responsive to recent changes while still using historical information.

**Disadvantage:**

Requires choosing weight scheme (additional decision).

---

## Linear vs Exponential Weights

**Linear weights ($w=5$):** [5, 4, 3, 2, 1]

Total weight: 15

Most recent: $\frac{5}{15} = 33.3\%$

**Exponential weights ($\alpha=0.5$, $w=5$):** [1, 0.5, 0.25, 0.125, 0.0625]

Total weight: 1.9375

Most recent: $\frac{1}{1.9375} = 51.6\%$

**Exponential gives more emphasis to recent observations.**

**Linear provides more gradual decay.**

---

## Relationship to Exponential Moving Average

**EMA with parameter $\alpha$:**

$$
\text{EMA}_t = \alpha y_t + (1-\alpha)\text{EMA}_{t-1}
$$

**Expanded form:**

$$
\text{EMA}_t = \alpha \sum_{i=0}^{\infty} (1-\alpha)^i y_{t-i}
$$

**Weights:** $w_i = \alpha(1-\alpha)^i$

Exponentially decaying, infinite history.

**WMA with finite exponential weights:**

Truncated version of EMA using only last $w$ observations.

**Difference:** WMA has finite memory, EMA has infinite memory (with decay).

---

## Lag Properties

**Lag in WMA:**

$$
\text{Lag} = \frac{\sum_{i=0}^{w-1} i \cdot w_i}{\sum_{i=0}^{w-1} w_i}
$$

**Linear weights ($w=5$: [5,4,3,2,1]):**

$$
\text{Lag} = \frac{0(5) + 1(4) + 2(3) + 3(2) + 4(1)}{15} = \frac{20}{15} = 1.33
$$

**SMA ($w=5$):**

$$
\text{Lag} = \frac{w-1}{2} = 2
$$

**WMA with declining weights has less lag than SMA of same window size.**

---

## Computational Efficiency

**Naive WMA:**

Compute weighted sum each period.

Time: $O(w)$ per observation, $O(Tw)$ total.

**No simple incremental update like SMA:**

Cannot use running sum trick (weights change relative position).

**Optimization:**

Maintain rolling window buffer, compute weighted sum when needed.

**Trade-off:** WMA is slower than SMA but faster than complex filters.

---

## Optimal Weight Selection

**Minimize forecast error:**

$$
\{w_0^*, ..., w_{w-1}^*\} = \arg\min_{w_i} \sum_{t=w}^{T} (y_t - \text{WMA}_{t-1})^2
$$

**Subject to:** $\sum w_i = 1$ (normalized)

**Methods:**

1. Grid search over predefined weight schemes
2. Regression: Regress $y_t$ on lagged values, use coefficients as weights
3. Cross-validation: Evaluate different schemes on validation set

**Practical:** Linear weights work well for many applications.

---

## Centered Weighted Moving Average

**Centered (non-causal):**

$$
\text{CWMA}_t = \frac{\sum_{i=-k}^{k} w_i y_{t+i}}{\sum_{i=-k}^{k} w_i}
$$

Uses observations on both sides of time $t$.

**Symmetric weights:**

$w_{-i} = w_i$

**Advantage:** No lag, better smoothing.

**Disadvantage:** Requires future values (cannot forecast).

**Use:** Historical analysis, not real-time prediction.

---

## Forecasting with WMA

**One-step ahead forecast:**

$$
\hat{y}_{t+1} = \text{WMA}_t
$$

**Multi-step forecast:**

$$
\hat{y}_{t+h} = \text{WMA}_t \text{ for all } h > 0
$$

Flat forecast (no trend).

**Assumption:** Future value equals weighted average of recent past.

**Limitation:** Like SMA, assumes stationarity. Poor for trending data.

**Better for trends:** Use weighted regression or double exponential smoothing.

---

## Volume-Weighted Average Price

**Financial application:**

$$
\text{VWAP}_t = \frac{\sum_{i=t-w+1}^{t} P_i \cdot V_i}{\sum_{i=t-w+1}^{t} V_i}
$$

where $P_i$ is price and $V_i$ is volume.

**Interpretation:** Average price weighted by trading volume.

High-volume trades more influential.

**Use case:** Benchmark for execution quality. Traders aim to beat VWAP.

---

## Time-Weighted Averages

**Irregular time intervals:**

Weight by time duration.

$$
\text{TWA}_t = \frac{\sum_{i=1}^{n} y_i \cdot \Delta t_i}{\sum_{i=1}^{n} \Delta t_i}
$$

where $\Delta t_i$ is time duration for observation $i$.

**Example:** Sensor reports at irregular intervals.

Observation lasting 10 seconds weighted twice as much as observation lasting 5 seconds.

**Application:** Process monitoring, environmental data.

---

## Triangular Moving Average

**Two-stage smoothing:**

1. Apply SMA with window $w_1$
2. Apply SMA to result with window $w_2$

**Result:** Weights form triangular shape.

**Effective window:** $w_1 + w_2 - 1$

**Weights:** Maximum at center, decreasing linearly toward edges.

**Advantage:** Smoother than single MA, no edge discontinuity like simple weights.

**Example:** $(w_1=3, w_2=3)$ gives weights proportional to [1, 2, 3, 2, 1].

---

## Hull Moving Average

**Reduces lag while maintaining smoothness:**

$$
\text{HMA}_t = \text{WMA}\left(2 \cdot \text{WMA}(w/2) - \text{WMA}(w), \sqrt{w}\right)
$$

**Steps:**

1. Compute WMA with period $w/2$
2. Compute WMA with period $w$
3. Compute $2 \times \text{WMA}_{w/2} - \text{WMA}_w$
4. Apply WMA with period $\sqrt{w}$ to result

**Advantage:** Combines smoothness with responsiveness.

**Application:** Trading systems requiring fast, smooth indicators.

---

## Gaussian-Weighted Moving Average

**Weights follow Gaussian distribution:**

$$
w_i = \exp\left(-\frac{i^2}{2\sigma^2}\right)
$$

**Normalized:**

$$
\text{GWMA}_t = \frac{\sum_{i=0}^{w-1} w_i y_{t-i}}{\sum_{i=0}^{w-1} w_i}
$$

**Parameter $\sigma$:** Controls width of Gaussian.

**Advantage:** Smooth continuous weights, well-founded in statistics.

**Application:** Image processing, signal filtering.

---

## Bias-Variance Trade-off

**High weight on recent observations:**

- Lower bias (tracks changes quickly)
- Higher variance (more noise)

**Distributed weights (more equal):**

- Higher bias (lags behind changes)
- Lower variance (more smoothing)

**Optimal weights depend on:**

- Signal-to-noise ratio
- Rate of change in underlying process
- Forecasting horizon

---

## Adaptive Weighted Moving Average

**Dynamic weight adjustment:**

Change weights based on recent forecast performance.

**Example rule:**

If recent errors are large, increase weight on most recent observation.

**Implementation:**

$$
w_{0,t} = \max\left(w_{\min}, \min\left(w_{\max}, f(|e_{t-1}|)\right)\right)
$$

where $f$ is increasing function of error magnitude.

**Trade-off:** Complexity vs adaptability.

---

## Combining Multiple WMAs

**Crossover strategies:**

- Fast WMA: Small window, high recent weight
- Slow WMA: Large window, distributed weight

**Golden cross:** Fast WMA crosses above slow WMA (bullish signal).

**Death cross:** Fast WMA crosses below slow WMA (bearish signal).

**Example:**

Fast: $w=5$ with linear weights

Slow: $w=20$ with linear weights

---

## Weighted Least Squares Relationship

**WLS regression:**

$$
\min_{\beta} \sum_{i=t-w+1}^{t} w_i (y_i - \beta_0 - \beta_1 i)^2
$$

Fit linear trend with more weight on recent observations.

**Forecast:** Extrapolate fitted line.

**Comparison to WMA:**

WLS assumes linear trend, WMA assumes level (no trend).

**Use WLS when:** Trending data.

**Use WMA when:** Stationary or slowly varying mean.

---

## Seasonal Weighted Moving Average

**Seasonal weights:**

Give higher weight to same season in previous cycles.

**Example:** Monthly data, forecast December.

$$
\text{SWMA}_{\text{Dec}} = \frac{\sum_{i=1}^{n} w_i \cdot y_{\text{Dec}, i}}{\sum_{i=1}^{n} w_i}
$$

Recent Decembers weighted more than distant Decembers.

**Application:** Seasonal forecasting with adaptation to recent seasonal patterns.

---

## Median-Weighted Alternatives

**Robust alternative:**

Instead of weighted mean, use weighted median.

**Implementation:**

1. Replicate observations according to weights
2. Compute median of replicated set

**Example:** Observation with weight 3 appears 3 times.

**Advantage:** Robust to outliers while incorporating weighting scheme.

---

## Kernel Smoothing

**General framework:**

$$
\hat{y}_t = \frac{\sum_{i=1}^{T} K\left(\frac{t-i}{h}\right) y_i}{\sum_{i=1}^{T} K\left(\frac{t-i}{h}\right)}
$$

where $K$ is kernel function and $h$ is bandwidth.

**WMA as kernel smoother:**

Kernel $K$ corresponds to chosen weight function.

**Common kernels:**

- Uniform: Equal weights (SMA)
- Triangular: Linear weights (WMA)
- Gaussian: Normal weights
- Epanechnikov: Quadratic weights

---

## Forecast Intervals

**Prediction interval:**

$$
\hat{y}_{t+1} \pm z_{\alpha/2} \sigma_{\text{forecast}}
$$

**Variance of WMA forecast:**

$$
\sigma_{\text{forecast}}^2 = \sigma^2 \left(1 + \frac{\sum w_i^2}{(\sum w_i)^2}\right)
$$

where $\sigma^2$ is variance of innovations.

**Interpretation:** Higher weight concentration increases forecast variance.

---

## Overfitting Concerns

**Too many weight parameters:**

Risk of fitting noise rather than signal.

**Regularization:**

Constrain weights to smooth function (penalize roughness).

$$
\min_{w_i} \sum (y_t - \text{WMA}_t)^2 + \lambda \sum (w_i - w_{i-1})^2
$$

**Cross-validation:** Test out-of-sample to avoid overfitting.

---

## Weighted Moving Average Regression

**Combine WMA with other predictors:**

$$
y_t = \beta_0 + \beta_1 \text{WMA}_t + \beta_2 x_t + \epsilon_t
$$

**Use WMA as feature in regression model.**

**Example:** Predict sales using weighted average of past sales plus promotional spending.

**Advantage:** Captures both momentum (WMA) and external factors (other predictors).

---

## Digital Filter Interpretation

**WMA as FIR filter:**

Finite impulse response filter with coefficients equal to weights.

**Transfer function:**

$$
H(z) = \frac{\sum_{i=0}^{w-1} w_i z^{-i}}{\sum_{i=0}^{w-1} w_i}
$$

**Frequency response:** Determines which frequencies are attenuated.

**Low-pass filter:** Removes high-frequency noise.

**Design consideration:** Choose weights to achieve desired frequency response.

---

## Weighted Differencing

**First difference with weights:**

$$
\Delta y_t = \sum_{i=0}^{w-1} w_i (y_{t-i} - y_{t-i-1})
$$

**Weighted rate of change.**

**Application:** Emphasize recent changes more than distant changes.

**Trend detection:** Identify acceleration/deceleration patterns.

---

## Implementation Considerations

**Normalization:** Always divide by sum of weights.

**Edge cases:** First $w-1$ observations typically set to NaN.

**Weight storage:** Pre-compute and store weight vector.

**Numerical stability:** Use numerically stable summation (Kahan summation for high precision).

**Vectorization:** Leverage array operations for efficiency in modern languages.

---

## Applications by Domain

**Finance:**

- VWAP for trade execution
- Technical indicators (WMA crossovers)
- Portfolio rebalancing weights

**Manufacturing:**

- Quality control with recent emphasis
- Production smoothing

**Retail:**

- Demand forecasting with recent trends
- Inventory optimization

**Energy:**

- Load forecasting (recent weather patterns)
- Price prediction

**Healthcare:**

- Patient monitoring (vital signs trending)
- Epidemiology (disease incidence smoothing)

---

## Comparison Summary

**Simple MA:** Equal weights, simple, no parameters.

**Weighted MA:** Flexible weights, more responsive, requires weight scheme.

**Exponential MA:** Infinite memory with decay, one parameter ($\alpha$).

**WMA advantages:**

- More control than SMA
- Finite memory (vs EMA)
- Intuitive weighting

**WMA disadvantages:**

- Requires choosing weights
- Slower computation than SMA
- More parameters to tune
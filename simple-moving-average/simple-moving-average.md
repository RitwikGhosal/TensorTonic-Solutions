## What Is a Simple Moving Average?

A simple moving average (SMA) is the mean of the most recent $w$ observations in a time series. It creates a smoothed version of the data by averaging out short-term fluctuations.

$$
\text{SMA}_t(w) = \frac{1}{w} \sum_{i=0}^{w-1} y_{t-i}
$$

where $w$ is the window size.

---

## The Formula

For a time series $y_1, y_2, ..., y_T$, the SMA at time $t$ with window $w$ is:

$$
\text{SMA}_t = \frac{y_t + y_{t-1} + ... + y_{t-w+1}}{w}
$$

**Example with $w = 3$:**

$$
\text{SMA}_t = \frac{y_t + y_{t-1} + y_{t-2}}{3}
$$

---

## Worked Example

**Time series:** [10, 12, 15, 14, 16, 18, 17]

**Window size:** $w = 3$

**Calculations:**

- $t=3$: $\text{SMA}_3 = \frac{10 + 12 + 15}{3} = \frac{37}{3} = 12.33$
- $t=4$: $\text{SMA}_4 = \frac{12 + 15 + 14}{3} = \frac{41}{3} = 13.67$
- $t=5$: $\text{SMA}_5 = \frac{15 + 14 + 16}{3} = \frac{45}{3} = 15.00$
- $t=6$: $\text{SMA}_6 = \frac{14 + 16 + 18}{3} = \frac{48}{3} = 16.00$
- $t=7$: $\text{SMA}_7 = \frac{16 + 18 + 17}{3} = \frac{51}{3} = 17.00$

**Result:** [NaN, NaN, 12.33, 13.67, 15.00, 16.00, 17.00]

First $w-1$ values are undefined.

---

## Purpose and Use Cases

**Smoothing:**

Remove random noise and reveal underlying trends.

**Trend identification:**

Upward slope indicates rising trend, downward slope indicates declining trend.

**Support and resistance:**

In trading, SMAs act as dynamic support/resistance levels.

**Forecast baseline:**

Simple predictions: next value = current SMA.

---

## Window Size Selection

**Small window (e.g., $w=3$):**

- More responsive to recent changes
- Follows data closely
- More noise remains

**Large window (e.g., $w=50$):**

- Smoother curve
- Less responsive (lagging indicator)
- Removes more noise

**Trade-off:** Responsiveness vs smoothness.

---

## Lag Effect

SMA is a lagging indicator: it reacts to changes after they occur.

**Delay:** Approximately $\frac{w-1}{2}$ time steps.

**Example:** With $w=10$, SMA lags by about 4.5 time steps.

**Implication:** Not suitable for real-time change detection. Trend has already changed when SMA shows it.

---

## Handling Edges

**At the start ($t < w$):**

Cannot compute full SMA.

**Options:**

1. Leave as NaN/undefined
2. Use expanding window: $\text{SMA}_t = \frac{1}{t} \sum_{i=1}^{t} y_i$
3. Pad with initial value or global mean

Most common: Leave initial values as NaN.

---

## SMA for Forecasting

**Naive forecast:**

$$
\hat{y}_{t+1} = \text{SMA}_t
$$

Predict next value as the average of recent values.

**Multi-step:**

$$
\hat{y}_{t+h} = \text{SMA}_t \text{ for all } h > 0
$$

Flat forecast (constant).

---

## Centered Moving Average

Instead of using past values only, center the window:

$$
\text{CMA}_t = \frac{1}{w} \sum_{i=-(w-1)/2}^{(w-1)/2} y_{t+i}
$$

**For odd $w=5$:**

$$
\text{CMA}_t = \frac{y_{t-2} + y_{t-1} + y_t + y_{t+1} + y_{t+2}}{5}
$$

**Advantage:** No lag, better for smoothing.

**Disadvantage:** Requires future values (not causal).

Use for retrospective analysis, not forecasting.

---

## SMA in Stock Trading

**Golden Cross:** Fast SMA (50-day) crosses above slow SMA (200-day). Bullish signal.

**Death Cross:** Fast SMA crosses below slow SMA. Bearish signal.

**Price vs SMA:**

- Price > SMA: Uptrend, bullish
- Price < SMA: Downtrend, bearish

---

## Multiple SMAs

Use several windows simultaneously:

- Short-term: $w=10$
- Medium-term: $w=50$
- Long-term: $w=200$

**Interpretation:**

- All SMAs rising: Strong uptrend
- Short > Medium > Long: Bullish alignment
- Crossovers: Trend changes

---

## SMA vs Exponential Moving Average

**SMA:** Equal weight to all $w$ observations.

**EMA:** Exponentially decaying weights, more weight to recent values.

$$
\text{EMA}_t = \alpha y_t + (1-\alpha) \text{EMA}_{t-1}
$$

**SMA advantages:**

- Simple to compute and understand
- No parameters except window size

**EMA advantages:**

- More responsive to recent changes
- Smooth continuous weights

---

## Computational Efficiency

**Naive approach:** Recompute sum each time.

Time complexity: $O(w)$ per value, $O(Tw)$ total.

**Efficient approach:** Running sum.

$$
\text{SMA}_t = \text{SMA}_{t-1} + \frac{y_t - y_{t-w}}{w}
$$

Time complexity: $O(1)$ per value, $O(T)$ total.

Subtract oldest value, add newest, divide by $w$.

---

## SMA for Seasonal Adjustment

Use SMA with window = seasonal period to remove seasonality:

**Monthly data with yearly seasonality:**

$w = 12$ averages out the seasonal pattern.

**Result:** Trend-cycle component without seasonal fluctuations.

**Caveat:** Also removes any signal at that frequency.

---

## Signal Extraction

SMA decomposes the series:

$$
y_t = \text{Trend}_t + \text{Noise}_t
$$

where $\text{Trend}_t \approx \text{SMA}_t$ and $\text{Noise}_t = y_t - \text{SMA}_t$.

**Residual analysis:** Examine $y_t - \text{SMA}_t$ for patterns.

---

## Frequency Domain Interpretation

SMA acts as a low-pass filter:

- Attenuates high-frequency components (noise, rapid changes)
- Preserves low-frequency components (trends)

**Transfer function:** Sinc function in frequency domain.

**Cutoff frequency:** Depends on $w$; larger $w$ has lower cutoff.

---

## Weighted vs Unweighted

SMA assigns equal weight $\frac{1}{w}$ to each observation.

**Truncation effect:** Observation $t-w$ has weight $\frac{1}{w}$, observation $t-w-1$ has weight $0$.

Discontinuous weights cause edge effects.

**Alternative:** Weighted MA with smooth weights (triangular, Gaussian).

---

## SMA for Anomaly Detection

**Method:**

1. Compute SMA
2. Compute standard deviation of residuals
3. Flag points where $|y_t - \text{SMA}_t| > k \sigma$

**Typical:** $k=3$ for 3-sigma rule.

**Use case:** Detecting unusual spikes or drops in metrics.

---

## Double Moving Average

Apply SMA twice:

$$
\text{SMA2}_t = \text{SMA}(\text{SMA}_t)
$$

**Effect:** Even smoother, but even more lag.

Used in double exponential smoothing context for capturing trends.
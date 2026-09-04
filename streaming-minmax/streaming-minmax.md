## What is Streaming Min-Max?

Streaming min-max computes the minimum and maximum values of a data stream without storing all elements. As each new value arrives, the running min and max are updated with O(1) time and O(1) space complexity. This is essential for processing data that is too large to fit in memory or arrives continuously.

---

## Why Streaming Algorithms?

**Memory constraints**: Cannot store all data points when processing terabytes of data

**Real-time processing**: Data arrives continuously; need instant statistics

**Infinite streams**: Logs, sensor data, and user activity have no defined end

**Distributed systems**: Aggregating statistics across shards without centralization

---

## The Streaming Min-Max Algorithm

**State**: Maintain two variables
- current_min: smallest value seen so far
- current_max: largest value seen so far

**Update rule** for each new value $x$:

$$
\text{current\_min} = \min(\text{current\_min}, x)
$$

$$
\text{current\_max} = \max(\text{current\_max}, x)
$$

**Initialization**: 
- First element: min = max = first value
- Alternative: min = +∞, max = −∞

---

## Worked Example

**Stream**: [5, 2, 8, 1, 9, 3]

**Processing**:

After element 5:
- min = 5, max = 5

After element 2:
- min = min(5, 2) = 2, max = max(5, 2) = 5

After element 8:
- min = min(2, 8) = 2, max = max(5, 8) = 8

After element 1:
- min = min(2, 1) = 1, max = max(8, 1) = 8

After element 9:
- min = min(1, 9) = 1, max = max(8, 9) = 9

After element 3:
- min = min(1, 3) = 1, max = max(9, 3) = 9

**Final result**: min = 1, max = 9

**Note**: Each step requires only the current element and previous min/max.

---

## Complexity Analysis

**Time complexity**: O(1) per element (two comparisons)

**Space complexity**: O(1) total (two values regardless of stream size)

**Total time for N elements**: O(N)

**Comparison with batch approach**:
- Batch: O(N) time, O(N) space to store all elements
- Streaming: O(N) time, O(1) space

---

## Multi-Dimensional Streams

For streams of vectors with $D$ dimensions:

**State**: Maintain $D$ min values and $D$ max values

**Update**: Apply scalar update independently to each dimension

$$
\text{min}_d = \min(\text{min}_d, x_d) \quad \text{for } d = 1, ..., D
$$

$$
\text{max}_d = \max(\text{max}_d, x_d) \quad \text{for } d = 1, ..., D
$$

**Space**: O(D) - constant with respect to stream length

---

## Initialization Strategies

**Using first element**:
- Wait for first element
- Set min = max = first element
- Simple but requires special handling for empty streams

**Using infinity**:
- Initialize min = +∞, max = −∞
- First element automatically becomes both min and max
- Works even for empty streams (returns infinity values)

**Numerical representation**:
- Use language-specific infinity values
- Or use maximum/minimum representable numbers

---

## Handling Empty Streams

**Option 1**: Return None or raise exception

**Option 2**: Return (−∞, +∞) to indicate no valid range

**Option 3**: Require at least one element before querying

**Best practice**: Document behavior and handle consistently

---

## Streaming Min-Max for Normalization

After processing the stream, use min/max for normalization:

$$
x_{normalized} = \frac{x - \text{min}}{\text{max} - \text{min}}
$$

**Challenge**: Normalization requires a second pass through the data

**Solutions**:
1. **Two-pass streaming**: First pass finds min/max, second pass normalizes
2. **Approximate bounds**: Use estimated min/max from initial samples
3. **Incremental normalization**: Update normalization as bounds change

---

## Sliding Window Min-Max

For recent data only (last W elements):

**Challenge**: Elements leave the window; min/max might need recalculation

**Naive approach**: Store last W elements, recompute min/max on each update - O(W) time

**Deque approach**: Maintain monotonic deques for efficient O(1) amortized updates

**Space**: O(W) for the sliding window

---

## Parallel/Distributed Min-Max

When processing in parallel across multiple workers:

**Local computation**: Each worker computes local min/max on its partition

**Global aggregation**:

$$
\text{global\_min} = \min(\text{min}_1, \text{min}_2, ..., \text{min}_k)
$$

$$
\text{global\_max} = \max(\text{max}_1, \text{max}_2, ..., \text{max}_k)
$$

**Properties**:
- Associative: grouping does not matter
- Commutative: order does not matter
- Enables MapReduce-style processing

---

## Exponentially Weighted Min-Max

For streams where recent values matter more:

**Exponential decay**: Older values gradually "forgotten"

**Not exact min/max**: Approximates range of recent values

**Use case**: Adapting to concept drift in changing distributions

---

## Numerical Stability

**Integer overflow**: Sum of two large integers might overflow
- Use larger integer types
- Or use floating point

**Floating point**: Generally safe for comparisons
- NaN handling: NaN comparisons return false
- Infinity handling: +∞ stays as max, −∞ stays as min

---

## Streaming Statistics Ecosystem

Min-max is one of several streaming statistics:

**Exact streaming computation**:
- Count (trivial)
- Min, Max (this problem)
- Sum

**Approximate streaming computation**:
- Mean (running average)
- Variance (Welford's algorithm)
- Quantiles (approximate with sketches)
- Distinct count (HyperLogLog)

---

## Where Streaming Min-Max Shows Up

- **Data Pipelines**: ETL processes computing statistics on large datasets

- **Monitoring Systems**: Tracking min/max latency, memory usage, or error rates

- **Sensor Networks**: Processing continuous sensor readings in real-time

- **Log Analysis**: Finding extreme values in application logs

- **Financial Systems**: Tracking price ranges, trading volumes

- **Feature Engineering**: Computing min/max for normalization in online learning

- **Quality Control**: Monitoring measurement ranges in manufacturing

- **Network Traffic**: Analyzing packet sizes, response times, bandwidth usage

- **IoT Devices**: Resource-constrained devices tracking environmental sensors

- **Real-time Analytics**: Dashboard metrics updated as data arrives

def winsorize(values: list, lower_pct: float, upper_pct: float) -> list:
    """
    Returns values clipped to the interpolated percentile bounds.
    """
    n = len(values)
    s = sorted(values)
    
    if lower_pct <= 0:
        lo = s[0]
    
    if upper_pct >= 100:
        hi = s[-1]
    
    def percentile(l, p):
        k = (n - 1)*p /100.0
        f = int(k)
        c= f + 1
        if c >= n:
            return l[f]
        return l[f] + (k - f) * (l[c] - l[f])
    lo = percentile(s, lower_pct)
    hi = percentile(s, upper_pct)

    return [max(lo, min(hi, v)) for v in values] 
    
    
import math

def poisson_pmf_cdf(lam: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    pmf = (math.exp(-lam) * lam**k)/(math.factorial(k))
    cdf = 0.0
    for i in range(0, k+1):
        cdf += (math.exp(-lam) * lam**i)/(math.factorial(i))
    return {"pmf": pmf, "cdf": cdf}
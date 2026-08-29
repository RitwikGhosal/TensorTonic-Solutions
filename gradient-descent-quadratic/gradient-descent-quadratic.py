def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Returns the final scalar x after the requested iterations.
    """
    x = float(x0)
    for _ in range(steps):
        grad = 2 * a * x + b
        x = x - lr * grad
    return x
    
        
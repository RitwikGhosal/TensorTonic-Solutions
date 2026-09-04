import math

def gaussian_kernel(size: int, sigma: float) -> list:
    """
    Returns a square two-dimensional list.
    """
    center = size // 2
    kernel = []
    total = 0.0
    for row in range(size):
        kernel_row = []
        for column in range(size):
            x = column - center
            y = row - center
            weight = math.exp(-(x**2 + y**2) / (2 * sigma**2))
            kernel_row.append(weight)
            total += weight
        kernel.append(kernel_row)
    return [[weight / total for weight in row] for row in kernel]
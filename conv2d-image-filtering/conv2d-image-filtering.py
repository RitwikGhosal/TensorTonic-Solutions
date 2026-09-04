def conv2d(image: list, kernel: list, stride: int = 1, padding: int = 0) -> list:
    """
    Returns a two-dimensional list.
    """
    height = len(image)
    width = len(image[0])
    kernel_h = len(kernel)
    kernel_w = len(kernel[0])
    padded = [[0.0] * (width + 2 * padding) for _ in range(height + 2 * padding)]
    for row in range(height):
        for column in range(width):
            padded[row + padding][column + padding] = image[row][column]
    output_height = (len(padded) - kernel_h) // stride + 1
    output_width = (len(padded[0]) - kernel_w) // stride + 1
    output = []
    for row in range(output_height):
        output_row = []
        for column in range(output_width):
            value = 0.0
            for kernel_row in range(kernel_h):
                for kernel_column in range(kernel_w):
                    value += padded[row * stride + kernel_row][column * stride + kernel_column] * kernel[kernel_row][kernel_column]
            output_row.append(value)
        output.append(output_row)
    return output
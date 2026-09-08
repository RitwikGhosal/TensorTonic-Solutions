import numpy as np

def conv2d(x: list, W: list, b: list) -> np.ndarray:
    """
    Returns the convolved batch as a floating-point NumPy array.
    """
    x = np.asarray(x, dtype = float)
    W =  np.asarray(W, dtype = float)
    b = np.asarray(b, dtype = float)
    out_height = x.shape[-2] - W.shape[-2] + 1
    out_width = x.shape[-1] - W.shape[-1] + 1
    batch_size =  x.shape[0]
    output_channels = W.shape[0]
    out = np.zeros((batch_size, output_channels, out_height, out_width), dtype = float)
    for n in range(batch_size):
        for output_channel in range(output_channels):
            for row in range(out_height):
                for column in range(out_width):
                    patch = x[n, :, row:row+W.shape[-2], column: column + W.shape[-1]]
                    out[n, output_channel, row, column] = (np.sum(patch * W[output_channel]) + b[output_channel])
    return out
    
    
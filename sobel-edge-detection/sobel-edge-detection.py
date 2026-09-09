import math

def sobel_edges(image: list) -> list:
    """
    Returns the zero-padded Sobel gradient magnitude at every pixel.
    """
    H, W = len(image), len(image[0])
    padded = [[0 for _ in range(W+2)] for _ in range(H+2)]
    for i in range(H):
        for j in range(W):
            padded[i+1][j+1] = image[i][j]
    k_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0 ,1]]
    k_y = [[-1, -2, -1], [0,0,0], [1, 2, 1]]
    res=  []
    for i in range(H):
        row = []
        for j in range(W):
            g_x = 0.0
            g_y = 0.0
            for di in range(3):
                for dj in range(3):
                    g_x += padded[i+di][j+dj] * k_x[di][dj]
                    g_y += padded[i+di][j+dj] * k_y[di][dj]
            row.append(math.sqrt(g_x**2 + g_y**2))
        res.append(row)
    return res
                    
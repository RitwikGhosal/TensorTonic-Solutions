def histogram_equalize(image: list) -> list:
    """
    Returns the histogram-equalized grayscale image.
    """
    height = len(image)
    width = len(image[0])
    hist = [0] * 256
    for row in image:
        for col in row:
            hist[col] += 1
    cdf = [0] * 256
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + hist[i]
    cdf_min = next(val for val in cdf if val > 0)
    denom = (height * width) - cdf_min 
    new = [0] * 256
    if denom > 0:
        for i in range(256):
            if cdf[i] > 0:
                new[i] = round((cdf[i] - cdf_min) / denom * 255)
    return [[new[i] for i in row] for row in image]
    
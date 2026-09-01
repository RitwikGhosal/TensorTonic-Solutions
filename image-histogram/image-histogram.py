def image_histogram(image: list) -> list:
    """
    Returns a list of intensity and count pairs.
    """
    counts = [0] * 256
    for row in image:
        for pixel in row:
            counts[pixel] += 1

    return [[intensity, count] for intensity, count in enumerate(counts) if count]
def color_to_grayscale(image: list) -> list:
    """
    Returns the luminance value of every RGB pixel.
    """
    res = []
    for i in range(len(image)):
        row = []
        for j in range(len(image[0])):
            r, g, b = image[i][j][0], image[i][j][1], image[i][j][2] 
            gray = 0.299*r + 0.587*g + 0.114*b
            row.append(round(gray, 6))
        res.append(row)
    return res
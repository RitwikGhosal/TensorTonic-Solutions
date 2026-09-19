import math

def rotate_image(image, angle_degrees):
    """
    Rotate the image counterclockwise by the given angle using nearest neighbor interpolation.
    """
    h, w = len(image), len(image[0])
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    theta = math.radians(angle_degrees)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    out = []
    for i in range(h):
        row = []
        for j in range(w):
            dy, dx = i - cy, j - cx
            src_y = cy + dy * cos_t + dx * sin_t
            src_x = cx - dy * sin_t + dx * cos_t
            sy, sx = round(src_y), round(src_x)
            row.append(image[sy][sx] if 0 <= sy < h and 0 <= sx < w else 0)
        out.append(row)
    return out
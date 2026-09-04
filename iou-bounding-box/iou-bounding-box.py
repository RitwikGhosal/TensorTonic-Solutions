def iou(box_a: list, box_b: list) -> float:
    """
    Returns IoU as a float.
    """

    left = max(box_a[0], box_b[0])
    right = min(box_a[2], box_b[2])
    top = max(box_a[1], box_b[1])
    bottom = min(box_a[3], box_b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0
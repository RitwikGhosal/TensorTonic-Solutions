def robust_scaling(values: list) -> list:
    s = sorted(values)
    n = len(values)

    if n % 2 == 1:
        median = s[n // 2]
    else:
        median = (s[n // 2 - 1] + s[n // 2]) / 2.0

    first_half = s[:n // 2]
    second_half = s[(n + 1) // 2:]

    q1_len = len(first_half)
    if q1_len == 0:
        q1 = 0.0
    elif q1_len % 2 == 1:
        q1 = first_half[q1_len // 2]
    else:
        q1 = (first_half[q1_len // 2 - 1] + first_half[q1_len // 2]) / 2.0

    q3_len = len(second_half)
    if q3_len == 0:
        q3 = 0.0
    elif q3_len % 2 == 1:
        q3 = second_half[q3_len // 2]
    else:
        q3 = (second_half[q3_len // 2 - 1] + second_half[q3_len // 2]) / 2.0

    iqr = q3 - q1

    if iqr == 0:
        return [v - median for v in values]

    return [(v - median) / iqr for v in values]
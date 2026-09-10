def word_count_dict(sentences: list) -> dict:
    """
    Returns a dictionary of token counts.
    """
    counts = {}
    for sentence in sentences:
        for word in sentence:
            counts[word] = counts.get(word, 0) + 1
    return counts
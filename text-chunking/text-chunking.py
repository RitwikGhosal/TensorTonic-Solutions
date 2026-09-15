def text_chunking(tokens: list, chunk_size: int, overlap: int) -> list:
    """
    Returns fixed-size token chunks with the requested overlap.
    """
    step = chunk_size - overlap
    out = []
    for i in range(0, len(tokens), step):
        chunk = tokens[i: i + chunk_size]
        out.append(chunk)
        if i + chunk_size >= len(tokens):
            break 
    return out
    
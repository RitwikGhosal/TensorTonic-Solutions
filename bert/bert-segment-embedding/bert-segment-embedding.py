import numpy as np

def bert_embeddings(token_ids: np.ndarray, segment_ids: np.ndarray,
                    token_embeddings: np.ndarray, position_embeddings: np.ndarray,
                    segment_embeddings: np.ndarray) -> np.ndarray:
    """
    Returns the float64 BERT input embeddings with shape (B, S, H).
    """
    seq_len = token_ids.shape[1]
    token_values = token_embeddings[token_ids]
    position_values = position_embeddings[np.arange(seq_len)][None, :, :]
    segment_values = segment_embeddings[segment_ids]
    return token_values + position_values + segment_values
    
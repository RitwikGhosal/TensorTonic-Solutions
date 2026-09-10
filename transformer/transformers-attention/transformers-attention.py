import torch
import math
import torch.nn.functional as F

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns the scaled dot-product attention output.
    """
    d_k = K.shape[-1]
    score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    att = F.softmax(score, dim = -1)
    res = torch.matmul(att, V)
    return res
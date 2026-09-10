import numpy as np

def softmax(x, axis = -1):
    e_x = np.exp(x - np.max(x, axis = axis, keepdims = True))
    return e_x / np.sum(e_x, axis = axis, keepdims = True)    

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns projected multi-head attention outputs.
    """
    d_k = K.shape[-1]//num_heads
    batch_size, seq_len, d_model = K.shape 

    K_proj = np.dot(K, W_k)
    Q_proj = np.dot(Q, W_q)
    V_proj = np.dot(V, W_v)

    # initially - (batch, num_heads, seq_len, head_dim)
    K = K_proj.reshape(batch_size, seq_len, num_heads, d_model//num_heads).transpose(0, 2, 1, 3) 
    Q = Q_proj.reshape(batch_size, seq_len, num_heads, d_model//num_heads).transpose(0, 2, 1, 3)
    V = V_proj.reshape(batch_size, seq_len, num_heads, d_model//num_heads).transpose(0, 2, 1, 3)

    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    attention = softmax(scores, axis= -1)
    attention_out = np.matmul(attention, V)

    attention_out = attention_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    return np.dot(attention_out, W_o)
        
        
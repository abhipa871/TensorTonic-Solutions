import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here
    Q = Q@W_q
    K = K@W_k
    V = V@W_v
    seq_len = Q.shape[1]
    d_k = Q.shape[-1]//num_heads
    Q = Q.reshape(-1, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K = K.reshape(-1, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V = V.reshape(-1, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)    
    
    scores = softmax((Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(d_k))@V
    mha = scores.transpose(0,2,1,3).reshape(-1, seq_len, num_heads*d_k)
    return mha@W_o
    

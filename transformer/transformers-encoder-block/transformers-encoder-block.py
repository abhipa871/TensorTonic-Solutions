import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    # Your code here
    return gamma*(x-np.mean(x, axis=-1, keepdims=True))/np.sqrt(np.var(x, axis=-1, keepdims=True)+eps)+beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
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

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    ffn = np.maximum(0, x@W1+b1)@W2+b2
    return ffn

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    # Your code here
    mha = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    norm1 = layer_norm(mha+x, gamma1, beta1)
    ffn = feed_forward(norm1, W1, b1, W2, b2)
    output = layer_norm(norm1+ffn, gamma2, beta2)
    return output
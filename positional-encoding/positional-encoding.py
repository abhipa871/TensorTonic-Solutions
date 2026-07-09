import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    pos = np.arange(seq_len)[:,None]
    dim = np.repeat(np.arange(0, d_model, 2), repeats=2)[:d_model][None, :]
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2]  = np.sin(pos/(np.power(base, (dim[:, 0::2]/d_model))))
    pe[:, 1::2] = np.cos(pos/(np.power(base, (dim[:, 1::2]/d_model))))
    return pe
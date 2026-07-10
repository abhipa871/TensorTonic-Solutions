import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    positional_encoding = np.zeros((seq_length, d_model))
    pos = np.arange(seq_length)[:, None]
    dim = np.arange(d_model, step=2)[None]
    positional_encoding[:, 0::2] = np.sin(pos/(np.power(10000, dim/d_model)))
    positional_encoding[:, 1::2] = np.cos(pos/(np.power(10000, dim[:, :positional_encoding[:, 1::2].shape[-1]]/d_model)))

    
    return positional_encoding
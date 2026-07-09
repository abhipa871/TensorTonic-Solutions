import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.array(x)
    prob = rng.random(size = x.shape) if rng else np.random.random(size=x.shape)
    dropout_pattern = np.where(prob<p, 0, 1/(1-p))
    output = x*dropout_pattern
    return output, dropout_pattern

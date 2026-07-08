import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    out = 1/(1+np.exp(np.array(x)*-1))
    return out
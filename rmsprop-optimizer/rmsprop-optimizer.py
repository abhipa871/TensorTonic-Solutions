import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    # Write code here
    s = np.array(s)
    g = np.array(g)
    s = beta*s + (1-beta)*(g**2)
    w-=(lr/np.sqrt(s+eps))*g
    return (w,s)
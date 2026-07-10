import numpy as np

def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def _as2d(a, feat):
    """Convert 1D array to 2D and track if conversion happened"""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, feat)
    return a

import numpy as np

def gru_cell_forward(x, h_prev, params):
    x = np.asarray(x)
    h_prev = np.asarray(h_prev)

    x,_ = _as2d(x, x.shape[-1])
    h_prev,_ = _as2d(h_prev, h_prev.shape[-1])
        
    zt = _sigmoid(x@params['Wz'] + h_prev@params['Uz'] + params['bz'])
    rt = _sigmoid(x@params['Wr'] + h_prev@params['Ur'] + params['br'])
    
    ht_cand = np.tanh(x@params['Wh'] + (rt*h_prev)@params['Uh'] + params['bh'])
    ht = (1-zt)*h_prev + zt*ht_cand
    
    return ht.squeeze()
    
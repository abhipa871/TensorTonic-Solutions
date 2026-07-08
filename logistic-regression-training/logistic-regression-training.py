import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    w, b = np.zeros(X.shape[1]), np.zeros(1)
    for i in range(steps):
        output = _sigmoid(X@w + b)
        w-=lr*(X.T@(output-y)/X.shape[0])
        b-=lr*np.mean(output-y)

    return (w,b)
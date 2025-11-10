import numpy as np

def fweierstrass(x: np.ndarray) -> np.ndarray:
    NP, D = x.shape
    a = 0.5
    b = 3.0
    k_max = 20
    c1 = a ** np.arange(0, k_max + 1)
    c2 = 2.0 * np.pi * b ** np.arange(0, k_max + 1)
    # compute sum over k for each dimension, then sum over dimensions
    t = np.zeros((NP, D))
    c1_row = c1[None, :]
    c2_row = c2[None, :]
    for i in range(D):
        xi = x[:, i:i+1] + 0.5
        t[:, i] = np.sum(c1_row * np.cos(c2_row * xi), axis=1)
    return np.sum(t, axis=1) - D * np.sum(c1 * np.cos(c2 * 0.5))

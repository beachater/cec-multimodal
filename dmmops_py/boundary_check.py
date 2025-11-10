import numpy as np

def boundary_check(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Reflect solutions into [lb, ub] elementwise replicating MATLAB logic.
    Parameters
    ----------
    x : (N,D) array
    lb, ub : (D,) arrays
    Returns
    -------
    (N,D) array within bounds via reflection.
    """
    x = np.asarray(x, dtype=float)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    assert x.shape[1] == lb.shape[0] == ub.shape[0]
    length_bound = ub - lb
    # vectorized reflection (MATLAB implementation using logical masks and remainder)
    below = x < lb
    if np.any(below):
        x = below * (lb + np.remainder(lb - x, length_bound)) + (~below) * x
    above = x > ub
    if np.any(above):
        x = above * (ub - np.remainder(x - ub, length_bound)) + (~above) * x
    return x

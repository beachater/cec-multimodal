import numpy as np

def fgrienwank(x: np.ndarray) -> np.ndarray:
    D = x.shape[1]
    f1 = np.sum(x ** 2 / 4000.0, axis=1)
    f2 = np.prod(np.cos(x / np.sqrt(np.arange(1, D + 1))), axis=1)
    return f1 - f2 + 1.0

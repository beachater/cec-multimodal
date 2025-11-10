import numpy as np

def frastrigin(x: np.ndarray) -> np.ndarray:
    return np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x) + 10.0, axis=1)

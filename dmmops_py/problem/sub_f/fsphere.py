import numpy as np

def fsphere(x: np.ndarray) -> np.ndarray:
    return np.sum(x ** 2, axis=1)

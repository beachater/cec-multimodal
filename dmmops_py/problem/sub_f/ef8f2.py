import numpy as np

def f8f2(x: np.ndarray) -> np.ndarray:
    f2 = 100.0 * (x[:, 0] ** 2 - x[:, 1]) ** 2 + (1 - x[:, 0]) ** 2
    return 1 + (f2 ** 2) / 4000.0 - np.cos(f2)

def ef8f2(x: np.ndarray) -> np.ndarray:
    # Sum of F8F2 over consecutive pairs with wrap-around, x shifted by +1 as in MATLAB
    D = x.shape[1]
    total = np.zeros(x.shape[0])
    for i in range(D - 1):
        total += f8f2(x[:, [i, i + 1]] + 1)
    total += f8f2(x[:, [D - 1, 0]] + 1)
    return total

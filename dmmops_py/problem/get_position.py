import numpy as np
from .utils import pdist2


def get_position(N: int, D: int, lb: np.ndarray, ub: np.ndarray, dpeak: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos = np.zeros((N, D))
    unsatisfied_index = np.arange(N)
    while unsatisfied_index.size > 0:
        pos[unsatisfied_index, :] = rng.random((unsatisfied_index.size, D)) * (ub - lb) + lb
        pos_dis = pdist2(pos, pos)
        # For each row, ensure all distances > dpeak; rows with any distance <= dpeak remain unsatisfied
        ok = np.all(pos_dis > dpeak + np.eye(N) * 1e9, axis=1)
        unsatisfied_index = np.where(~ok)[0]
    return pos

import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.io import loadmat


def pdist2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return cdist(a, b)


def resolve_data_file(*relative_parts: str) -> Path:
    """Resolve a data file path trying package data and original MATLAB data paths."""
    here = Path(__file__).parent
    # 1) package data folder (data/)
    p1 = here / 'Data' / Path(*relative_parts)
    if p1.exists():
        return p1
    # 2) original MATLAB folder dmmops/problem/Data
    p2 = here.parent.parent / 'dmmops' / 'problem' / 'Data' / Path(*relative_parts)
    if p2.exists():
        return p2
    # 3) same directory
    p3 = here / Path(*relative_parts)
    if p3.exists():
        return p3
    raise FileNotFoundError(f"Data file not found: {relative_parts}")


def load_mat_variable(filename: str, varname: str):
    p = resolve_data_file(filename)
    mat = loadmat(p)
    if varname not in mat:
        # some MAT files may store without struct naming; try keys
        raise KeyError(f"Variable '{varname}' not found in {p}")
    return mat[varname]

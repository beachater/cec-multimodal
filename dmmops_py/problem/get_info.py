from typing import Tuple


def get_info(fn: int) -> Tuple[int, int, int]:
    """Return (fun_num, change_type, dimension) mapping as MATLAB get_info.m."""
    flist = [1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 1, 2, 3, 4, 5, 6, 7, 8]
    clist = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 1, 1, 1, 1, 1, 1]
    dlist = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 10, 10, 10]
    idx = fn - 1
    return flist[idx], clist[idx], dlist[idx]

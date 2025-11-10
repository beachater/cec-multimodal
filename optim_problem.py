"""Optimization problem base class."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from data_structures import GlobalMinimaData


@dataclass
class OptimProblem:
    """Base class for optimization problems."""
    suite: str = 'internal'
    PID: int = 0
    dim: int = 0
    lowBound: np.ndarray | None = None
    upBound: np.ndarray | None = None
    maxEval: int = 0
    globMinData: GlobalMinimaData | None = None
    numCallF: int = 0
    isDynamic: bool = False
    knownChanges: bool = False

    def __init__(self, pid: int, D: int | None = None):
        self.suite = 'internal'
        self.PID = pid
        if D is not None:
            self.dim = D
            self.lowBound = np.full(D, np.nan)
            self.upBound = np.full(D, np.nan)
        self.globMinData = GlobalMinimaData()
        self.numCallF = 0
        self.isDynamic = False
        self.knownChanges = False

    def set_problem_data(self):
        """Calculate and set the problem information."""
        if self.PID == 100:
            self.lowBound = 0.25 * np.ones(self.dim)
            self.upBound = 10.0 * np.ones(self.dim)
            self.maxEval = 200000 * (self.dim - 1)
            self.globMinData.Ngmin = 6 ** self.dim
            self.globMinData.Rnich = 0.1
            self.globMinData.val = -1.0
        elif self.PID == 101:
            self.lowBound = -10.0 * np.ones(self.dim)
            self.upBound = 10.0 * np.ones(self.dim)
            self.maxEval = 200000 * (self.dim - 1)
            self.globMinData.Ngmin = 1
            self.globMinData.Rnich = 0.1
            self.globMinData.val = 0.0

    def func_eval(self, x: np.ndarray) -> float | np.ndarray:
        """Objective function: Evaluate the solution x."""
        self.numCallF += x.shape[0] if x.ndim == 2 else 1
        if self.PID == 100:
            return -np.mean(np.sin(10 * np.log(x)), axis=-1)
        elif self.PID == 101:
            if x.ndim == 1:
                return 100 * np.sum((x[:-1] ** 2 - x[1:]) ** 2) + np.sum((x[:-1] - 1) ** 2)
            else:
                return 100 * np.sum((x[:, :-1] ** 2 - x[:, 1:]) ** 2, axis=1) + np.sum((x[:, :-1] - 1) ** 2, axis=1)
        return np.nan

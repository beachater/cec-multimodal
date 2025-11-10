"""Dummy archive for CEC competition reporting format."""
from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .optim_problem import OptimProblem


class DummyArchive:
    """Archive used only to report found solutions according to CEC'2020 template."""
    
    def __init__(self, problem: OptimProblem):
        self.solution = np.zeros((0, problem.dim))
        self.value = np.array([])
        self.foundEval = np.array([])
        self.foundEval2 = np.array([])
        self.foundTime = np.array([])
        self.actionCode = np.array([])

    def append(self, action: int, index: int | list | np.ndarray, archive, problem: OptimProblem):
        """Append solutions with proper action code to the dummy archive."""
        if isinstance(index, int):
            index = [index]
        index = np.asarray(index)
        
        self.solution = np.vstack([self.solution, archive.solution[index, :]])
        self.value = np.concatenate([self.value, archive.value[index]])
        self.foundEval = np.concatenate([self.foundEval, archive.foundEval[index]])
        self.foundEval2 = np.concatenate([self.foundEval2, np.full(len(index), problem.numCallF)])
        self.foundTime = np.concatenate([self.foundTime, archive.foundTime[index]])
        self.actionCode = np.concatenate([self.actionCode, np.full(len(index), action)])

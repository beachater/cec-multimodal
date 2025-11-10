"""Optimization problem for DMMOP (CEC 2022 dynamic benchmark)."""
from __future__ import annotations
import numpy as np
import sys
from pathlib import Path

# Add dmmops_py to path
sys.path.insert(0, str(Path(__file__).parent / 'dmmops_py'))

from optim_problem import OptimProblem
from data_structures import GlobalMinimaData
from dmmops_py.problem.dmmop import DMMOP


class OptimProblemDMMOP(OptimProblem):
    """Optimization problem using DMMOP external benchmark."""
    
    def __init__(self, function_number: int):
        """
        Initialize DMMOP problem.
        
        Args:
            function_number: DMMOP function number (1-24)
        
        Note: Dimension is determined automatically by get_info() based on function number.
        """
        self.suite = 'DMMOP'
        self.isDynamic = True
        self.PID = function_number
        
        # Initialize DMMOP problem (dimension is set by get_info)
        self.extProb = DMMOP(function_number)
        
        # Set problem parameters
        self.lowBound = self.extProb.lower
        self.upBound = self.extProb.upper
        self.dim = self.extProb.D
        self.maxEval = self.extProb.evaluation
        
        # Initialize from parent
        self.numCallF = 0
        self.knownChanges = True
        self.globMinData = GlobalMinimaData()
        self.globMinData.Ngmin = self.extProb.o.shape[0] if self.extProb.o is not None else 0

    def func_eval(self, x: np.ndarray) -> float | np.ndarray:
        """Objective function: Evaluate using DMMOP."""
        f = self.extProb.GetFits(x if x.ndim == 2 else x[None, :])
        if f.size == 0:
            return np.inf if x.ndim == 1 else np.full(x.shape[0], np.inf)
        f = -f  # DMMOP returns negative fitness
        self.numCallF = self.extProb.evaluated
        return f[0] if x.ndim == 1 and f.size == 1 else f

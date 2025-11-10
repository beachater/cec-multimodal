"""Dynamic option and optimization option classes."""
from __future__ import annotations
from dataclasses import dataclass
from core_search import CoreSearchCMSA
from data_structures import Niching, Archiving, StopCriteria
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optim_problem import OptimProblem


@dataclass
class DynamicOption:
    """Options peculiar to dynamic problems."""
    benchSolSizeCoeff: float = 1.0
    recStepSizeCoeff: float = 0.5
    tolChangeF: float = 1e-5
    predictMethod: str = "AMLP"
    chCheckFr: int = 10
    maxPL: int = 4


class OptimOption:
    """Options for the control parameters of the algorithm."""
    
    def __init__(self, problem: OptimProblem, coreSearchName: str = 'CMSA'):
        self.archiving = Archiving(problem)
        if coreSearchName == 'CMSA':
            self.coreSearch = CoreSearchCMSA(problem)
        else:
            raise ValueError(f'Option for core search is invalid: {coreSearchName}')
        self.niching = Niching(problem)
        self.stopCr = StopCriteria(problem)
        self.dyna = DynamicOption()

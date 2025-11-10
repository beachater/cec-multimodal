"""Core search algorithms: base class and CMSA variant."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .optim_problem import OptimProblem


@dataclass
class CoreSearch:
    """Options for all potential core search algorithms."""
    algorithm: str = 'CMSA'
    targetNumSubpop: int = 1
    iniSubpopSizeCoeff: float = 6.0
    finSubpopSizeCoeff: float = 6.0
    iniSigCoeff: float = 2.0
    muToPopSizeRatio: float = 0.2
    maxIniSigma: float = 0.3
    objValUpLimit: float = 1e100
    repairInfeasMethodCode: int = 1

    def __init__(self, problem: OptimProblem):
        pass


@dataclass
class CoreSearchCMSA(CoreSearch):
    """Strategy parameters peculiar to CMSA (if it is employed as the core search algorithm)."""
    tauSigmaCoeff: float = 0.5
    eltRatio: float = 0.1
    tauCovCoeff: float = 1.0
    sigmaUpdateBiasImp: float = 1.0

    def __init__(self, problem: OptimProblem):
        super().__init__(problem)
        self.algorithm = 'CMSA'
        self.tauSigmaCoeff = 0.5
        self.eltRatio = 0.1
        self.tauCovCoeff = 1.0
        self.sigmaUpdateBiasImp = 1.0


@dataclass
class SpecStrParCMSA:
    """Update the parameters of the core search algorithm: elite CMSA."""
    numElt: int = 0
    tauCov: float = 0.0
    tauSigma: float = 0.0

    def __init__(self, process, opt, problem: OptimProblem):
        self.update(process, opt, problem)

    def update(self, process, opt, problem: OptimProblem):
        self.numElt = int(np.floor(np.ceil(opt.coreSearch.eltRatio * process.subpopSize)))
        self.tauCov = 1.0 + problem.dim * (1.0 + problem.dim) / (2.0 * process.muEff * opt.coreSearch.tauCovCoeff)
        self.tauSigma = np.sqrt(0.5 * opt.coreSearch.tauSigmaCoeff / problem.dim)

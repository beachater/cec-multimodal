"""Core data structures for AMLP-RS-CMSA-ESII algorithm (Python translation)."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, TYPE_CHECKING
import time

if TYPE_CHECKING:
    from .optim_problem import OptimProblem


@dataclass
class GlobalMinimaData:
    """Information about the global minima (not used during optimization, only for performance evaluation)."""
    Rnich: float = np.nan
    Ngmin: int = np.nan
    val: float = np.nan
    X: np.ndarray | None = None


@dataclass
class TabooRegion:
    """Information of taboo regions for a subpopulation."""
    center: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    normTabDis: np.ndarray = field(default_factory=lambda: np.array([]))
    criticality: np.ndarray = field(default_factory=lambda: np.array([]))
    criticInd: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    def __init__(self, D: int):
        self.center = np.zeros((0, D))
        self.normTabDis = np.array([])
        self.criticality = np.array([])
        self.criticInd = np.array([], dtype=int)


@dataclass
class MergeCheck:
    """Data for checking which archive point should the subpopulation merge into."""
    bestCandidArchIndHist: List[int] = field(default_factory=list)
    bestCandidArchMergeabilityHist: List[float] = field(default_factory=list)
    matchArchInd: int | None = None
    checkAfterIterNo: int = 0
    candidArchCountHist: List[int] = field(default_factory=list)
    mergeAtUsedEval: float = np.inf


@dataclass
class LocalConvergeCheck:
    """Data for checking local convergence."""
    stopAtUsedEval: float = np.inf


@dataclass
class MergePredictor:
    """Options for merging a subpopulation to an archived solution."""
    threshold: float = 0.5
    windowSizeCoeff: float = 0.1
    maxEval: int = 10
    chkIntervalCoeff: float = 0.1
    addedConst: float = 1.0


@dataclass
class LocalConvergencePredictor:
    """Options for local convergence predictor."""
    tolCoeff: float = 0.04
    windowSizeCoeff: float = 0.5
    tabooCriticUpLimit: float = 0.01


@dataclass
class Niching:
    """Options for diversity preservation."""
    criticTabooThresh: float = 0.01
    redCoeff: float = field(init=False)
    iniR0IncFac: float = 1.04
    maxRejectIni: int = 100

    def __init__(self, problem: OptimProblem):
        self.criticTabooThresh = 0.01
        self.iniR0IncFac = 1.04
        self.maxRejectIni = 100
        self.redCoeff = 0.99 ** (1.0 / problem.dim)


@dataclass
class Archiving:
    """Options for archiving the best solution of each subpopulation when a restart concludes."""
    hillVallBudget: int = 10
    iniNormTabDis: float = 1.0
    newNormTabDisPrc: float = 10.0
    targetNewNicheFr: float = 0.5
    targetGlobFr: float = 0.5
    tauNormTabDis: float = field(init=False)
    tolFunArch: float = 1e-5
    neighborSize: int = 5

    def __init__(self, problem: OptimProblem):
        self.hillVallBudget = 10
        self.iniNormTabDis = 1.0
        self.newNormTabDisPrc = 10.0
        self.targetNewNicheFr = 0.5
        self.targetGlobFr = 0.5
        self.tolFunArch = 1e-5
        self.neighborSize = 5
        self.tauNormTabDis = (1.0 / problem.dim) ** 0.5


@dataclass
class StopCriteria:
    """Stopping criterion for termination of a subpopulation."""
    tolHistFun: float = 1e-6
    tolHistSizePar: Tuple[int, int] = (10, 30)
    maxCondC: float = 1e14
    maxIterPar: Tuple[int, int] = (100, 50)
    tolX: float = 1e-12
    stagPar: Tuple[float, float, float] = (120.0, 0.2, 30.0)
    merge: MergePredictor = field(default_factory=MergePredictor)
    localConverge: LocalConvergencePredictor = field(default_factory=LocalConvergencePredictor)

    def __init__(self, problem: OptimProblem):
        self.tolHistFun = 1e-6
        self.tolHistSizePar = (10, 30)
        self.maxCondC = 1e14
        self.maxIterPar = (100, 50)
        self.tolX = 1e-12
        self.stagPar = (120.0, 0.2, 30.0)
        self.merge = MergePredictor()
        self.localConverge = LocalConvergencePredictor()

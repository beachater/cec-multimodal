"""Mutation profile and sampling classes."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


@dataclass
class MutationProfile:
    """Mutation profile of the subpopulation."""
    smean: float
    stretch: np.ndarray
    C: np.ndarray = field(init=False)
    Cinv: np.ndarray = field(init=False)
    rotMat: np.ndarray = field(init=False)

    def __init__(self, smean: float, stretch: np.ndarray):
        self.smean = smean
        self.stretch = stretch
        self.C = np.diag(self.stretch ** 2)
        self.Cinv = np.diag(self.stretch ** (-2))
        self.rotMat = np.eye(len(stretch))


@dataclass
class Sampling:
    """Information of the sampled solutions from the subpopulation."""
    s: np.ndarray
    X: np.ndarray
    Z: np.ndarray
    f: np.ndarray
    isFeas: np.ndarray
    wasRepaired: np.ndarray
    argsortNoElite: np.ndarray | None = None
    argsortWithElite: np.ndarray | None = None

    def __init__(self, popSize: int, dim: int):
        self.s = np.full(popSize, np.nan)
        self.X = np.full((popSize, dim), np.nan)
        self.Z = np.full((popSize, dim), np.nan)
        self.f = np.full(popSize, np.inf)
        self.isFeas = np.full(popSize, np.nan)
        self.wasRepaired = np.zeros(popSize, dtype=bool)
        self.argsortNoElite = None
        self.argsortWithElite = None


@dataclass
class Elite:
    """Elite solutions of the subpopulations."""
    sol: np.ndarray
    val: np.ndarray = field(default_factory=lambda: np.array([]))
    s: np.ndarray = field(default_factory=lambda: np.array([]))
    Z: np.ndarray = field(init=False)
    wasRepaired: np.ndarray = field(default_factory=lambda: np.array([]))

    def __init__(self, D: int):
        self.sol = np.zeros((0, D))
        self.Z = np.zeros((0, D))
        self.val = np.array([])
        self.s = np.array([])
        self.wasRepaired = np.array([])

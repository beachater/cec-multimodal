"""
OptimProcess class managing the optimization process state and parameters.

Written by Ali Ahrari (aliahrari1983@gmail.com)
Translated to Python from MATLAB
"""

import numpy as np
import time
from core_search import SpecStrParCMSA
from dynamic_manager import DynamicManager
from utility_methods import UtilityMethods


class OptimProcess:
    """Manages optimization process after performing a restart and updating the archive."""
    
    def __init__(self, opt, problem):
        """Initialize optimization process."""
        self.restartNo = 0
        self.subpopSize = int(opt.coreSearch.finSubpopSizeCoeff * np.sqrt(problem.dim))
        self.mu = max(1, int(0.5 + self.subpopSize * opt.coreSearch.muToPopSizeRatio))
        
        # Recombination weights
        W = np.log(1 + self.mu) - np.log(np.arange(1, self.mu + 1))
        self.recWeights = W / np.sum(W)
        self.muEff = np.sum(self.recWeights) ** 2 / np.sum(self.recWeights ** 2)
        
        # Core search-specific parameters
        if opt.coreSearch.algorithm == 'CMSA':
            self.coreSpecStrPar = SpecStrParCMSA(self, opt, problem)
        else:
            raise ValueError(f"Core search algorithm {opt.coreSearch.algorithm} not implemented")
        
        self.defNormTabDis = opt.archiving.iniNormTabDis
        self.usedEvalTillRestart = 0
        self.bestValTillRestart = np.inf
        self.iniR0 = np.sqrt(problem.dim) / 2
        self.startTime = round(time.time() * 1000)
        self.dynamics = DynamicManager(opt, problem)
    
    def reset_static(self, opt, problem):
        """Reset static optimization for new time step."""
        self.restartNo = 0
        self.subpopSize = int(opt.coreSearch.finSubpopSizeCoeff * np.sqrt(problem.dim))
        self.mu = max(1, int(0.5 + self.subpopSize * opt.coreSearch.muToPopSizeRatio))
        
        W = np.log(1 + self.mu) - np.log(np.arange(1, self.mu + 1))
        self.recWeights = W / np.sum(W)
        self.muEff = np.sum(self.recWeights) ** 2 / np.sum(self.recWeights ** 2)
        
        self.coreSpecStrPar.update(self, opt, problem)
        self.defNormTabDis = opt.archiving.iniNormTabDis
        self.bestValTillRestart = np.inf
        self.iniR0 = np.sqrt(problem.dim) / 2
    
    def update_due_to_change(self, restart, archive, opt, problem):
        """Update process due to dynamic change."""
        usedEvalThisRestart = (archive.usedEvalHist[-1] if len(archive.usedEvalHist) > 0 else 0) + \
                              restart.usedEvalEvolve + restart.usedEvalMerge + restart.usedEvalChangeDetect
        self.usedEvalTillRestart += usedEvalThisRestart
    
    def update(self, restart, archive, opt, problem):
        """Update process for the upcoming restart."""
        self.restartNo += 1
        
        # Update subpop size for the upcoming restart
        usedEvalThisRestart = (archive.usedEvalHist[-1] if len(archive.usedEvalHist) > 0 else 0) + \
                              restart.usedEvalEvolve + restart.usedEvalMerge + restart.usedEvalChangeDetect
        usedEvalSoFar = self.usedEvalTillRestart + usedEvalThisRestart
        
        # Geometric interpolation for subpop size coefficient
        subpopSizeCoeff = opt.coreSearch.iniSubpopSizeCoeff * \
                          (opt.coreSearch.finSubpopSizeCoeff / opt.coreSearch.iniSubpopSizeCoeff) ** \
                          (usedEvalSoFar / problem.maxEval)
        self.subpopSize = int(subpopSizeCoeff * np.sqrt(problem.dim))
        self.mu = max(1, int(0.5 + self.subpopSize * opt.coreSearch.muToPopSizeRatio))
        
        # Calculate recombination weights and muEff
        W = np.log(1 + self.mu) - np.log(np.arange(1, self.mu + 1))
        self.recWeights = W / np.sum(W)
        self.muEff = np.sum(self.recWeights) ** 2 / np.sum(self.recWeights ** 2)
        
        self.coreSpecStrPar.update(self, opt, problem)
        
        # Update default normalized taboo distance
        self.defNormTabDis = UtilityMethods.lin_prctile(archive.normTabDis, opt.archiving.newNormTabDisPrc)
        
        self.usedEvalTillRestart += usedEvalThisRestart
        self.bestValTillRestart = min(restart.bestVal, self.bestValTillRestart)
        self.iniR0 = min(restart.recIniR0 * opt.niching.iniR0IncFac, 0.5 * np.sqrt(problem.dim))

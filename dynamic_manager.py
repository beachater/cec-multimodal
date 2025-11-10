"""
Dynamic Manager for handling dynamic optimization problems with prediction.

Written by Ali Ahrari (aliahrari1983@gmail.com)
Translated to Python from MATLAB
"""

import numpy as np
from prediction import Prediction


class DynamicManager:
    """Manages dynamic environment changes and predictions."""
    
    def __init__(self, opt, problem):
        """Initialize dynamic manager."""
        self.endArchive = []  # Archives at the end of each time step
        self.endSolNorm = []  # End archive solutions sorted and normalized wrt search range
        self.chDetectSolXf = np.zeros((0, problem.dim + 1))  # Candidate solutions for change detection
        self.currentTimeStep = 0  # Current time step
        self.recCenter = np.zeros((0, problem.dim))  # Recommended centers for next subpopulation
        self.recStepSize = np.array([])  # Recommended step sizes
        self.recInd = 1  # Index of recommended solution for current restart
        self.usedEvalChDetect = 0  # Evaluations used by change detection
        self.lastChCheckAtFE = 0  # Last time (evaluation) change mechanism was called
        self.usedPredictLevel = []  # Selected prediction level
        
    def update(self, archive, opt, problem):
        """Call this when a change occurs."""
        self.currentTimeStep += 1
        self.recInd = 1
        
        # Sort and normalize archive solutions
        ind = np.argsort(archive.value)
        N = len(ind)
        
        if N > 0:
            tmp = (archive.solution[ind, :] - problem.lowBound) / (problem.upBound - problem.lowBound)
            self.endSolNorm.append(tmp)
        else:
            self.endSolNorm.append(np.zeros((0, problem.dim)))
        
        self.endArchive.append(archive)
        self.gen_rec_ini_pop(opt, problem)
        self.chDetectSolXf = np.zeros((0, problem.dim + 1))
        
    def detect_change(self, opt, problem):
        """Change detection mechanism."""
        N = self.chDetectSolXf.shape[0]
        ind = np.random.randint(0, N)
        
        self.usedEvalChDetect += 1
        self.lastChCheckAtFE = problem.numCallF
        
        if problem.suite == 'GMPB':
            if problem.extProb.RecentChange:
                problem.extProb.RecentChange = 0
                return True
            return False
        elif problem.suite == 'DMMOP':
            hasChanged = problem.extProb.CheckChange(
                self.chDetectSolXf[ind, :problem.dim],
                self.chDetectSolXf[ind, problem.dim]
            )
            if not problem.knownChanges:
                problem.func_eval(self.chDetectSolXf[ind, :problem.dim])
            return hasChanged
        else:
            testF = problem.func_eval(self.chDetectSolXf[ind, :problem.dim])
            return abs(testF - self.chDetectSolXf[ind, problem.dim]) > opt.dyna.tolChangeF
    
    def track_past_history(self, index, endSolNorm):
        """Find the time history of one solution given history of multiple solutions."""
        maxLevel = len(endSolNorm)
        D = endSolNorm[-1].shape[1] if endSolNorm[-1].shape[0] > 0 else 0
        histX = np.full((maxLevel, D), np.nan)
        
        if endSolNorm[-1].shape[0] > 0:
            histX[maxLevel - 1, :] = endSolNorm[-1][index, :]
            
            for L in range(maxLevel - 1, 1, -1):
                if endSolNorm[L - 2].shape[0] > 0:
                    dis = np.linalg.norm(endSolNorm[L - 2] - histX[L - 1, :], axis=1)
                    ind = np.argmin(dis)
                    histX[L - 2, :] = endSolNorm[L - 2][ind, :]
        
        return histX
    
    def gen_rec_ini_pop(self, opt, problem):
        """Generate recommended centers and step sizes for future subpopulations."""
        self.usedPredictLevel = []
        maxLevelLimit1 = 0
        
        val1 = opt.dyna.maxPL + 1
        val2 = len(self.endSolNorm)
        
        for k in range(min(val1, val2)):
            if len(self.endSolNorm) > 0 and self.endSolNorm[-(k + 1)].shape[0] > 0:
                maxLevelLimit1 += 1
            else:
                break
        
        maxPL = min(opt.dyna.maxPL, maxLevelLimit1)
        maxDataPoint = min(opt.dyna.maxPL + 1, maxLevelLimit1)
        
        if maxLevelLimit1 == 0:
            self.recCenter = np.zeros((0, problem.dim))
            self.recStepSize = np.array([])
            self.usedPredictLevel = []
        elif maxLevelLimit1 == 1:
            # Use the last solutions from previous time steps
            Nsol = self.endSolNorm[-1].shape[0]
            self.recCenter = self.endSolNorm[-1] * (problem.upBound - problem.lowBound) + problem.lowBound
            self.recStepSize = opt.coreSearch.maxIniSigma * np.ones(Nsol)
            self.usedPredictLevel.append(1)
        elif maxLevelLimit1 > 1:
            Nsol = self.endSolNorm[-1].shape[0]
            self.recCenter = np.full((Nsol, problem.dim), np.nan)
            self.recStepSize = np.full(Nsol, np.nan)
            
            for solNo in range(Nsol):
                Xhist = self.track_past_history(
                    solNo,
                    self.endSolNorm[-maxDataPoint:]
                )
                xhatNow, estPreErrNowNorm, bestLevel = Prediction.AMLP(Xhist, maxPL)
                
                if opt.dyna.predictMethod == "FLP":
                    L = len(estPreErrNowNorm)
                elif opt.dyna.predictMethod == "AMLP":
                    L = bestLevel
                else:
                    L = bestLevel
                
                self.recCenter[solNo, :] = xhatNow[L - 1, :] * (problem.upBound - problem.lowBound) + problem.lowBound
                recStepSize_val = min(
                    max(estPreErrNowNorm[L - 1] * opt.dyna.recStepSizeCoeff,
                        opt.coreSearch.maxIniSigma * 1e-6),
                    opt.coreSearch.maxIniSigma
                )
                self.recStepSize[solNo] = recStepSize_val
                self.usedPredictLevel.append(L)

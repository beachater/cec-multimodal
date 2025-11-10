"""
Archive class for storing and managing identified global/desirable minima with normalized taboo distances.

Written by Ali Ahrari (aliahrari1983@gmail.com)
Translated to Python from MATLAB
"""

import numpy as np
import time
from dummy_archive import DummyArchive


class Archive:
    """Archive of presumably distinct solutions referring to distinct global and desirable minima."""
    
    def __init__(self, problem):
        """Initialize archive for a given problem."""
        self.solution = np.zeros((0, problem.dim))  # archived solutions
        self.value = np.array([])  # values of the archived solutions
        self.normTabDis = np.array([])  # normalized taboo distance of the archived solutions
        self.foundEval = np.array([])  # When the archived solution was added (evaluation number according to algorithm internal count)
        self.foundEval2 = np.array([])  # according to numCallF
        self.usedEvalHist = np.array([])  # the number of evaluations used by the archive (analyzing final solutions)
        self.hitTimesSoFar = np.array([])  # the number of times an archived solution was detected from the beginning
        self.hitTimesThisRestart = np.array([])  # the number of times an archived solution was detected in the last restart
        self.size = 0  # number of solutions in the archive
        self.foundTime = np.array([])  # the time (in ms) the global minimum was found
        self.dummyArchive = DummyArchive(problem)  # Dummy archive for CEC'2020 format reporting
        
    def append(self, sol, val, restart, process, opt, problem):
        """Append a new solution to the archive."""
        self.solution = np.vstack([self.solution, sol])
        self.value = np.append(self.value, val)
        self.normTabDis = np.append(self.normTabDis, process.defNormTabDis)
        
        # When it was appended (internal count of evaluations)
        foundEval_val = (self.usedEvalHist[-1] if len(self.usedEvalHist) > 0 else 0) + \
                        restart.usedEvalMerge + restart.usedEvalEvolve + process.usedEvalTillRestart
        self.foundEval = np.append(self.foundEval, foundEval_val)
        self.foundEval2 = np.append(self.foundEval2, problem.numCallF)
        
        self.size += 1
        self.hitTimesThisRestart = np.append(self.hitTimesThisRestart, 0)
        self.hitTimesSoFar = np.append(self.hitTimesSoFar, 0)
        self.foundTime = np.append(self.foundTime, round(time.time() * 1000))
        
    def update(self, restart, process, opt, problem):
        """Update the archive after a restart."""
        self.usedEvalHist = np.append(self.usedEvalHist, 0)
        self.hitTimesThisRestart = np.zeros(self.size)
        
        bestValSoFar = min(restart.bestVal if restart.bestVal != np.inf else np.inf,
                           process.bestValTillRestart)
        
        # Find and discard archived solutions that are not desirable
        if self.size > 0:
            keepIt = (self.value - opt.archiving.tolFunArch) < bestValSoFar
            discardInd = np.where(~keepIt)[0]
            
            # Flag these solutions for removal in dummy archive
            if len(discardInd) > 0:
                self.dummyArchive.append(-1, discardInd, self, problem)
            
            # Discard the solutions that are not global minima
            self.solution = self.solution[keepIt, :]
            self.value = self.value[keepIt]
            self.normTabDis = self.normTabDis[keepIt]
            self.foundEval = self.foundEval[keepIt]
            self.foundEval2 = self.foundEval2[keepIt]
            self.hitTimesThisRestart = self.hitTimesThisRestart[keepIt]
            self.hitTimesSoFar = self.hitTimesSoFar[keepIt]
            self.foundTime = self.foundTime[keepIt]
            self.size = len(self.value)
        
        # Check the best solution of the subpopulation
        Ndesirable = 0
        chkEvolved = restart.iterNo > 1
        chkIsGlobal = (restart.bestVal - opt.archiving.tolFunArch) <= bestValSoFar
        
        # Check if it is a new global minimum
        if chkEvolved and chkIsGlobal:
            Ndesirable = 1
            
            if self.size == 0:  # archive is empty
                isNew = True
                self.append(restart.bestSol, restart.bestVal, restart, process, opt, problem)
                self.dummyArchive.append(1, self.size - 1, self, problem)
            else:  # archive is not empty
                # Check if it is a new solution
                isNew, matchArchNo, usedEval0Total = self.is_new_basin(
                    restart.bestSol, restart.bestVal, restart, opt, problem)
                self.usedEvalHist[-1] += usedEval0Total
                
                if not isNew:  # shares basin with existing archived solution
                    self.hitTimesSoFar[matchArchNo] += 1
                    self.hitTimesThisRestart[matchArchNo] += 1
                    
                    # If better than the already archived solution, replace it
                    if restart.bestVal < (self.value[matchArchNo] - opt.stopCr.tolHistFun):
                        self.dummyArchive.append(-1, matchArchNo, self, problem)
                        
                        # Replace the old solution
                        self.value[matchArchNo] = restart.bestVal
                        self.solution[matchArchNo, :] = restart.bestSol
                        self.foundEval[matchArchNo] = self.usedEvalHist[-1] + restart.usedEvalMerge + \
                                                      restart.usedEvalEvolve + restart.usedEvalChangeDetect + \
                                                      process.usedEvalTillRestart
                        self.foundTime[matchArchNo] = round(time.time() * 1000)
                        
                        # Append the better solution to dummy archive
                        self.dummyArchive.append(1, matchArchNo, self, problem)
                else:  # it is a new global basin
                    self.append(restart.bestSol, restart.bestVal, restart, process, opt, problem)
                    self.dummyArchive.append(1, self.size - 1, self, problem)
        
        # Adapt the normalized taboo distances
        if Ndesirable == 0:  # subpopulation did not converge to a global minimum
            self.normTabDis = self.normTabDis * np.exp(-opt.archiving.tauNormTabDis * 
                                                        opt.archiving.targetGlobFr / max(self.size, 1))
        elif isNew:  # converged to a new global minimum
            pass  # don't change normalized taboo distances
        elif not isNew:  # converged to an already detected global minimum
            repDiff = self.hitTimesThisRestart.copy()
            if np.any(self.hitTimesThisRestart == 0):
                repDiff[self.hitTimesThisRestart == 0] = (-(1 - opt.archiving.targetNewNicheFr)) / \
                                                          max(self.size - 1, 1)
            self.normTabDis = self.normTabDis * np.exp(opt.archiving.tauNormTabDis * repDiff)
    
    def is_new_basin(self, x, f, restart, opt, problem):
        """
        Check if solution (x, f) shares the basin with one of the archived solutions.
        
        Returns:
            isNew: bool - whether it's a new basin
            sameArchInd: int or None - index of archive sharing basin with x
            usedEval0Total: int - total evaluations used by hill-valley heuristic
        """
        sameArchInd = None
        
        # Check against each archived solution from closest to farthest
        dis = np.linalg.norm(self.solution - x, axis=1)
        candidInd = np.argsort(dis)
        candidInd = candidInd[:min(opt.archiving.neighborSize, len(candidInd))]
        
        usedEval0Total = 0
        
        for archNo in candidInd:
            usedEval0 = 0
            isNew = False
            
            while usedEval0 < opt.archiving.hillVallBudget:
                r = 0.8 * np.random.rand() + 0.1  # random number between 0.1 and 0.9
                testX = self.solution[archNo, :] + r * (x - self.solution[archNo, :])
                testF = problem.func_eval(testX)
                usedEval0 += 1
                
                if testF > (max(f, self.value[archNo]) + opt.stopCr.tolHistFun):
                    isNew = True
                    break
            
            usedEval0Total += usedEval0
            
            if not isNew:  # shares basin with this archive solution
                sameArchInd = archNo
                break
        
        return isNew, sameArchInd, usedEval0Total

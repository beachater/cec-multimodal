"""
Restart class for managing restart-dependent information.

Written by Ali Ahrari (aliahrari1983@gmail.com)
Translated to Python from MATLAB
"""

import numpy as np
from subpopulation_cmsa import SubpopulationCMSA


class Restart:
    """Create the restart object with all required restart-dependent information."""
    
    def __init__(self, process, opt, problem):
        """Initialize restart object."""
        self.stagSize = int(opt.stopCr.stagPar[0] + opt.stopCr.stagPar[2] * problem.dim / process.subpopSize)
        self.tolHistSize = int(opt.stopCr.tolHistSizePar[0] + opt.stopCr.tolHistSizePar[1] * problem.dim / process.subpopSize)
        self.usedEvalEvolve = 0
        self.usedEvalMerge = 0
        self.usedEvalChangeDetect = 0
        self.iterNo = 0
        self.terminationFlag = 0
        self.bestVal = np.inf
        self.bestSol = np.zeros((0, problem.dim))
        self.recIniR0 = np.nan
    
    def initialize_subpop(self, archive, process, opt, problem):
        """Initialize subpopulations for the restart."""
        # Rescale archived solutions to [0,1]^D
        archSolRescaled = np.zeros((archive.size, problem.dim))
        for k in range(archive.size):
            archSolRescaled[k, :] = (archive.solution[k, :] - problem.lowBound) / (problem.upBound - problem.lowBound)
        
        # Generate the center using Maximin strategy
        numReject = 0
        R0 = process.iniR0
        wasSuccess = False
        
        while not wasSuccess:
            X = np.random.rand(problem.dim)  # Candidate center - random sampling
            
            # Check different requirements
            chkDis = True
            
            # Requirement: center should be outside taboo regions of archived solutions
            if archive.size > 0:
                dis2dis = np.linalg.norm(archSolRescaled - X, axis=1)
                chkDis = np.all(dis2dis > (archive.normTabDis * R0))
            
            if chkDis:  # Center is acceptable
                wasSuccess = True
                self.recIniR0 = R0
            else:
                numReject += 1
            
            if numReject > opt.niching.maxRejectIni:  # Too many successive rejections
                numReject = 0
                R0 *= opt.niching.redCoeff
        
        # Scale back to the problem search range
        center = X * (problem.upBound - problem.lowBound) + problem.lowBound
        smean = min(opt.coreSearch.maxIniSigma, R0 * opt.coreSearch.iniSigCoeff)
        stretch = problem.upBound - problem.lowBound
        
        # Create the subpopulation based on the employed core search
        if opt.coreSearch.algorithm == 'CMSA':
            subpop = SubpopulationCMSA(center, smean, stretch, process.subpopSize)
        else:
            raise ValueError(f"Core search algorithm {opt.coreSearch.algorithm} not implemented")
        
        return subpop
    
    def run_one_restart(self, subpop, archive, process, opt, problem):
        """Perform one restart."""
        while self.terminationFlag == 0:  # If restart has not been terminated
            self.iterNo += 1
            self.stagSize = int(opt.stopCr.stagPar[0] + opt.stopCr.stagPar[1] * self.iterNo +
                               opt.stopCr.stagPar[2] * problem.dim / process.subpopSize)
            
            # Evolve the subpopulation
            subpop.update_taboo_region(self, archive, process, opt, problem)
            subpop.update_merge_check(self, archive, opt, problem)
            subpop.evolve(self, archive, process, opt, problem)
            
            self.usedEvalEvolve = subpop.usedEvalEvolve
            subpop.update_term_flag(self, archive, process, opt, problem)
            self.usedEvalMerge = subpop.usedEvalMerge
            self.usedEvalChangeDetect = subpop.usedEvalChangeDetect
            
            self.bestVal = subpop.bestVal
            self.bestSol = subpop.bestSol
            self.terminationFlag = subpop.terminationFlag
            
            # Break if not enough evaluation budget left
            remainEvalAfter = problem.maxEval - (process.usedEvalTillRestart + 
                                                 self.usedEvalEvolve + 
                                                 self.usedEvalMerge + 
                                                 process.subpopSize)
            reqEvalForDetectMult = min(archive.size, opt.archiving.neighborSize) * opt.archiving.hillVallBudget
            
            if problem.numCallF >= problem.maxEval:
                print(f"[{problem.numCallF}]")
                self.terminationFlag = -10  # Terminate restart due to shortage of evaluation budget
                break

"""
Subpopulation class for evolutionary optimization with niching and taboo regions.

Written by Ali Ahrari (aliahrari1983@gmail.com)
Translated to Python from MATLAB
"""

import numpy as np
from data_structures import TabooRegion, MergeCheck, LocalConvergeCheck
from mutation_sampling import MutationProfile, Sampling
from utility_methods import UtilityMethods


class Subpopulation:
    """A subpopulation object managing evolution of solutions."""
    
    def __init__(self, center, smean, stretch, popSize):
        """Initialize subpopulation."""
        D = len(center)
        self.center = center
        self.mutProfile = MutationProfile(smean, stretch)
        self.samples = Sampling(popSize, D)
        self.bestSol = np.zeros((0, D))
        self.bestVal = np.inf
        self.tabooRegion = TabooRegion(D)
        self.mergeCheck = MergeCheck()
        self.localConvergeCheck = LocalConvergeCheck()
        self.bestValNonEliteHist = []
        self.medValNonEliteHist = []
        self.maxCriticalityHist = []
        self.terminationFlag = 0  # 0: not terminated, 1: converged, -1: no improvement, etc.
        self.usedEvalEvolve = 0
        self.usedEvalMerge = 0
        self.usedEvalChangeDetect = 0
        self.iterNo = 0
    
    def calc_norm_dis(self, x1, x2, disMetric):
        """Calculate normalized distance between points given mutation profile."""
        if disMetric == 'Mahalanobis':
            diff = x1 - x2
            normDis = np.sqrt(diff @ self.mutProfile.Cinv @ diff.T) / self.mutProfile.smean
        elif disMetric == 'Euclidean':
            mean_str = np.exp(np.mean(np.log(self.mutProfile.stretch)))  # geometric mean
            normDis = np.linalg.norm(x1 - x2) / (self.mutProfile.smean * mean_str)
        else:
            raise ValueError('Distance metric is not valid')
        return normDis
    
    def is_taboo_acceptable(self, sample, tempRedRatio, opt):
        """Check if sample solution is taboo acceptable."""
        tabAccept = True
        for tabInd in self.tabooRegion.criticInd:
            normDis = self.calc_norm_dis(sample, self.tabooRegion.center[tabInd, :], 'Mahalanobis')
            tabAccept = normDis >= (self.tabooRegion.normTabDis[tabInd] * tempRedRatio)
            if not tabAccept:
                break
        return tabAccept
    
    def estimate_taboo_regions_criticality(self, opt, problem):
        """Estimate the criticality of the taboo points."""
        from scipy.stats import norm
        L = np.zeros(len(self.tabooRegion.normTabDis))
        for k in range(len(L)):
            L[k] = self.calc_norm_dis(self.tabooRegion.center[k, :], self.center, 'Mahalanobis')
        
        intU = L + self.tabooRegion.normTabDis
        intL = L - self.tabooRegion.normTabDis
        self.tabooRegion.criticality = norm.cdf(intU) - norm.cdf(intL)
    
    def update_taboo_region(self, restart, archive, process, opt, problem):
        """Determine critical taboo regions for the subpopulation."""
        maxCount = archive.size
        self.tabooRegion.center = np.zeros((maxCount, problem.dim))
        self.tabooRegion.normTabDis = np.zeros(maxCount)
        count = 0
        
        for k in range(maxCount):
            if archive.value[k] < self.bestVal:
                self.tabooRegion.center[count, :] = archive.solution[k, :]
                self.tabooRegion.normTabDis[count] = archive.normTabDis[k]
                count += 1
        
        # Discard unused indexes
        self.tabooRegion.center = self.tabooRegion.center[:count, :]
        self.tabooRegion.normTabDis = self.tabooRegion.normTabDis[:count]
        
        # Determine which taboo regions are critical
        if count > 0:
            self.estimate_taboo_regions_criticality(opt, problem)
            crInd = np.argsort(1 - self.tabooRegion.criticality)
            Ncr = np.sum(self.tabooRegion.criticality > opt.niching.criticTabooThresh)
            self.tabooRegion.criticInd = crInd[:Ncr]
        else:
            self.tabooRegion.criticInd = np.array([], dtype=int)
        
        max_crit = np.max(self.tabooRegion.criticality) if count > 0 else 0
        self.maxCriticalityHist.append(max(max_crit, 0))
    
    def update_merge_check(self, restart, archive, opt, problem):
        """Update mergeCheck property to determine likely basin sharing."""
        if archive.size > 0:
            L = np.zeros(archive.size)
            for k in range(archive.size):
                L[k] = self.calc_norm_dis(archive.solution[k, :], self.center, 'Mahalanobis')
            
            Mergeability = (archive.normTabDis + opt.stopCr.merge.addedConst) / L
            indMax = np.argmax(Mergeability)
            maxMergeability = Mergeability[indMax]
            candidArchCount = np.sum(Mergeability > opt.stopCr.merge.threshold)
        else:
            indMax = -1
            maxMergeability = 0
            candidArchCount = 0
        
        self.mergeCheck.bestCandidArchIndHist.append(indMax)
        self.mergeCheck.bestCandidArchMergeabilityHist.append(maxMergeability)
        self.mergeCheck.candidArchCountHist.append(candidArchCount)
    
    def evolve(self, restart, archive, process, opt, problem):
        """Evolve the subpopulation (mutation, selection, recombination)."""
        self.sample_solutions(restart, archive, process, opt, problem)
        self.eval_solutions(restart, archive, process, opt, problem)
        self.select(restart, archive, process, opt, problem)
        self.recombine(restart, archive, process, opt, problem)
    
    def repair_infeas(self, solNo, opt, problem):
        """Repair infeasible solution."""
        if opt.coreSearch.repairInfeasMethodCode > 0:
            if np.any(problem.upBound < self.samples.X[solNo, :]) or \
               np.any(problem.lowBound > self.samples.X[solNo, :]):
                self.samples.wasRepaired[solNo] = True
                
                # Relocate elements greater than upper bound
                relocItU = np.where(self.samples.X[solNo, :] > problem.upBound)[0]
                if len(relocItU) > 0:
                    tmpUp = problem.upBound[relocItU]
                    tmpLow = 2 * self.center[relocItU] - problem.upBound[relocItU]
                    tmpLow = np.maximum(tmpLow, problem.lowBound[relocItU])
                    self.samples.X[solNo, relocItU] = tmpLow + np.random.rand(len(relocItU)) * (tmpUp - tmpLow)
                
                # Relocate elements smaller than lower bound
                relocItL = np.where(self.samples.X[solNo, :] < problem.lowBound)[0]
                if len(relocItL) > 0:
                    tmpLow = problem.lowBound[relocItL]
                    tmpUp = 2 * self.center[relocItL] - problem.lowBound[relocItL]
                    tmpUp = np.minimum(tmpUp, problem.upBound[relocItL])
                    self.samples.X[solNo, relocItL] = tmpLow + np.random.rand(len(relocItL)) * (tmpUp - tmpLow)
                
                # Update strategy parameter
                self.samples.Z[solNo, :] = (self.samples.X[solNo, :] - self.center) / self.samples.s[solNo]
    
    def eval_solutions(self, restart, archive, process, opt, problem):
        """Perform evaluation of sampled solutions."""
        for solNo in range(process.subpopSize):
            # Calculate bound violation
            penU = self.samples.X[solNo, :] - problem.upBound
            penU = penU * (penU > 0)
            penL = problem.lowBound - self.samples.X[solNo, :]
            penL = penL * (penL > 0)
            penUL = np.sum(penU + penL)
            
            self.samples.isFeas[solNo] = not (penUL > 0)
            
            if not self.samples.isFeas[solNo]:
                self.samples.f[solNo] = opt.coreSearch.objValUpLimit * (1 + penUL)
            else:
                oldNumCallF = problem.numCallF
                self.samples.f[solNo] = problem.func_eval(self.samples.X[solNo, :])
                self.usedEvalEvolve += (problem.numCallF - oldNumCallF)
            
            # Check for potential change in the problem
            if problem.isDynamic:
                chk1 = self.samples.f[solNo] != opt.coreSearch.objValUpLimit and \
                       process.dynamics.chDetectSolXf.shape[0] >= (opt.dyna.benchSolSizeCoeff * process.subpopSize)
                chk2 = np.random.rand() < (1 / opt.dyna.chCheckFr)
                
                if chk1 and chk2:
                    oldNumCallF = problem.numCallF
                    hasChanged = process.dynamics.detect_change(opt, problem)
                    self.usedEvalChangeDetect += (problem.numCallF - oldNumCallF)
                    if hasChanged:
                        self.terminationFlag = -5
                        break
                
                if not chk1:
                    tmp = np.hstack([self.samples.X[solNo, :], self.samples.f[solNo]])
                    process.dynamics.chDetectSolXf = np.vstack([process.dynamics.chDetectSolXf, tmp])
        
        self.iterNo += 1
        
        # Update indicators for stagnation check
        self.bestValNonEliteHist.append(np.min(self.samples.f))
        self.medValNonEliteHist.append(np.median(self.samples.f))
        
        # Drop old values if array is too long
        maxSize = max(restart.stagSize, restart.tolHistSize)
        if len(self.bestValNonEliteHist) > maxSize:
            self.bestValNonEliteHist = self.bestValNonEliteHist[-maxSize:]
            self.medValNonEliteHist = self.medValNonEliteHist[-maxSize:]
    
    def update_term_flag(self, restart, archive, process, opt, problem):
        """Update the termination flag of the subpopulation."""
        # Condition number of C
        condC = (np.max(self.mutProfile.stretch) / np.min(self.mutProfile.stretch)) ** 2
        if condC > opt.stopCr.maxCondC:
            self.terminationFlag = -2
        
        # Stagnation check
        if self.iterNo >= restart.stagSize and self.terminationFlag == 0:
            N0 = len(self.bestValNonEliteHist)
            ind1 = np.arange(N0 - restart.stagSize, N0 - restart.stagSize + 20)
            ind2 = np.arange(N0 - 20, N0)
            minImpBest = np.median([self.bestValNonEliteHist[i] for i in ind2]) - \
                         np.median([self.bestValNonEliteHist[i] for i in ind1])
            minImpMed = np.median([self.medValNonEliteHist[i] for i in ind2]) - \
                        np.median([self.medValNonEliteHist[i] for i in ind1])
            if min(minImpBest, minImpMed) > 0:
                self.terminationFlag = -1
        
        # tolHistFun convergence criterion
        if self.iterNo >= restart.tolHistSize and self.terminationFlag == 0:
            maxDiff = UtilityMethods.peak2peak(
                self.bestValNonEliteHist[-restart.tolHistSize:]
            )
            if maxDiff < opt.stopCr.tolHistFun:
                self.terminationFlag = 1
        
        # Step-size criterion
        if (np.max(self.mutProfile.stretch) * self.mutProfile.smean) < opt.stopCr.tolX and \
           self.terminationFlag == 0:
            self.terminationFlag = 2
        
        # Check for potential merge with archived solution
        if self.terminationFlag == 0:
            chk1 = self.iterNo >= (opt.stopCr.merge.windowSizeCoeff * restart.tolHistSize)
            N1 = int(np.ceil(opt.stopCr.merge.windowSizeCoeff * restart.tolHistSize))
            chk2 = False
            if chk1 and len(self.mergeCheck.candidArchCountHist) >= N1:
                chk2 = all([c == 1 for c in self.mergeCheck.candidArchCountHist[-N1:]])
            chk3 = self.bestVal < opt.coreSearch.objValUpLimit
            chk4 = self.iterNo > self.mergeCheck.checkAfterIterNo
            chk5 = self.mergeCheck.mergeAtUsedEval == np.inf
            
            if chk1 and chk2 and chk3 and chk4 and chk5:
                usedEval = 0
                isNew = False
                
                stp = 1 / opt.stopCr.merge.maxEval
                r = np.arange(stp / 2, 1, stp)
                
                endX1 = self.bestSol
                endF1 = self.bestVal
                endX2 = archive.solution[self.mergeCheck.bestCandidArchIndHist[-1], :]
                endF2 = archive.value[self.mergeCheck.bestCandidArchIndHist[-1]]
                
                while usedEval < opt.stopCr.merge.maxEval:
                    testX = r[usedEval] * endX1 + (1 - r[usedEval]) * endX2
                    testF = problem.func_eval(testX)
                    self.usedEvalMerge += 1
                    usedEval += 1
                    
                    if testF > (max(endF1, endF2) + opt.stopCr.tolHistFun):
                        isNew = True
                        break
                
                if not isNew:
                    if self.mergeCheck.matchArchInd is None:
                        self.mergeCheck.matchArchInd = self.mergeCheck.bestCandidArchIndHist[-1]
                        self.mergeCheck.mergeAtUsedEval = self.usedEvalMerge + self.usedEvalEvolve
                        self.terminationFlag = 3
                        self.bestSol = archive.solution[self.mergeCheck.matchArchInd, :]
                        self.bestVal = archive.value[self.mergeCheck.matchArchInd]
                else:
                    chkInterval = opt.stopCr.merge.chkIntervalCoeff * restart.tolHistSize
                    self.mergeCheck.checkAfterIterNo = chkInterval + self.iterNo
        
        # Predict if likely to converge to local minimum
        if self.terminationFlag == 0:
            ws = int(2 + opt.stopCr.localConverge.windowSizeCoeff * restart.tolHistSize)
            chk1 = self.iterNo > (2 + max(opt.stopCr.localConverge.windowSizeCoeff * restart.tolHistSize, ws))
            chk2 = archive.size > 0
            chk3 = False
            if chk1 and len(self.maxCriticalityHist) >= ws:
                chk3 = max(self.maxCriticalityHist[-ws:]) < opt.stopCr.localConverge.tabooCriticUpLimit
            chk4 = self.bestVal < opt.coreSearch.objValUpLimit
            chk5 = self.localConvergeCheck.stopAtUsedEval == np.inf
            
            if chk1 and chk2 and chk3 and chk4 and chk5:
                diffs = np.diff(self.bestValNonEliteHist[-ws:])
                meanDiff = np.mean(np.abs(diffs))
                willBeLocal = meanDiff < (opt.stopCr.localConverge.tolCoeff * 
                                          (self.bestVal - np.max(archive.value) - opt.archiving.tolFunArch))
                if willBeLocal:
                    self.localConvergeCheck.stopAtUsedEval = self.usedEvalMerge + self.usedEvalEvolve
                    self.terminationFlag = 4
    
    def sample_solutions(self, restart, archive, process, opt, problem):
        """Sample new population members (to be implemented in subclass)."""
        raise NotImplementedError("Must be implemented in subclass")
    
    def select(self, restart, archive, process, opt, problem):
        """Perform selection (to be implemented in subclass)."""
        raise NotImplementedError("Must be implemented in subclass")
    
    def recombine(self, restart, archive, process, opt, problem):
        """Perform recombination (to be implemented in subclass)."""
        raise NotImplementedError("Must be implemented in subclass")

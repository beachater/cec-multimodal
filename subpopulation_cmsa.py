"""
SubpopulationCMSA - CMSA variant of Subpopulation with elite solutions.

Written by Ali Ahrari (aliahrari1983@gmail.com)
Translated to Python from MATLAB
"""

import numpy as np
from subpopulation import Subpopulation
from mutation_sampling import Sampling, Elite


class SubpopulationCMSA(Subpopulation):
    """Subpopulation using CMSA (Covariance Matrix Self-Adaptation) evolution strategy."""
    
    def __init__(self, center, smean, stretch, popSize):
        """Initialize CMSA subpopulation."""
        super().__init__(center, smean, stretch, popSize)
        self.elite = Elite(len(stretch))
    
    def sample_solutions(self, restart, archive, process, opt, problem):
        """Sample new population members and evaluate them."""
        tempRedRatio = 1.0  # For temporary reduction of taboo region if too many samples are rejected
        solNo = 0  # Solution counter
        
        # Preallocate
        self.samples = Sampling(process.subpopSize, problem.dim)
        
        # Sampling stage: generate taboo-acceptable solutions
        while solNo < process.subpopSize:
            # Calculate individual step sizes
            self.samples.s[solNo] = self.mutProfile.smean * np.exp(np.random.randn() * process.coreSpecStrPar.tauSigma)
            # Clamp tiny/invalid step sizes
            if not np.isfinite(self.samples.s[solNo]) or self.samples.s[solNo] < 1e-12:
                self.samples.s[solNo] = 1e-12
            
            # Generate vector Z
            self.samples.Z[solNo, :] = (self.mutProfile.rotMat @ 
                                        (self.mutProfile.stretch * np.random.randn(problem.dim)))
            
            # Sampled solution
            self.samples.X[solNo, :] = self.center + self.samples.s[solNo] * self.samples.Z[solNo, :]
            
            # Relocate to search range if outside
            self.repair_infeas(solNo, opt, problem)
            
            # Check if this sampled solution is taboo acceptable
            acceptIt = self.is_taboo_acceptable(self.samples.X[solNo, :], tempRedRatio, opt)
            
            if acceptIt:
                solNo += 1
            else:
                tempRedRatio *= opt.niching.redCoeff  # Temporary reduction of taboo region sizes
    
    def recombine(self, restart, archive, process, opt, problem):
        """Perform recombination when the core search is elitist CMSA."""
        oldCenter = self.center.copy()
        ind = self.samples.argsortWithElite
        
        # Update the center
        self.center = process.recWeights @ self.samples.X[ind[:process.mu], :]
        
        # Relocate center to the closest point in the search range
        self.center = np.maximum(self.center, problem.lowBound)
        self.center = np.minimum(self.center, problem.upBound)
        
        # Update the best solution of the subpopulation
        self.bestSol = self.samples.X[ind[0], :]
        self.bestVal = self.samples.f[ind[0]]
        
        # Update the global step size
        geomean_s = np.exp(np.mean(np.log(self.samples.s)))
        correctionTerm = (geomean_s / self.mutProfile.smean) ** opt.coreSearch.sigmaUpdateBiasImp
        
        weights_log_s = process.recWeights @ np.log(self.samples.s[ind[:process.mu]])
        self.mutProfile.smean = np.exp(weights_log_s) / correctionTerm
        # Safety clamp for global step size
        if not np.isfinite(self.mutProfile.smean) or self.mutProfile.smean < 1e-12:
            self.mutProfile.smean = 1e-12
        
        # Update the covariance matrix
        suggC = np.zeros((problem.dim, problem.dim))
        for parNo in range(process.mu):
            Z = (self.samples.X[ind[parNo], :] - oldCenter) / self.samples.s[ind[parNo]]
            suggC += process.recWeights[parNo] * np.outer(Z, Z)
        
        cc = 1 / process.coreSpecStrPar.tauCov
        newC = (1 - cc) * self.mutProfile.C + cc * suggC
        # Enforce symmetry explicitly and sanitize any numerical issues
        newC = 0.5 * (newC + newC.T)
        # Replace inf/NaN with finite values (fallback to previous covariance)
        if not np.isfinite(newC).all():
            newC = np.where(np.isfinite(newC), newC, self.mutProfile.C)
        # Add a tiny jitter to the diagonal to avoid singularities
        jitter = 1e-12 * np.eye(problem.dim)
        self.mutProfile.C = newC + jitter

        # Perform stable eigen decomposition (matrix is symmetric)
        try:
            eigvals, eigvecs = np.linalg.eigh(self.mutProfile.C)
        except np.linalg.LinAlgError:
            # Fallback: use identity
            eigvals = np.ones(problem.dim)
            eigvecs = np.eye(problem.dim)
            self.mutProfile.C = np.eye(problem.dim)
        # Clamp eigenvalues to a minimum positive threshold
        eigvals = np.maximum(eigvals, 1e-20)
        self.mutProfile.rotMat = eigvecs
        self.mutProfile.stretch = np.sqrt(eigvals)
        
        # Calculate inverse of covariance matrix
        self.mutProfile.Cinv = self.mutProfile.rotMat @ np.diag(self.mutProfile.stretch ** (-2)) @ self.mutProfile.rotMat.T
        
        # Update elite solutions
        if process.coreSpecStrPar.numElt > 0:
            surviveInd = np.argsort(self.samples.f)
            # Count feasible solutions (ignoring NaN values)
            num_feasible = np.sum(self.samples.isFeas[~np.isnan(self.samples.isFeas)])
            limit1 = num_feasible + len(self.elite.val)
            limit2 = process.coreSpecStrPar.numElt
            actEltNum = int(0.5 + min(limit1, limit2))
            
            self.elite.sol = self.samples.X[surviveInd[:actEltNum], :]
            self.elite.val = self.samples.f[surviveInd[:actEltNum]]
            self.elite.Z = self.samples.Z[surviveInd[:actEltNum], :]
            self.elite.s = self.samples.s[surviveInd[:actEltNum]]
            self.elite.wasRepaired = self.samples.wasRepaired[surviveInd[:actEltNum]]
    
    def select(self, restart, archive, process, opt, problem):
        """Perform selection."""
        # Indexes of best non-elite samples
        self.samples.argsortNoElite = np.argsort(self.samples.f)
        
        # Append acceptable elite solutions to the selection pool
        if opt.coreSearch.algorithm == 'CMSA' and process.coreSpecStrPar.numElt > 0 and len(self.elite.val) > 0:
            appendIt = np.ones(len(self.elite.val), dtype=bool)
            
            # Only append elite solutions that are taboo acceptable
            for eltNo in range(len(self.elite.val)):
                appendIt[eltNo] = self.is_taboo_acceptable(self.elite.sol[eltNo, :], 1, opt)
            
            appendIdx = np.where(appendIt)[0]
            
            # Append the approved elite solutions to the selection pool
            self.samples.X = np.vstack([self.samples.X, self.elite.sol[appendIdx, :]])
            self.samples.Z = np.vstack([self.samples.Z, self.elite.Z[appendIdx, :]])
            self.samples.s = np.concatenate([self.samples.s, self.elite.s[appendIdx]])
            self.samples.f = np.concatenate([self.samples.f, self.elite.val[appendIdx]])
            self.samples.wasRepaired = np.concatenate([self.samples.wasRepaired, self.elite.wasRepaired[appendIdx]])
        
        # For performing elite selection
        self.samples.argsortWithElite = np.argsort(self.samples.f)

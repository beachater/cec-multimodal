"""Prediction methods for dynamic optimization (AMLP)."""
import numpy as np
from typing import Tuple, List


class Prediction:
    """Adaptive multilevel prediction methods."""
    
    @staticmethod
    def AMLP(xopt: np.ndarray, maxLevel: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """Adaptive Multi-Level Prediction.
        
        Parameters
        ----------
        xopt : (T, D) array
            Time history of a particular global optimum at different time steps.
        maxLevel : int
            Maximum level considered for prediction.
            
        Returns
        -------
        xhatNow : (maxLevelPrime, D) array
            Predicted solution from each prediction level for time step t+1.
        estPreErrNowNorm : (maxLevelPrime,) array
            Estimated prediction error norm for each level.
        bestLevel : int
            Most reliable prediction level for time step t+1.
        """
        T, D = xopt.shape
        
        # Calculate the prediction and error from all requested levels
        xhat, preErr, maxLevelPrime = Prediction.MLP(xopt, maxLevel)
        
        estPreErrNowNorm = np.full(maxLevelPrime, np.nan)
        xhatNow = np.full((maxLevelPrime, D), np.nan)
        
        for L in range(maxLevelPrime):
            estPreErrNowNorm[L] = np.linalg.norm(preErr[L][-1, :])
            xhatNow[L, :] = xhat[L][-1, :]
        
        # Return the outcome of the best prediction level
        if T == 1:
            bestLevel = 1  # 1-based index as in MATLAB
            estPreErrNowNorm = np.array([np.nan])
        else:
            bestLevel = int(np.argmin(estPreErrNowNorm)) + 1  # 1-based
            # If best is second-to-last and maxLevelPrime == T, use next level
            if (bestLevel == maxLevelPrime - 1) and (maxLevelPrime == T):
                bestLevel = bestLevel + 1
        
        return xhatNow, estPreErrNowNorm, bestLevel
    
    @staticmethod
    def MLP(xopt: np.ndarray, maxLevel: int) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
        """Multi-Level Prediction.
        
        Parameters
        ----------
        xopt : (T, D) array
            Time history of a particular global optimum.
        maxLevel : int
            Maximum prediction level.
            
        Returns
        -------
        xhat : List of (t+2, D) arrays
            Predicted solution by each prediction level for each time step.
        preErr : List of (t+1, D) arrays
            Prediction error for each level and each time step.
        maxLevelPrime : int
            Actual maximum level used.
        """
        t, D = xopt.shape
        t = t - 1  # Last finished time step (0-based)
        
        maxLevelPrime = min(maxLevel, t + 1)
        t_recent = max(0, t - maxLevel)
        
        xhat = [None] * maxLevelPrime
        preErr = [None] * maxLevelPrime
        
        # Level 1
        L = 0  # 0-based indexing in Python
        xhat[L] = np.full((t + 2, D), np.nan)
        preErr[L] = np.full((t + 1, D), np.nan)
        for tau in range(t_recent, t + 1):
            xhat[L][tau + 1, :] = xopt[tau, :]  # tau+1 in MATLAB becomes tau in 0-based
            if tau < t:
                preErr[L][tau + 1, :] = xopt[tau + 1, :] - xhat[L][tau + 1, :]
        
        # Higher levels
        for L in range(1, maxLevelPrime):
            xhat[L] = np.full((t + 2, D), np.nan)
            preErr[L] = np.full((t + 1, D), np.nan)
            for tau in range(t_recent, t + 1):
                if tau >= L:  # L is prediction level (1-based originally, 0-based here)
                    xhat[L][tau + 1, :] = xhat[L - 1][tau + 1, :] + preErr[L - 1][tau, :]
                    if tau < t:
                        preErr[L][tau + 1, :] = xopt[tau + 1, :] - xhat[L][tau + 1, :]
        
        # Handle exceptional case
        if (maxLevelPrime == t + 1) and (maxLevelPrime > 1):
            preErr[maxLevelPrime - 1][t, :] = preErr[maxLevelPrime - 2][t, :]
        
        return xhat, preErr, maxLevelPrime

"""Utility methods translated from MATLAB."""
import numpy as np
from scipy.interpolate import interp1d


class UtilityMethods:
    @staticmethod
    def lin_prctile(data: np.ndarray, q: float) -> float:
        """Linear percentile interpolation matching MATLAB logic."""
        n = len(data)
        if n == 1:
            return float(data[0])
        else:
            data = np.sort(data)
            x = np.linspace(0, 100, n)
            p = interp1d(x, data, kind='linear', fill_value='extrapolate')(q)
            return float(p)

    @staticmethod
    def peak2peak(x: np.ndarray) -> float:
        """Return peak-to-peak (max - min)."""
        return float(np.max(x) - np.min(x))

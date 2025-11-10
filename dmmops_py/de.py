import numpy as np
from .problem import DMMOP
from .boundary_check import boundary_check
import csv


def de(fn: int = 16, run: int = 1, csv_path: str | None = None):
    rng = np.random.default_rng(run)
    pro = DMMOP(fn)
    N = 100
    F = 0.5
    CR = 0.5
    D = pro.D
    pop = rng.random((N, D)) * (pro.upper - pro.lower) + pro.lower
    fits = pro.GetFits(pop)

    while not pro.Terminate():
        # mutation DE/rand/1
        index = np.zeros((N, 3), dtype=int)
        for i in range(N):
            perm = rng.choice(N - 1, size=3, replace=False)
            perm = np.where(perm >= i, perm + 1, perm)
            index[i, :] = perm
        mutant = pop[index[:, 0], :] - F * (pop[index[:, 2], :] - pop[index[:, 1], :])
        # crossover
        cross = rng.random((N, D)) < CR
        parent_only = np.where(cross.sum(axis=1) == 0)[0]
        for i in parent_only:
            cross[i, rng.integers(D)] = True
        off = np.where(cross, mutant, pop)
        off = boundary_check(off, pro.lower, pro.upper)
        off_fits = pro.GetFits(off)
        num = off_fits.shape[0]
        if num > 0:
            cmp = np.where(off_fits > fits[:num])[0]
            pop[cmp, :] = off[cmp, :]
            fits[cmp] = off_fits[cmp]
        if pro.CheckChange(pop, fits):
            fits = pro.GetFits(pop)
    peak, allpeak = pro.GetPeak()
    speak = peak.sum(axis=1)  # 3 accuracy levels
    sallpeak = allpeak.sum()
    if csv_path:
        # Write a CSV with per-accuracy results and grand totals
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['accuracy_tol', 'total_found_over_envs', 'total_peaks_over_envs'])
            acc = [1e-3, 1e-4, 1e-5]
            for i, tol in enumerate(acc):
                w.writerow([tol, int(speak[i]), int(sallpeak)])
    return speak, sallpeak

"""
TSD v4 (Temporal Substrate Drift) driver for DMMOP benchmark.

v4 adds function-aware parameter presets to improve performance on:
- Composition functions (F5-F8 base): tighter niche radius, generous archive threshold
- D=10 problems: larger population, more refinement budget, end-of-env polishing

Core TSD mechanisms preserved (substrate drift, HSAT, clearing niching, Rac1 forgetting).
Based on tsd_driver.py (v1) which already beats AMLP on F1-F4.
"""

import numpy as np
import time
import pickle
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
from tsd import ETFCSA_TSD, Antibody


# ============================================================================
# Function-Aware Preset System
# ============================================================================

@dataclass
class FunctionPreset:
    """Function-aware parameter preset for TSD v4."""
    # Problem classification
    base_func: int          # 1-8
    is_composition: bool    # True for F5-F8 base
    dim: int                # 5 or 10
    n_peaks: int            # Expected number of peaks
    optimal_value: float    # 75.0 or 0.0

    # Population
    N: int
    budget_per_tick: int

    # Clearing/niching
    clearing_niche_radius: float
    clearing_quality_threshold: float
    clearing_penalty_strength: float

    # Archive
    archive_dist_threshold: float  # For archiving (separate from CEC's 0.05 for PR)

    # Refinement
    refine_max_evals: int          # Per-peak refinement budget (~20*D)
    refine_interval: int           # Generations between archive refinement

    # Exploration
    exploration_interval: int
    exploration_rate: float

    # End-of-env polishing
    end_polish_budget_per_peak: int


def get_function_preset(pid: int, dim: int) -> FunctionPreset:
    """
    Create function-aware parameter preset based on PID and dimension.

    PID mapping (CEC 2022 DMMOP):
    - P1-P4  : F1-F4 (simple, 4 peaks, optimal=75)
    - P5-P8  : F5-F8 (composition, 6-8 peaks, optimal=0)
    - P9-P16 : F8 variants (composition, 8 peaks, different change types)
    - P17-P20: F1-F4 in D=10
    - P21-P24: F5-F8 in D=10
    """
    # Determine base function from PID
    if 1 <= pid <= 8:
        base_func = pid
    elif 9 <= pid <= 16:
        base_func = 8  # P9-P16 all use F8
    elif 17 <= pid <= 24:
        base_func = pid - 16  # P17->F1, ..., P24->F8
    else:
        base_func = 1

    is_composition = base_func >= 5

    # Number of peaks by base function
    peak_counts = {1: 4, 2: 4, 3: 4, 4: 4, 5: 6, 6: 8, 7: 6, 8: 8}
    n_peaks = peak_counts.get(base_func, 4)

    optimal_value = 0.0 if is_composition else 75.0

    if not is_composition:
        # === SIMPLE FUNCTIONS (F1-F4 base) ===
        # v1 already works well here. Minor tuning only.
        if dim <= 5:
            return FunctionPreset(
                base_func=base_func, is_composition=False,
                dim=dim, n_peaks=n_peaks, optimal_value=optimal_value,
                N=60, budget_per_tick=200,
                clearing_niche_radius=1.0,
                clearing_quality_threshold=65.0,   # optimal - 10
                clearing_penalty_strength=200.0,
                archive_dist_threshold=0.10,        # Was 0.05, wider for archiving
                refine_max_evals=100,               # 20 * dim
                refine_interval=80,
                exploration_interval=20,
                exploration_rate=0.30,
                end_polish_budget_per_peak=100,
            )
        else:  # D=10
            return FunctionPreset(
                base_func=base_func, is_composition=False,
                dim=dim, n_peaks=n_peaks, optimal_value=optimal_value,
                N=100, budget_per_tick=250,
                clearing_niche_radius=1.0,
                clearing_quality_threshold=65.0,
                clearing_penalty_strength=200.0,
                archive_dist_threshold=0.10,
                refine_max_evals=200,               # 20 * dim
                refine_interval=60,                  # More frequent for D10
                exploration_interval=15,
                exploration_rate=0.25,
                end_polish_budget_per_peak=200,
            )
    else:
        # === COMPOSITION FUNCTIONS (F5-F8 base) ===
        # Middle-ground preset: compromise between v1's strong clearing and
        # round1's softer exploration-focused approach.
        # N=70 (between v1=60 and round1=80), penalty=150, niche_r=0.8
        # Keeps v4 exploration improvements (interval=10, rate=0.35).
        if dim <= 5:
            return FunctionPreset(
                base_func=base_func, is_composition=True,
                dim=dim, n_peaks=n_peaks, optimal_value=optimal_value,
                N=70, budget_per_tick=200,
                clearing_niche_radius=0.8,            # Middle ground (v1=1.0, round1=0.5)
                clearing_quality_threshold=-15.0,      # Middle ground (v1=-10, round1=-20)
                clearing_penalty_strength=150.0,       # Middle ground (v1=200, round1=100)
                archive_dist_threshold=0.08,           # Slightly wider than v1=0.05, under dpeaks=0.1
                refine_max_evals=100,                  # 20 * dim
                refine_interval=60,
                exploration_interval=10,               # v4 improvement: faster exploration
                exploration_rate=0.35,                 # v4 improvement: more explorers for more peaks
                end_polish_budget_per_peak=100,
            )
        else:  # D=10
            return FunctionPreset(
                base_func=base_func, is_composition=True,
                dim=dim, n_peaks=n_peaks, optimal_value=optimal_value,
                N=100, budget_per_tick=200,
                clearing_niche_radius=0.8,
                clearing_quality_threshold=-15.0,
                clearing_penalty_strength=150.0,
                archive_dist_threshold=0.08,
                refine_max_evals=200,                  # 20 * dim
                refine_interval=50,
                exploration_interval=10,
                exploration_rate=0.35,
                end_polish_budget_per_peak=200,
            )


# ============================================================================
# Archive with Configurable Distance Threshold
# ============================================================================

@dataclass
class TSDArchive:
    """Archive for tracking found optima in dynamic environments."""
    solution: np.ndarray = None
    value: np.ndarray = None
    foundEval: np.ndarray = None
    foundEval2: np.ndarray = None
    size: int = 0
    widths: np.ndarray = None
    dist_threshold: float = 0.05  # Configurable (was hardcoded 0.05 in v1)

    def __init__(self, dim: int, widths: np.ndarray = None, dist_threshold: float = 0.05):
        self.solution = np.empty((0, dim), dtype=float)
        self.value = np.empty((0,), dtype=float)
        self.foundEval = np.empty((0,), dtype=int)
        self.foundEval2 = np.empty((0,), dtype=int)
        self.size = 0
        self.widths = widths if widths is not None else np.ones(dim) * 10.0
        self.dist_threshold = dist_threshold

    def add(self, x: np.ndarray, f: float, evals: int) -> bool:
        """Add a solution to archive if it's unique. Returns True if added."""
        MAX_ARCHIVE_SIZE = 50

        # Check if already in archive (using configurable distance threshold)
        if self.size > 0:
            dists = np.linalg.norm(self.solution - x[None, :], axis=1)
            min_idx = np.argmin(dists)
            if dists[min_idx] < self.dist_threshold:
                # If new solution is better, replace existing
                if f > self.value[min_idx]:
                    self.solution[min_idx] = x
                    self.value[min_idx] = f
                return False

        # If archive is full, only add if better than worst
        if self.size >= MAX_ARCHIVE_SIZE:
            worst_idx = np.argmin(self.value)
            if f > self.value[worst_idx]:
                self.solution[worst_idx] = x
                self.value[worst_idx] = f
                self.foundEval[worst_idx] = evals
                self.foundEval2[worst_idx] = evals
            return False

        # Add to archive
        self.solution = np.vstack([self.solution, x[None, :]])
        self.value = np.append(self.value, f)
        self.foundEval = np.append(self.foundEval, evals)
        self.foundEval2 = np.append(self.foundEval2, evals)
        self.size += 1
        return True

    def reset(self):
        """Clear archive for new environment."""
        dim = self.solution.shape[1] if self.size > 0 else 0
        self.solution = np.empty((0, dim), dtype=float)
        self.value = np.empty((0,), dtype=float)
        self.foundEval = np.empty((0,), dtype=int)
        self.foundEval2 = np.empty((0,), dtype=int)
        self.size = 0


# ============================================================================
# TSD v4 Wrapper for DMMOP
# ============================================================================

class TSDForDMMOP_v4:
    """
    TSD v4 wrapper for DMMOP dynamic multimodal optimization.

    Improvements over v1:
    - Function-aware parameter presets (simple vs composition, D5 vs D10)
    - Configurable archive distance threshold (wider for archiving)
    - Dimension-scaled local refinement budget
    - End-of-environment polishing for precision
    - Preset-driven exploration/refinement intervals

    Core TSD mechanisms preserved:
    - Substrate drift (s vector, eta learning, rho decay)
    - HSAT (Hybrid Substrate-Aligned Triggering)
    - Clearing-based niching with additive quadratic penalty
    - Rac1 forgetting
    - Partial reset response to environment changes
    """

    def __init__(
        self,
        problem,
        seed: int = 1,
        preset: FunctionPreset = None,
        # TSD-specific params
        eta: float = 0.25,
        lambda_s: float = 0.5,
        rho: float = 0.98,
        drift_interval: int = 2000,
        # Response params
        response_mode: str = 'partial_reset',
    ):
        self.problem = problem
        self.seed = seed

        # Get preset or auto-detect from PID
        if preset is None:
            pid = getattr(problem, 'PID', 1)
            preset = get_function_preset(pid, problem.dim)
        self.preset = preset

        # Apply preset values
        self.N = preset.N
        self.budget_per_tick = preset.budget_per_tick
        self.response_mode = response_mode
        self.exploration_interval = preset.exploration_interval
        self.exploration_rate = preset.exploration_rate

        # TSD parameters
        self.eta = eta
        self.lambda_s = lambda_s
        self.rho = rho
        self.drift_interval = drift_interval

        # Compute search space widths
        self.widths = problem.upBound - problem.lowBound

        # Archive with function-aware distance threshold
        self.archive = TSDArchive(
            problem.dim, self.widths,
            dist_threshold=preset.archive_dist_threshold
        )
        self.archive_history = []

        # Function-aware clearing parameters from preset
        self.optimal_value = preset.optimal_value
        self.clearing_niche_radius = preset.clearing_niche_radius
        self.clearing_quality_threshold = preset.clearing_quality_threshold
        self.clearing_penalty_strength = preset.clearing_penalty_strength

        # Statistics
        self.current_env = 0
        self.env_start_eval = 0
        self.best_per_env = []
        self.env_best_f = float('-inf')
        self.generation = 0

        # Print preset info
        func_type = "composition" if preset.is_composition else "simple"
        print(f"[v4] PID={getattr(problem, 'PID', '?')}, base=F{preset.base_func}, "
              f"type={func_type}, D={preset.dim}, peaks={preset.n_peaks}")
        print(f"[v4] N={preset.N}, niche_r={preset.clearing_niche_radius}, "
              f"quality_thresh={preset.clearing_quality_threshold}, "
              f"archive_dist={preset.archive_dist_threshold}")

        # Convert problem bounds
        bounds = [(problem.lowBound[i], problem.upBound[i]) for i in range(problem.dim)]

        # Create TSD optimizer
        self.tsd = ETFCSA_TSD(
            func=self._objective_wrapper,
            bounds=bounds,
            N=self.N,
            seed=seed,
            max_evals=problem.maxEval,
            budget_per_tick=self.budget_per_tick,
            eta=eta,
            lambda_s=lambda_s,
            rho=rho,
            drift_interval=drift_interval,
            progress=self._on_generation,
        )

    def _refine_peak(self, x: np.ndarray, f: float, max_evals: int = None) -> Tuple[np.ndarray, float]:
        """
        Budget-conscious local refinement via coordinate descent.

        Budget scales with dimension: ~20*D evals (was fixed 80 in v1).
        Step sizes adapted for composition vs simple function landscapes.

        Args:
            x: Starting solution
            f: Starting fitness (DMMOP value, higher = better)
            max_evals: Maximum evaluations (default: preset.refine_max_evals)

        Returns:
            (refined_x, refined_f)
        """
        if max_evals is None:
            max_evals = self.preset.refine_max_evals

        x_best = x.copy()
        f_best = f
        evals_used = 0
        golden = 0.618

        # Adaptive initial step based on function type and gap to optimal
        gap = self.optimal_value - f_best
        if self.preset.is_composition:
            # Composition functions: different fitness scale
            if gap > 5.0:
                step = np.ones(self.problem.dim) * 0.1
            elif gap > 0.1:
                step = np.ones(self.problem.dim) * 0.01
            else:
                step = np.ones(self.problem.dim) * 0.001
        else:
            # Simple functions: existing v1 logic
            if gap > 1.0:
                step = np.ones(self.problem.dim) * 0.05
            elif gap > 0.01:
                step = np.ones(self.problem.dim) * 0.005
            else:
                step = np.ones(self.problem.dim) * 0.0005

        while evals_used < max_evals and np.max(step) > 1e-8:
            if f_best >= self.optimal_value - 1e-5:
                break

            improved_this_round = False

            for d in range(self.problem.dim):
                if evals_used >= max_evals:
                    break

                for direction in [+1.0, -1.0]:
                    if evals_used >= max_evals:
                        break

                    x_trial = x_best.copy()
                    x_trial[d] = np.clip(
                        x_trial[d] + direction * step[d],
                        self.problem.lowBound[d],
                        self.problem.upBound[d]
                    )

                    f_array = self.problem.extProb.GetFits(x_trial.reshape(1, -1))
                    evals_used += 1

                    if f_array.size == 0:
                        return x_best, f_best

                    f_trial = float(f_array[0])

                    if f_trial > f_best:
                        x_best = x_trial
                        f_best = f_trial
                        improved_this_round = True
                        break
                else:
                    step[d] *= golden

            if not improved_this_round:
                step *= golden

        return x_best, f_best

    def _refine_archive_peaks(self):
        """
        Periodically refine the best solution in each archive cluster.
        Budget: ~n_peaks x refine_max_evals per call.
        """
        if self.archive.size == 0:
            return

        solutions = self.archive.solution.copy()
        values = self.archive.value.copy()
        n = self.archive.size

        # Greedy clustering by niche radius (process best-first)
        visited = np.zeros(n, dtype=bool)
        clusters = []
        sorted_idx = np.argsort(-values)

        for idx in sorted_idx:
            if visited[idx]:
                continue
            visited[idx] = True
            cluster_center = solutions[idx]

            for j in sorted_idx:
                if not visited[j]:
                    if np.linalg.norm(solutions[j] - cluster_center) < self.clearing_niche_radius:
                        visited[j] = True

            clusters.append(idx)

        # Refine best solution per cluster (with preset budget)
        for arch_idx in clusters:
            x_start = self.archive.solution[arch_idx].copy()
            f_start = self.archive.value[arch_idx]

            x_refined, f_refined = self._refine_peak(x_start, f_start)

            if f_refined > f_start:
                self.archive.solution[arch_idx] = x_refined
                self.archive.value[arch_idx] = f_refined
                if f_refined > self.env_best_f:
                    self.env_best_f = f_refined

            self.archive.add(x_refined, f_refined, self.problem.extProb.evaluated)

    def _end_of_env_polish(self):
        """
        Final refinement pass on top archive peaks before environment ends.
        Addresses D10 precision collapse by dedicating budget to peak precision.
        Budget: ~n_peaks x end_polish_budget_per_peak (2-3% of per-env budget).
        """
        if self.archive.size == 0:
            return

        solutions = self.archive.solution.copy()
        values = self.archive.value.copy()
        n = self.archive.size

        # Cluster archive to find distinct peaks
        visited = np.zeros(n, dtype=bool)
        peak_indices = []
        sorted_idx = np.argsort(-values)

        for idx in sorted_idx:
            if visited[idx]:
                continue
            visited[idx] = True
            for j in sorted_idx:
                if not visited[j]:
                    if np.linalg.norm(solutions[j] - solutions[idx]) < self.clearing_niche_radius:
                        visited[j] = True
            peak_indices.append(idx)

        # Refine each peak with dedicated end-of-env budget
        n_to_polish = min(len(peak_indices), self.preset.n_peaks)
        for arch_idx in peak_indices[:n_to_polish]:
            x_start = self.archive.solution[arch_idx].copy()
            f_start = self.archive.value[arch_idx]

            x_refined, f_refined = self._refine_peak(
                x_start, f_start,
                max_evals=self.preset.end_polish_budget_per_peak
            )

            if f_refined > f_start:
                self.archive.solution[arch_idx] = x_refined
                self.archive.value[arch_idx] = f_refined
                if f_refined > self.env_best_f:
                    self.env_best_f = f_refined

            self.archive.add(x_refined, f_refined, self.problem.extProb.evaluated)

    def _on_generation(self, gen: int, pop: np.ndarray, fitness: np.ndarray,
                       best_fitness: float, gbest: np.ndarray, evals: int):
        """Callback after each TSD generation for exploration and refinement."""
        self.generation = gen

        # Periodic exploration with preset interval (was hardcoded 20 in v1)
        if gen > 0 and gen % self.preset.exploration_interval == 0 and self.archive.size > 0:
            self._apply_exploration_reset()

        # Periodic archive refinement with preset interval (was hardcoded 80 in v1)
        if gen > 0 and gen % self.preset.refine_interval == 0:
            self._refine_archive_peaks()

    def _apply_exploration_reset(self):
        """Teleport worst individuals to regions far from archived peaks."""
        if not hasattr(self.tsd, 'pop') or len(self.tsd.pop) == 0:
            return

        pop = self.tsd.pop
        n_reset = max(3, int(len(pop) * self.exploration_rate))

        # Select worst individuals
        fitnesses = np.array([-ab.aff for ab in pop])
        reset_indices = np.argsort(fitnesses)[:n_reset]

        # Get high-quality archived solution positions as "known peaks"
        peak_positions = []
        if self.archive.size > 0:
            high_quality_mask = self.archive.value >= self.clearing_quality_threshold
            if np.any(high_quality_mask):
                peak_positions = list(self.archive.solution[high_quality_mask])

        for idx in reset_indices:
            best_pos = None
            best_min_dist = 0

            for attempt in range(30):
                new_pos = self.problem.lowBound + np.random.random(self.problem.dim) * self.widths

                if peak_positions:
                    dists = [np.linalg.norm(new_pos - p) for p in peak_positions]
                    min_dist = min(dists)

                    if min_dist > best_min_dist:
                        best_min_dist = min_dist
                        best_pos = new_pos
                else:
                    best_pos = new_pos
                    break

            if best_pos is not None:
                pop[idx].x = best_pos
                pop[idx].T = 0
                pop[idx].S = 0
                pop[idx].I = 0.0
                pop[idx].A_orth = 1.0  # High exploration signal
                pop[idx].C = 0.0       # Reset coherence

    def _objective_wrapper(self, x: np.ndarray) -> float:
        """
        Wrapper that calls problem.GetFits and handles environment changes.

        IMPORTANT: DMMOP uses MAXIMIZATION (higher f = better peak).
        TSD uses MINIMIZATION (lower f = better).
        We return -f to TSD so minimizing -f maximizes f.
        Archive stores ORIGINAL f values for correct PR calculation.
        """
        # Evaluate
        f_array = self.problem.extProb.GetFits(x.reshape(1, -1))

        if f_array.size == 0:
            return 1e6

        f = float(f_array[0])

        # Track environment best
        if f > self.env_best_f:
            self.env_best_f = f

        # Archive any solution above quality threshold
        if f >= self.clearing_quality_threshold:
            self.archive.add(x.copy(), f, self.problem.extProb.evaluated)

        # Check for environment change
        if self.archive.size > 0:
            changed = self.problem.extProb.CheckChange(
                self.archive.solution, self.archive.value)
        else:
            changed = self.problem.extProb.CheckChange(
                x.reshape(1, -1), f_array)

        if changed:
            self._handle_environment_change()

        # CLEARING-BASED NICHING: additive penalty near high-quality archived solutions
        f_adjusted = f

        if self.archive.size > 0:
            high_quality_mask = self.archive.value >= self.clearing_quality_threshold
            if np.any(high_quality_mask):
                hq_solutions = self.archive.solution[high_quality_mask]
                hq_values = self.archive.value[high_quality_mask]

                dists = np.linalg.norm(hq_solutions - x[None, :], axis=1)
                min_idx = np.argmin(dists)
                min_dist = dists[min_idx]

                if min_dist < self.clearing_niche_radius:
                    nearest_f = hq_values[min_idx]
                    if f <= nearest_f:
                        penalty = (1.0 - (min_dist / self.clearing_niche_radius) ** 2) * self.clearing_penalty_strength
                        f_adjusted = f - penalty

        return -f_adjusted

    def _handle_environment_change(self):
        """Respond to environment change detected by DMMOP."""
        # End-of-environment polishing (NEW in v4)
        self._end_of_env_polish()

        # Log archive stats
        if self.archive.size > 0:
            n_hq = int(np.sum(self.archive.value >= self.clearing_quality_threshold))
            print(f"\n[Env {self.current_env}] Archive: {self.archive.size} solutions, "
                  f"{n_hq} high-quality, best={np.max(self.archive.value):.4f}")

        # Save current archive
        self.archive_history.append({
            'env': self.current_env,
            'archive': self.archive,
            'evals': self.problem.numCallF,
        })

        self.current_env += 1
        self.env_start_eval = self.problem.numCallF

        # Response strategy (same as v1)
        if self.response_mode == 'full_reset':
            self.tsd._init()

        elif self.response_mode == 'partial_reset':
            self.tsd.s *= 0.5  # Decay substrate

            n_keep = max(2, int(self.N * 0.4))
            affs = np.array([ab.aff for ab in self.tsd.pop])
            keep_idx = np.argsort(affs)[-n_keep:]

            new_pop = [self.tsd.pop[i] for i in keep_idx]

            for _ in range(self.N - n_keep):
                x = self.problem.lowBound + np.random.random(self.problem.dim) * self.widths

                f_array = self.problem.extProb.GetFits(x.reshape(1, -1))
                if f_array.size == 0:
                    f_val = 0.0
                else:
                    f_val = float(f_array[0])
                    if f_val >= self.clearing_quality_threshold:
                        self.archive.add(x.copy(), f_val, self.problem.extProb.evaluated)
                    chk = self.problem.extProb.CheckChange(x.reshape(1, -1), f_array)
                    if chk:
                        self.tsd.pop = new_pop
                        self._handle_environment_change()
                        return

                ab = Antibody(x=x, aff=f_val)
                ab.x_prev = x.copy()
                ab.A_sub = 0.0
                ab.A_orth = 1.0
                ab.C = 0.0
                ab.displacement_norm = 0.0
                new_pop.append(ab)

            self.tsd.pop = new_pop

        elif self.response_mode == 'adapt':
            self.tsd.s *= 0.7
            for ab in self.tsd.pop:
                ab.T = max(0, ab.T - 50)

        # Reset environment best and create new archive with preset dist_threshold
        self.env_best_f = float('-inf')
        self.archive = TSDArchive(
            self.problem.dim, self.widths,
            dist_threshold=self.preset.archive_dist_threshold
        )

    def optimize(self):
        """Run TSD v4 optimization on DMMOP problem."""
        func_type = "composition" if self.preset.is_composition else "simple"
        print(f"Starting TSD v4 optimization ({func_type}, D={self.preset.dim})")
        print(f"Max evaluations: {self.problem.maxEval}")
        print(f"Response mode: {self.response_mode}")
        print(f"Population size: {self.N}")
        print("-" * 60)

        start_time = time.time()

        # Run TSD
        x_best, f_best, info = self.tsd.optimize()

        # Log final archive stats
        if self.archive.size > 0:
            n_hq = int(np.sum(self.archive.value >= self.clearing_quality_threshold))
            print(f"\n[Env {self.current_env} - FINAL] Archive: {self.archive.size} solutions, "
                  f"{n_hq} high-quality, best={np.max(self.archive.value):.4f}")

        # Save final archive
        self.archive_history.append({
            'env': self.current_env,
            'archive': self.archive,
            'evals': self.problem.numCallF,
        })

        elapsed = time.time() - start_time

        print("-" * 60)
        print(f"Optimization completed!")
        print(f"Evaluations used: {info['evals_used']}")
        print(f"Generations: {info['generations_run']}")
        print(f"Environments encountered: {len(self.archive_history)}")
        print(f"Best fitness: {f_best:.6e}")
        print(f"Time elapsed: {elapsed:.2f}s")
        print(f"Substrate norm: {info['substrate_norm']:.6e}")

        return {
            'best_x': x_best,
            'best_f': f_best,
            'archive_history': self.archive_history,
            'final_archive': self.archive,
            'info': info,
            'elapsed_time': elapsed,
        }


# ============================================================================
# Driver Function
# ============================================================================

def tsd_driver_v4(problem, seedNo, filename, result_dir='result', **tsd_kwargs):
    """
    Driver function for TSD v4 (function-aware) on DMMOP problems.

    Args:
        problem: DMMOP problem instance
        seedNo: Random seed
        filename: Output filename
        result_dir: Output directory
        **tsd_kwargs: Override TSD parameters or preset fields
    """
    np.random.seed(seedNo)

    # Get function-aware preset
    pid = getattr(problem, 'PID', 1)
    preset = get_function_preset(pid, problem.dim)

    # Allow overrides via kwargs for preset fields
    preset_fields = [
        'N', 'budget_per_tick', 'clearing_niche_radius',
        'clearing_quality_threshold', 'clearing_penalty_strength',
        'archive_dist_threshold', 'refine_max_evals', 'refine_interval',
        'exploration_interval', 'exploration_rate', 'end_polish_budget_per_peak',
    ]
    for field in preset_fields:
        if field in tsd_kwargs:
            setattr(preset, field, tsd_kwargs.pop(field))

    # TSD-specific params
    default_params = {
        'eta': 0.25,
        'lambda_s': 0.5,
        'rho': 0.98,
        'drift_interval': 2000,
        'response_mode': 'partial_reset',
    }
    default_params.update(tsd_kwargs)

    # Create and run optimizer
    tsd = TSDForDMMOP_v4(problem, seed=seedNo, preset=preset, **default_params)
    results = tsd.optimize()

    # Reconstruct combined archive (compatible with postprocess_tsd.py)
    combined_archive = TSDArchive(problem.dim)
    for env_data in tsd.archive_history:
        arch = env_data['archive']
        for i in range(arch.size):
            combined_archive.add(
                arch.solution[i, :],
                arch.value[i],
                arch.foundEval[i]
            )

    # Save results
    save_data = {
        'problem': problem,
        'archive': combined_archive,
        'best_x': results['best_x'],
        'best_f': results['best_f'],
        'tsd_info': results['info'],
        'archive_history': results['archive_history'],
        'elapsed_time': results['elapsed_time'],
        'algorithm': 'TSD_v4',
        'seed': seedNo,
    }

    result_dir = Path(result_dir)
    result_dir.mkdir(exist_ok=True)

    output_path = result_dir / filename
    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)

    print(f"Results saved to: {output_path}")

"""
TSD (Temporal Substrate Drift) driver for DMMOP benchmark.

Adapted from the original TSD algorithm to work with dynamic multimodal problems.
The key adaptations:
1. Dynamic environment detection and response
2. Archive-based multimodal tracking
3. Per-environment statistics collection
4. Integration with DMMOP evaluation tracking

Original TSD by: (add original author info)
Adapted for DMMOP by: GitHub Copilot
"""

import numpy as np
import time
import pickle
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
from tsd import ETFCSA_TSD, Antibody


@dataclass
class TSDArchive:
    """Simple archive for tracking found optima in dynamic environments."""
    solution: np.ndarray = None  # Archive solutions (Narch × dim)
    value: np.ndarray = None     # Archive objective values (Narch,)
    foundEval: np.ndarray = None # Evaluations when found (Narch,)
    foundEval2: np.ndarray = None  # For compatibility with postprocess
    size: int = 0
    widths: np.ndarray = None    # Search space widths for scaled distance

    def __init__(self, dim: int, widths: np.ndarray = None):
        self.solution = np.empty((0, dim), dtype=float)
        self.value = np.empty((0,), dtype=float)
        self.foundEval = np.empty((0,), dtype=int)
        self.foundEval2 = np.empty((0,), dtype=int)
        self.size = 0
        # Store widths for scaled distance threshold (default to 10 if not provided)
        self.widths = widths if widths is not None else np.ones(dim) * 10.0

    def add(self, x: np.ndarray, f: float, evals: int) -> bool:
        """Add a solution to archive if it's unique. Returns True if added."""
        # Limit archive size for performance (max 50 solutions per environment)
        MAX_ARCHIVE_SIZE = 50

        # Check if already in archive (scaled distance threshold)
        if self.size > 0:
            dists = np.linalg.norm(self.solution - x[None, :], axis=1)
            # Use 2% of mean search space width as threshold (balance diversity vs performance)
            dist_threshold = 0.02 * np.mean(self.widths)
            min_idx = np.argmin(dists)
            if dists[min_idx] < dist_threshold:
                # If new solution is better, replace existing
                if f > self.value[min_idx]:
                    self.solution[min_idx] = x
                    self.value[min_idx] = f
                    # Keep original foundEval
                return False

        # If archive is full, only add if better than worst
        if self.size >= MAX_ARCHIVE_SIZE:
            worst_idx = np.argmin(self.value)
            if f > self.value[worst_idx]:
                # Replace worst solution
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
        # Keep widths for consistency


class TSDForDMMOP:
    """
    TSD wrapper adapted for DMMOP dynamic multimodal optimization.

    Key features:
    - Detects environment changes via DMMOP's CheckChange()
    - Maintains archive of found optima per environment
    - Responds to dynamics by resetting substrate and population diversity
    - Tracks statistics per environment for postprocessing
    - Periodic exploration to find multiple peaks
    """

    def __init__(
        self,
        problem,
        seed: int = 1,
        N: int = 60,
        budget_per_tick: int = 200,
        # TSD-specific params
        eta: float = 0.25,
        lambda_s: float = 0.5,
        rho: float = 0.98,
        drift_interval: int = 2000,
        # Response params
        response_mode: str = 'partial_reset',  # 'partial_reset', 'full_reset', 'adapt'
        # Multimodal params
        exploration_interval: int = 50,  # Generations between exploration resets
        exploration_rate: float = 0.15,  # Fraction of population to reset
    ):
        self.problem = problem
        self.seed = seed
        self.N = N
        self.budget_per_tick = budget_per_tick
        self.response_mode = response_mode

        # Multimodal parameters
        self.exploration_interval = exploration_interval
        self.exploration_rate = exploration_rate

        # TSD parameters
        self.eta = eta
        self.lambda_s = lambda_s
        self.rho = rho
        self.drift_interval = drift_interval

        # Compute search space widths for archive distance threshold
        self.widths = problem.upBound - problem.lowBound

        # Archive for tracking found optima (pass widths for scaled distance)
        self.archive = TSDArchive(problem.dim, self.widths)
        self.archive_history = []  # Store archives per environment

        # Track confirmed peaks (f >= 74.9) for fitness sharing
        self.confirmed_peaks = []  # List of (position, fitness) tuples

        # Statistics
        self.current_env = 0
        self.env_start_eval = 0
        self.best_per_env = []
        self.env_best_f = float('-inf')  # Track best fitness (MAXIMIZATION: higher is better)
        self.generation = 0

        # Convert problem bounds
        bounds = [(problem.lowBound[i], problem.upBound[i]) for i in range(problem.dim)]

        # Create TSD optimizer with progress callback
        self.tsd = ETFCSA_TSD(
            func=self._objective_wrapper,
            bounds=bounds,
            N=N,
            seed=seed,
            max_evals=problem.maxEval,
            budget_per_tick=budget_per_tick,
            eta=eta,
            lambda_s=lambda_s,
            rho=rho,
            drift_interval=drift_interval,
            progress=self._on_generation,  # Callback for periodic exploration
        )
    
    def _local_refine(self, x: np.ndarray, f: float, max_evals: int = 500) -> Tuple[np.ndarray, float]:
        """
        High-precision local refinement using two-stage coordinate descent.

        Stage 1: Coarse search with larger steps to get close
        Stage 2: Fine search with tiny steps for high precision

        For DMMOP tolerance 0.00001, we need |f - 75| < 0.00001
        With f = 75 - 8*dist, need dist < 0.00001/8 ≈ 1.25e-6

        Args:
            x: Starting solution
            f: Starting fitness (DMMOP value, higher = better)
            max_evals: Maximum evaluations to use

        Returns:
            (refined_x, refined_f)
        """
        x_best = x.copy()
        f_best = f
        evals_used = 0
        golden = 0.618

        # Stage 1: Coarse refinement (use 60% of budget)
        stage1_budget = int(max_evals * 0.6)
        step = np.ones(self.problem.dim) * 0.1

        while evals_used < stage1_budget and np.max(step) > 1e-6:
            # Early exit if already very precise
            if f_best >= 74.99999:
                break

            improved_this_round = False

            for d in range(self.problem.dim):
                if evals_used >= stage1_budget:
                    break

                for direction in [+1.0, -1.0]:
                    if evals_used >= stage1_budget:
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

        # Stage 2: Fine refinement with tiny steps (remaining budget)
        step = np.ones(self.problem.dim) * 1e-4  # Start with very small steps

        while evals_used < max_evals and np.max(step) > 1e-10:
            # Early exit if already at machine precision
            if f_best >= 74.999999:
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

    def _on_generation(self, gen: int, pop: np.ndarray, fitness: np.ndarray,
                       best_fitness: float, gbest: np.ndarray, evals: int):
        """Callback after each TSD generation for periodic exploration."""
        self.generation = gen

        # Periodic exploration: teleport worst individuals to unexplored regions
        if gen > 0 and gen % self.exploration_interval == 0 and len(self.confirmed_peaks) > 0:
            self._apply_exploration_reset()

    def _add_confirmed_peak(self, x: np.ndarray, f: float):
        """Add a solution to confirmed peaks if it's far enough from existing peaks."""
        # Distance threshold: 3% of mean search space width (to avoid duplicates)
        dist_threshold = 0.03 * np.mean(self.widths)

        for peak_x, peak_f in self.confirmed_peaks:
            dist = np.linalg.norm(x - peak_x)
            if dist < dist_threshold:
                # Update if better
                if f > peak_f:
                    # Replace in place (find index)
                    for i, (px, pf) in enumerate(self.confirmed_peaks):
                        if np.allclose(px, peak_x):
                            self.confirmed_peaks[i] = (x.copy(), f)
                            break
                return

        # New confirmed peak
        self.confirmed_peaks.append((x.copy(), f))

    def _apply_exploration_reset(self):
        """Teleport worst individuals to regions far from confirmed peaks."""
        if not hasattr(self.tsd, 'pop') or len(self.tsd.pop) == 0:
            return

        pop = self.tsd.pop
        n_reset = max(2, int(len(pop) * self.exploration_rate))

        # Select worst individuals
        fitnesses = np.array([-ab.aff for ab in pop])
        reset_indices = np.argsort(fitnesses)[:n_reset]

        # Get confirmed peak positions
        peak_positions = [p[0] for p in self.confirmed_peaks]

        for idx in reset_indices:
            # Try to find a position far from all peaks
            best_pos = None
            best_min_dist = 0

            for attempt in range(20):
                # Random position
                new_pos = self.problem.lowBound + np.random.random(self.problem.dim) * self.widths

                # Calculate distance to nearest peak
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
                # Teleport individual
                pop[idx].x = best_pos
                pop[idx].T = 0  # Reset age
                pop[idx].S = 0  # Reset selection count
                pop[idx].I = 0.0  # Reset improvement signal

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
            # Environment changed during evaluation
            return float('inf')

        f = float(f_array[0])  # Original DMMOP value (higher = better peak)

        # Track environment best (DMMOP uses MAXIMIZATION - higher is better)
        if f > self.env_best_f:
            self.env_best_f = f

        # For multimodal optimization: archive solutions and refine only the best
        # Use 90% threshold to balance diversity vs archive bloat
        if self.env_best_f > 0:
            quality_ratio = f / self.env_best_f

            if quality_ratio >= 0.995:  # Near-optimal - full refinement for precision
                x_refined, f_refined = self._local_refine(x.copy(), f, max_evals=500)
                self.archive.add(x_refined, f_refined, self.problem.extProb.evaluated)
                if f_refined > self.env_best_f:
                    self.env_best_f = f_refined
                # Track as confirmed peak if f >= 74.9 (near optimal for DMMOP)
                if f_refined >= 74.9:
                    self._add_confirmed_peak(x_refined, f_refined)
            elif quality_ratio >= 0.90:  # Good solutions (90%) - archive for diversity
                self.archive.add(x.copy(), f, self.problem.extProb.evaluated)
                # Also track as confirmed peak if f >= 74.9
                if f >= 74.9:
                    self._add_confirmed_peak(x.copy(), f)
        else:
            self.archive.add(x.copy(), f, self.problem.extProb.evaluated)
            if f >= 74.9:
                self._add_confirmed_peak(x.copy(), f)

        # Check for environment change
        pop_dummy = x.reshape(1, -1)
        fits_dummy = f_array
        changed = self.problem.extProb.CheckChange(pop_dummy, fits_dummy)

        if changed:
            self._handle_environment_change()

        # NICHING: Apply fitness sharing penalty for solutions near CONFIRMED peaks
        # Confirmed peaks are solutions with f >= 74.9 (near optimal for DMMOP)
        # This pushes TSD to explore NEW peaks instead of converging to known ones
        f_adjusted = f
        if len(self.confirmed_peaks) > 0:
            # Calculate distance to nearest confirmed peak
            peak_positions = np.array([p[0] for p in self.confirmed_peaks])
            peak_fitnesses = np.array([p[1] for p in self.confirmed_peaks])
            dists = np.linalg.norm(peak_positions - x[None, :], axis=1)
            min_dist = np.min(dists)
            nearest_idx = np.argmin(dists)

            # Niche radius: 8% of mean search space width (wider for confirmed peaks)
            niche_radius = 0.08 * np.mean(self.widths)

            if min_dist < niche_radius:
                # Strong penalty near confirmed peaks - quadratic decay
                # At distance 0: sharing_factor = 0 (full penalty)
                # At niche_radius: sharing_factor = 1 (no penalty)
                sharing_factor = (min_dist / niche_radius) ** 2

                # Apply penalty - always penalize near confirmed peaks to encourage exploration
                # The stronger penalty (min 0.1) ensures we don't get stuck
                f_adjusted = f * max(0.1, sharing_factor)

        # Return NEGATED value for TSD (which uses minimization)
        # TSD will minimize -f_adjusted, which maximizes f_adjusted
        # Note: Archive stores original f, not f_adjusted (for correct PR calculation)
        return -f_adjusted
    
    def _handle_environment_change(self):
        """
        Respond to environment change detected by DMMOP.
        """
        # Log archive and confirmed peaks stats before saving
        if self.archive.size > 0:
            print(f"\n[Env {self.current_env}] Archive stats:")
            print(f"  Size: {self.archive.size} solutions")
            print(f"  Best fitness: {np.max(self.archive.value):.6f}")
            print(f"  Fitness range: [{np.min(self.archive.value):.6f}, {np.max(self.archive.value):.6f}]")
            print(f"  Confirmed peaks (f>=74.9): {len(self.confirmed_peaks)}")

            # Show pairwise distances between archived solutions
            if self.archive.size > 1:
                dists = []
                for i in range(self.archive.size):
                    for j in range(i+1, self.archive.size):
                        dist = np.linalg.norm(self.archive.solution[i] - self.archive.solution[j])
                        dists.append(dist)
                print(f"  Pairwise distances: min={np.min(dists):.4f}, mean={np.mean(dists):.4f}, max={np.max(dists):.4f}")
                print(f"  Distance threshold: {0.02 * np.mean(self.widths):.4f}")

        # Save current archive
        self.archive_history.append({
            'env': self.current_env,
            'archive': self.archive,
            'evals': self.problem.numCallF,
        })

        self.current_env += 1
        self.env_start_eval = self.problem.numCallF
        
        # Response strategy
        if self.response_mode == 'full_reset':
            # Complete reset: reinitialize population and substrate
            self.tsd._init()
            
        elif self.response_mode == 'partial_reset':
            # Partial reset: keep best individuals, reset substrate with decay
            self.tsd.s *= 0.5  # Decay substrate
            
            # Keep top 20% of population, randomize rest
            n_keep = max(1, self.N // 5)
            affs = np.array([ab.aff for ab in self.tsd.pop])
            keep_idx = np.argsort(affs)[-n_keep:]
            
            new_pop = [self.tsd.pop[i] for i in keep_idx]
            
            # Re-randomize the rest
            for _ in range(self.N - n_keep):
                x = self.problem.lowBound + np.random.random(self.problem.dim) * (
                    self.problem.upBound - self.problem.lowBound
                )
                f = self._objective_wrapper(x)
                ab = Antibody(x=x, aff=-f)
                # Initialize HSAT attributes for new antibody
                ab.x_prev = x.copy()
                ab.A_sub = 0.0
                ab.A_orth = 1.0  # High exploration initially
                ab.C = 0.0
                ab.displacement_norm = 0.0
                new_pop.append(ab)
            
            self.tsd.pop = new_pop
            
        elif self.response_mode == 'adapt':
            # Adaptive: increase mutation diversity, decay substrate
            self.tsd.s *= 0.7
            # Increase activity penalties temporarily (implemented via parameters)
            for ab in self.tsd.pop:
                ab.T = max(0, ab.T - 50)  # Reduce age to allow more exploration

        # Reset environment best, confirmed peaks, and create new archive for new environment
        self.env_best_f = float('-inf')  # Reset for maximization
        self.confirmed_peaks = []  # Reset confirmed peaks for new environment
        self.archive = TSDArchive(self.problem.dim, self.widths)
    
    def optimize(self):
        """
        Run TSD optimization on DMMOP problem.
        """
        print(f"Starting TSD optimization on DMMOP")
        print(f"Max evaluations: {self.problem.maxEval}")
        print(f"Response mode: {self.response_mode}")
        print(f"Population size: {self.N}")
        print("-" * 60)
        
        start_time = time.time()
        
        # Run TSD
        x_best, f_best, info = self.tsd.optimize()

        # Log final archive stats
        if self.archive.size > 0:
            print(f"\n[Env {self.current_env} - FINAL] Archive stats:")
            print(f"  Size: {self.archive.size} solutions")
            print(f"  Best fitness: {np.max(self.archive.value):.6f}")
            print(f"  Fitness range: [{np.min(self.archive.value):.6f}, {np.max(self.archive.value):.6f}]")

            if self.archive.size > 1:
                dists = []
                for i in range(self.archive.size):
                    for j in range(i+1, self.archive.size):
                        dist = np.linalg.norm(self.archive.solution[i] - self.archive.solution[j])
                        dists.append(dist)
                print(f"  Pairwise distances: min={np.min(dists):.4f}, mean={np.mean(dists):.4f}, max={np.max(dists):.4f}")
                print(f"  Distance threshold: {0.02 * np.mean(self.widths):.4f}")

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


def tsd_driver(problem, seedNo, filename, result_dir='result', **tsd_kwargs):
    """
    Driver function for TSD algorithm on DMMOP problems.

    Args:
        problem: DMMOP problem instance
        seedNo: Random seed
        filename: Output filename
        result_dir: Output directory (default: 'result')
        **tsd_kwargs: Additional TSD parameters

    Returns:
        None (saves results to file)
    """
    np.random.seed(seedNo)

    # Default TSD parameters (can be overridden)
    default_params = {
        'N': 60,
        'budget_per_tick': 200,
        'eta': 0.25,
        'lambda_s': 0.5,
        'rho': 0.98,
        'drift_interval': 2000,
        'response_mode': 'partial_reset',
    }
    default_params.update(tsd_kwargs)
    
    # Create and run TSD optimizer
    tsd = TSDForDMMOP(problem, seed=seedNo, **default_params)
    results = tsd.optimize()
    
    # Prepare data for saving (compatible with postprocess_result.py format)
    # We need to create a structure similar to AMLP's output
    
    # Reconstruct archive with all environments
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
        'algorithm': 'TSD',
        'seed': seedNo,
    }
    
    # Create result directory if it doesn't exist
    result_dir = Path(result_dir)
    result_dir.mkdir(exist_ok=True)
    
    # Save to pickle
    output_path = result_dir / filename
    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"Results saved to: {output_path}")

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
    
    def __init__(self, dim: int):
        self.solution = np.empty((0, dim), dtype=float)
        self.value = np.empty((0,), dtype=float)
        self.foundEval = np.empty((0,), dtype=int)
        self.foundEval2 = np.empty((0,), dtype=int)
        self.size = 0
    
    def add(self, x: np.ndarray, f: float, evals: int):
        """Add a solution to archive if it's unique and good."""
        # Check if already in archive (distance threshold)
        if self.size > 0:
            dists = np.linalg.norm(self.solution - x[None, :], axis=1)
            if np.any(dists < 1e-3):  # Too close to existing solution
                return
        
        # Add to archive
        self.solution = np.vstack([self.solution, x[None, :]])
        self.value = np.append(self.value, f)
        self.foundEval = np.append(self.foundEval, evals)
        self.foundEval2 = np.append(self.foundEval2, evals)
        self.size += 1
    
    def reset(self):
        """Clear archive for new environment."""
        dim = self.solution.shape[1] if self.size > 0 else 0
        self.solution = np.empty((0, dim), dtype=float)
        self.value = np.empty((0,), dtype=float)
        self.foundEval = np.empty((0,), dtype=int)
        self.foundEval2 = np.empty((0,), dtype=int)
        self.size = 0


class TSDForDMMOP:
    """
    TSD wrapper adapted for DMMOP dynamic multimodal optimization.
    
    Key features:
    - Detects environment changes via DMMOP's CheckChange()
    - Maintains archive of found optima per environment
    - Responds to dynamics by resetting substrate and population diversity
    - Tracks statistics per environment for postprocessing
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
    ):
        self.problem = problem
        self.seed = seed
        self.N = N
        self.budget_per_tick = budget_per_tick
        self.response_mode = response_mode
        
        # TSD parameters
        self.eta = eta
        self.lambda_s = lambda_s
        self.rho = rho
        self.drift_interval = drift_interval
        
        # Archive for tracking found optima
        self.archive = TSDArchive(problem.dim)
        self.archive_history = []  # Store archives per environment
        
        # Statistics
        self.current_env = 0
        self.env_start_eval = 0
        self.best_per_env = []
        
        # Convert problem bounds
        bounds = [(problem.lowBound[i], problem.upBound[i]) for i in range(problem.dim)]
        
        # Create TSD optimizer
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
        )
    
    def _objective_wrapper(self, x: np.ndarray) -> float:
        """
        Wrapper that calls problem.GetFits and handles environment changes.
        """
        # Evaluate
        f_array = self.problem.extProb.GetFits(x.reshape(1, -1))
        
        if f_array.size == 0:
            # Environment changed during evaluation
            return float('inf')
        
        f = float(f_array[0])
        
        # Update archive with good solutions
        if f < 1e6:  # Reasonable value
            self.archive.add(x.copy(), f, self.problem.numCallF)
        
        # Check for environment change
        pop_dummy = x.reshape(1, -1)
        fits_dummy = f_array
        changed = self.problem.extProb.CheckChange(pop_dummy, fits_dummy)
        
        if changed:
            self._handle_environment_change()
        
        return f
    
    def _handle_environment_change(self):
        """
        Respond to environment change detected by DMMOP.
        """
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
                new_pop.append(Antibody(x=x, aff=-f))
            
            self.tsd.pop = new_pop
            
        elif self.response_mode == 'adapt':
            # Adaptive: increase mutation diversity, decay substrate
            self.tsd.s *= 0.7
            # Increase activity penalties temporarily (implemented via parameters)
            for ab in self.tsd.pop:
                ab.T = max(0, ab.T - 50)  # Reduce age to allow more exploration
        
        # Create new archive for new environment
        self.archive = TSDArchive(self.problem.dim)
    
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


def tsd_driver(problem, seedNo, filename, **tsd_kwargs):
    """
    Driver function for TSD algorithm on DMMOP problems.
    
    Args:
        problem: DMMOP problem instance
        seedNo: Random seed
        filename: Output filename
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
    result_dir = Path('result')
    result_dir.mkdir(exist_ok=True)
    
    # Save to pickle
    output_path = result_dir / filename
    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"Results saved to: {output_path}")

"""
Postprocess TSD results and save to CSV.

Usage:
    python postprocess_tsd.py result_TSD_F16_D10_seed1.pkl
"""

import numpy as np
import pickle
import argparse
import pandas as pd
from pathlib import Path
import sys

# Add dmmops to path
sys.path.insert(0, str(Path(__file__).parent / 'dmmops_py'))


def calc_pr_dmmop(archive, xopt, fopt, rangeFE, tolf, Rnich):
    """
    Calculate PR for an arbitrary time step (adapted for TSD archives).
    
    Args:
        archive: TSD Archive object or dict with solutions
        xopt: Global optimum positions
        fopt: Global optimum value
        rangeFE: Range of relevant function evaluations [start, end]
        tolf: Target function tolerance
        Rnich: Distance criterion for detection of a global optimum
    
    Returns:
        wasFound: Boolean array indicating which optima were found
        pr: Peak ratio (fraction of optima found)
    """
    # Handle both dict and object representations
    if archive is None:
        wasFound = np.zeros(xopt.shape[0], dtype=bool)
        return wasFound, 0.0
    
    if isinstance(archive, dict):
        archive_size = archive.get('size', 0)
        archive_value = archive.get('value', np.array([]))
        archive_solution = archive.get('solution', np.array([]))
        archive_foundEval = archive.get('foundEval', np.array([]))
    else:
        archive_size = archive.size
        archive_value = archive.value
        archive_solution = archive.solution
        archive_foundEval = archive.foundEval
    
    if archive_size == 0:
        wasFound = np.zeros(xopt.shape[0], dtype=bool)
        return wasFound, 0.0
    
    # Find indexes of relevant solutions
    ind = (archive_foundEval >= rangeFE[0]) & \
          (archive_foundEval <= rangeFE[1]) & \
          ((archive_value - fopt) < tolf)
    
    if np.sum(ind) == 0:  # No good solution
        wasFound = np.zeros(xopt.shape[0], dtype=bool)
        pr = 0.0
    else:
        repX = archive_solution[ind, :]  # Reported solutions
        wasFound = np.zeros(xopt.shape[0], dtype=int)
        
        for k in range(repX.shape[0]):
            # Check the distance metric
            dis = np.linalg.norm(xopt - repX[k, :], axis=1)
            ind_min = np.argmin(dis)
            minDis = dis[ind_min]
            
            if minDis < Rnich:
                wasFound[ind_min] += 1
        
        pr = np.mean(wasFound > 0)
    
    return wasFound, pr


def postprocess_tsd(filename, tolf=None, Rnich=0.05):
    """
    Convert TSD pickle results to CSV files with MPR calculation.
    
    Args:
        filename: Path to TSD result pickle file
        tolf: Target precision levels (default: [0.001, 0.0001, 0.00001])
        Rnich: Distance criterion for detection (default: 0.05)
    """
    if tolf is None:
        tolf = np.array([0.001, 0.0001, 0.00001])
    
    print(f"Loading TSD results from: {filename}")
    
    # Load the saved data
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    problem = data.get('problem')
    archive = data.get('archive')
    tsd_info = data.get('tsd_info', {})
    archive_history = data.get('archive_history', [])
    
    output_dir = Path('result')
    output_dir.mkdir(exist_ok=True)
    
    # Calculate MPR if we have problem data
    mpr = None
    pr = None
    if problem and hasattr(problem, 'extProb'):
        maxEnv = problem.extProb.maxEnv
        n_archives = len(archive_history)
        
        if n_archives == 0:
            print("Warning: No environment archives found!")
            return None, None
        
        print(f"Processing {n_archives} environments (out of {maxEnv} total) for MPR calculation...")
        
        pr = -1 * np.ones((len(tolf), n_archives))
        
        for t in range(n_archives):
            # Get global optima data - his_o grows dynamically so check bounds
            if t >= len(problem.extProb.his_o):
                print(f"Warning: Environment {t} not in his_o (len={len(problem.extProb.his_o)})")
                pr[:, t] = 0.0
                continue
            
            xopt = problem.extProb.his_o[t][2]  # Global optimum positions
            fopt = -np.max(problem.extProb.his_of[t][2])  # Global optimum value
            rangeFE = problem.extProb.his_o[t][1] + np.array([0, problem.extProb.freq])
            
            # Get archive for this environment
            if t < len(archive_history):
                env_archive = archive_history[t]
            else:
                env_archive = None
            
            pr0 = np.zeros(len(tolf))
            
            for tolFind in range(len(tolf)):
                wasFound, pr0[tolFind] = calc_pr_dmmop(
                    env_archive, xopt, fopt, rangeFE, tolf[tolFind], Rnich
                )
            
            pr[:, t] = pr0
            
            if (t + 1) % 10 == 0:
                print(f"  Processed {t + 1}/{n_archives} environments")
        
        mpr = np.mean(pr, axis=1)
        
        print("\nMPR (Mean Peak Ratio) for the selected accuracy:")
        for i, (tol, mpr_val) in enumerate(zip(tolf, mpr)):
            print(f"  tol={tol:.5f}: MPR={mpr_val:.6f}")
    
    # 1. Summary statistics (MPR + algorithm metrics)
    summary_data = {
        'tolerance': tolf.tolist() if mpr is not None else [],
        'MPR': mpr.tolist() if mpr is not None else [],
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = output_dir / f"{Path(filename).stem}_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSummary saved to: {summary_file}")
    
    # 2. Detailed PR per environment (if available)
    if pr is not None:
        pr_df = pd.DataFrame(
            pr.T,
            columns=[f'tol_{tol}' for tol in tolf]
        )
        pr_df.insert(0, 'environment', range(1, pr.shape[1] + 1))
        detail_file = output_dir / f"{Path(filename).stem}_detail.csv"
        pr_df.to_csv(detail_file, index=False)
        print(f"Detailed PR saved to: {detail_file}")
    
    # 3. Archive information
    if archive and archive.size > 0:
        archive_data = []
        for i in range(archive.size):
            archive_data.append({
                'index': i + 1,
                'fitness': archive.value[i],
                'found_at_eval': archive.foundEval[i] if len(archive.foundEval) > i else 0,
            })
        
        archive_df = pd.DataFrame(archive_data)
        archive_file = output_dir / f"{Path(filename).stem}_archive.csv"
        archive_df.to_csv(archive_file, index=False)
        print(f"Archive saved to: {archive_file}")
    
    # 4. Best solution
    best_x = data.get('best_x')
    if best_x is not None:
        solution_data = {
            'dimension': list(range(1, len(best_x) + 1)),
            'value': best_x,
        }
        solution_df = pd.DataFrame(solution_data)
        solution_file = output_dir / f"{Path(filename).stem}_best_solution.csv"
        solution_df.to_csv(solution_file, index=False)
        print(f"Best solution saved to: {solution_file}")
    
    return mpr, pr if pr is not None else None


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Postprocess TSD results and save to CSV'
    )
    parser.add_argument('filename', type=str, help='TSD result pickle file')
    
    args = parser.parse_args()
    
    postprocess_tsd(args.filename)


if __name__ == '__main__':
    main()

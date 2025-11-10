"""
Postprocess results and calculate Peak Ratio (PR) metrics, saving to CSV.

Written by Ali Ahrari (aliahrari1983@gmail.com)
Translated to Python from MATLAB

Usage:
    python postprocess_result.py result_F16_D10_seed1.pkl
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
    Calculate PR for an arbitrary time step.
    
    Args:
        archive: Archive object with solutions
        xopt: Global optimum positions
        fopt: Global optimum value
        rangeFE: Range of relevant function evaluations [start, end]
        tolf: Target function tolerance
        Rnich: Distance criterion for detection of a global optimum
    
    Returns:
        wasFound: Boolean array indicating which optima were found
        pr: Peak ratio (fraction of optima found)
    """
    # Find indexes of relevant solutions
    ind = (archive.foundEval2 >= rangeFE[0]) & \
          (archive.foundEval2 <= rangeFE[1]) & \
          ((archive.value - fopt) < tolf)
    
    if np.sum(ind) == 0:  # No good solution
        wasFound = np.zeros(xopt.shape[0], dtype=bool)
        pr = 0.0
    else:
        repX = archive.solution[ind, :]  # Reported solutions
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


def postprocess_result(filename, tolf=None, Rnich=0.05):
    """
    Calculate PR for a single run for different function tolerances.
    
    Args:
        filename: Path to result pickle file
        tolf: Target precision levels (default: [0.001, 0.0001, 0.00001])
        Rnich: Distance criterion for detection (default: 0.05)
    """
    if tolf is None:
        tolf = np.array([0.001, 0.0001, 0.00001])
    
    print(f"Loading results from: {filename}")
    
    # Load the saved data
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    process = data['process']
    problem = data['problem']
    
    Nenv = problem.extProb.maxEnv
    chFr = problem.extProb.freq
    
    pr = -1 * np.ones((len(tolf), Nenv))
    
    print(f"Processing {Nenv} environments...")
    
    for t in range(Nenv):
        xopt = problem.extProb.his_o[t][2]  # Global optimum positions
        fopt = -np.max(problem.extProb.his_of[t][2])  # Global optimum value
        rangeFE = problem.extProb.his_o[t][1] + np.array([0, problem.extProb.freq])
        
        archive = process.dynamics.endArchive[t]
        pr0 = np.zeros(len(tolf))
        
        for tolFind in range(len(tolf)):
            wasFound, pr0[tolFind] = calc_pr_dmmop(
                archive, xopt, fopt, rangeFE, tolf[tolFind], Rnich
            )
        
        pr[:, t] = pr0
        
        if (t + 1) % 10 == 0:
            print(f"  Processed {t + 1}/{Nenv} environments")
    
    mpr = np.mean(pr, axis=1)
    
    print("\nMPR (Mean Peak Ratio) for the selected accuracy:")
    for i, (tol, mpr_val) in enumerate(zip(tolf, mpr)):
        print(f"  tol={tol:.5f}: MPR={mpr_val:.6f}")
    
    # Save results to CSV
    output_dir = Path('result')
    output_dir.mkdir(exist_ok=True)
    
    # Create DataFrames for different outputs
    
    # 1. Summary statistics (MPR)
    summary_df = pd.DataFrame({
        'tolerance': tolf,
        'MPR': mpr
    })
    summary_file = output_dir / f"{Path(filename).stem}_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSummary saved to: {summary_file}")
    
    # 2. Detailed PR per environment
    pr_df = pd.DataFrame(
        pr.T,
        columns=[f'tol_{tol}' for tol in tolf]
    )
    pr_df.insert(0, 'environment', range(1, Nenv + 1))
    detail_file = output_dir / f"{Path(filename).stem}_detail.csv"
    pr_df.to_csv(detail_file, index=False)
    print(f"Detailed PR saved to: {detail_file}")
    
    # 3. Archive information per environment
    archive_data = []
    for t in range(Nenv):
        if t < len(process.dynamics.endArchive):
            archive = process.dynamics.endArchive[t]
            archive_data.append({
                'environment': t + 1,
                'archive_size': archive.size,
                'best_value': np.min(archive.value) if archive.size > 0 else np.nan,
                'num_solutions': archive.size
            })
    
    archive_df = pd.DataFrame(archive_data)
    archive_file = output_dir / f"{Path(filename).stem}_archive.csv"
    archive_df.to_csv(archive_file, index=False)
    print(f"Archive info saved to: {archive_file}")
    
    return mpr, pr


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Postprocess AMLP-RS-CMSA-ESII results and calculate PR metrics'
    )
    parser.add_argument('filename', type=str, help='Result pickle file')
    parser.add_argument('--tolf', type=float, nargs='+', 
                       default=[0.001, 0.0001, 0.00001],
                       help='Target precision levels (default: 0.001 0.0001 0.00001)')
    parser.add_argument('--Rnich', type=float, default=0.05,
                       help='Distance criterion for detection (default: 0.05)')
    
    args = parser.parse_args()
    
    postprocess_result(args.filename, tolf=np.array(args.tolf), Rnich=args.Rnich)


if __name__ == '__main__':
    main()

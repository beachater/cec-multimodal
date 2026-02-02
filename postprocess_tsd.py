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
    in_range = (archive_foundEval >= rangeFE[0]) & (archive_foundEval <= rangeFE[1])
    within_tolf = np.abs(archive_value - fopt) < tolf
    ind = in_range & within_tolf

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

        # Only process environments that have ground truth data
        n_process = min(n_archives, maxEnv, len(problem.extProb.his_o))
        print(f"Processing {n_process} environments (archives={n_archives}, maxEnv={maxEnv}) for MPR calculation...")

        pr = -1 * np.ones((len(tolf), n_process))

        for t in range(n_process):

            xopt = problem.extProb.his_o[t][2]  # Global optimum positions
            fopt = np.max(problem.extProb.his_of[t][2])  # Global optimum value (positive for maximization)
            rangeFE = problem.extProb.his_o[t][1] + np.array([0, problem.extProb.freq])
            
            # Get archive for this environment
            # Note: archive_history entries are dicts: {'env': int, 'archive': TSDArchive, 'evals': int}
            if t < len(archive_history):
                entry = archive_history[t]
                # Extract the actual archive from the nested dict structure
                if isinstance(entry, dict) and 'archive' in entry:
                    env_archive = entry['archive']
                else:
                    env_archive = entry  # Fallback for legacy format
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
    
    # 3. Archive information per environment (same format as AMLP)
    archive_data = []
    for t in range(len(archive_history)):
        entry = archive_history[t]
        if isinstance(entry, dict) and 'archive' in entry:
            env_archive = entry['archive']
        else:
            env_archive = entry

        if env_archive is not None and env_archive.size > 0:
            archive_data.append({
                'environment': t + 1,
                'archive_size': env_archive.size,
                'best_value': np.max(env_archive.value),  # max for maximization
                'num_solutions': env_archive.size
            })
        else:
            archive_data.append({
                'environment': t + 1,
                'archive_size': 0,
                'best_value': np.nan,
                'num_solutions': 0
            })

    if archive_data:
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


def postprocess_batch(result_dir: str, output_file: str = None):
    """
    Postprocess all TSD results in a directory and create aggregate summary.

    Args:
        result_dir: Directory containing .pkl result files
        output_file: Output CSV for aggregate summary (default: result_dir/aggregate_mpr.csv)
    """
    result_path = Path(result_dir)
    pkl_files = sorted(result_path.glob("*.pkl"))

    if not pkl_files:
        print(f"No .pkl files found in {result_dir}")
        return

    print(f"Found {len(pkl_files)} result files in {result_dir}")
    print("=" * 60)

    # Collect results
    all_results = []
    tolf = np.array([0.001, 0.0001, 0.00001])

    for pkl_file in pkl_files:
        print(f"\nProcessing: {pkl_file.name}")
        try:
            mpr, _ = postprocess_tsd(str(pkl_file), tolf=tolf)

            # Parse filename for metadata (e.g., smoke_test_result_TSD_F1_D5_seed1.pkl)
            name = pkl_file.stem
            parts = name.split('_')

            # Extract function and seed from filename
            func_num = None
            seed_num = None
            dim = None
            for p in parts:
                if p.startswith('F') and p[1:].isdigit():
                    func_num = int(p[1:])
                elif p.startswith('seed') and p[4:].isdigit():
                    seed_num = int(p[4:])
                elif p.startswith('D') and p[1:].isdigit():
                    dim = int(p[1:])

            if mpr is not None:
                all_results.append({
                    'filename': pkl_file.name,
                    'function': func_num,
                    'dimension': dim,
                    'seed': seed_num,
                    'MPR_0.001': mpr[0],
                    'MPR_0.0001': mpr[1],
                    'MPR_0.00001': mpr[2],
                })
        except Exception as e:
            print(f"  Error: {e}")
            continue

    if not all_results:
        print("\nNo results to aggregate!")
        return

    # Create aggregate DataFrame
    df = pd.DataFrame(all_results)

    # Save individual results
    if output_file is None:
        output_file = result_path / "aggregate_mpr.csv"
    df.to_csv(output_file, index=False)
    print(f"\n{'=' * 60}")
    print(f"Individual results saved to: {output_file}")

    # Create AMLP-style aggregated summary (same format as AMLP_F1_D5_aggregated.csv)
    # Format: tolerance,mean_MPR,worst_MPR,best_MPR,std_MPR
    tolerances = [0.001, 0.0001, 0.00001]
    mpr_cols = ['MPR_0.001', 'MPR_0.0001', 'MPR_0.00001']

    aggregated_data = []
    for tol, col in zip(tolerances, mpr_cols):
        values = df[col].values
        aggregated_data.append({
            'tolerance': f'{tol:.6f}',
            'mean_MPR': f'{np.mean(values):.6f}',
            'worst_MPR': f'{np.min(values):.6f}',
            'best_MPR': f'{np.max(values):.6f}',
            'std_MPR': f'{np.std(values):.6f}',
        })

    aggregated_df = pd.DataFrame(aggregated_data)

    # Determine output filename based on function number
    func_nums = df['function'].dropna().unique()
    dim_nums = df['dimension'].dropna().unique()
    if len(func_nums) == 1 and len(dim_nums) == 1:
        agg_filename = f"TSD_F{int(func_nums[0])}_D{int(dim_nums[0])}_aggregated.csv"
    else:
        agg_filename = "TSD_aggregated.csv"

    agg_file = result_path / agg_filename
    aggregated_df.to_csv(agg_file, index=False)
    print(f"Aggregated summary saved to: {agg_file}")

    # Print aggregated summary
    print(f"\n{'=' * 60}")
    print("Aggregated MPR Summary (AMLP format):")
    print("-" * 60)
    print(f"{'tolerance':<12} {'mean_MPR':<12} {'worst_MPR':<12} {'best_MPR':<12} {'std_MPR':<12}")
    for row in aggregated_data:
        print(f"{row['tolerance']:<12} {row['mean_MPR']:<12} {row['worst_MPR']:<12} {row['best_MPR']:<12} {row['std_MPR']:<12}")

    # Create summary by function (mean and std across seeds)
    if 'function' in df.columns and df['function'].notna().any():
        summary = df.groupby('function').agg({
            'MPR_0.001': ['mean', 'std', 'count'],
            'MPR_0.0001': ['mean', 'std'],
            'MPR_0.00001': ['mean', 'std'],
        }).round(6)

        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.reset_index()

        summary_file = result_path / "summary_by_function.csv"
        summary.to_csv(summary_file, index=False)
        print(f"\nSummary by function saved to: {summary_file}")

        # Print summary table
        print(f"\n{'=' * 60}")
        print("MPR Summary by Function (tol=0.001):")
        print("-" * 40)
        for _, row in summary.iterrows():
            func = int(row['function'])
            mean = row['MPR_0.001_mean']
            std = row['MPR_0.001_std']
            n = int(row['MPR_0.001_count'])
            print(f"  F{func:2d}: {mean:.4f} ± {std:.4f} (n={n})")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Postprocess TSD results and save to CSV'
    )
    parser.add_argument('filename', type=str, nargs='?', default=None,
                        help='TSD result pickle file (or directory for batch)')
    parser.add_argument('--batch', '-b', type=str, default=None,
                        help='Directory containing .pkl files for batch processing')

    args = parser.parse_args()

    if args.batch:
        postprocess_batch(args.batch)
    elif args.filename:
        if Path(args.filename).is_dir():
            postprocess_batch(args.filename)
        else:
            postprocess_tsd(args.filename)
    else:
        print("Usage:")
        print("  Single file:  python postprocess_tsd.py result_TSD_F1_seed1.pkl")
        print("  Batch:        python postprocess_tsd.py --batch result_tsd_full")
        print("  Batch (alt):  python postprocess_tsd.py result_tsd_full/")


if __name__ == '__main__':
    main()

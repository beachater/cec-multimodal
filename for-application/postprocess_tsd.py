"""
Postprocess TSD results and save to CSV.

CEC 2022 Competition Compliant Implementation.
Implements Formula (13) for PR calculation: PR = Sum(NPF) / Sum(Peaks)

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


def calc_npf_dmmop(archive, xopt, fopt, rangeFE, tolf, Rnich):
    """
    Calculate NPF (Number of Peaks Found) for an environment.

    Implements Algorithm 1 from CEC 2022 specification.

    Args:
        archive: TSD Archive object or dict with solutions
        xopt: Global optimum positions (n_peaks x dim)
        fopt: Global optimum value
        rangeFE: Range of relevant function evaluations [start, end]
        tolf: Target function tolerance (epsilon_f)
        Rnich: Distance criterion for detection (epsilon_d = 0.05)

    Returns:
        npf: Number of peaks found (NPF)
        n_peaks: Total number of peaks in this environment
    """
    n_peaks = xopt.shape[0]

    # Handle both dict and object representations
    if archive is None:
        return 0, n_peaks

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
        return 0, n_peaks

    # Find indexes of relevant solutions (within FE range and fitness tolerance)
    in_range = (archive_foundEval >= rangeFE[0]) & (archive_foundEval <= rangeFE[1])
    within_tolf = np.abs(archive_value - fopt) < tolf
    ind = in_range & within_tolf

    if np.sum(ind) == 0:  # No good solution
        return 0, n_peaks

    # Algorithm 1: Obtain NPF value
    repX = archive_solution[ind, :]  # Reported solutions (population P)
    found_set = set()  # Set S of found optima

    for k in range(repX.shape[0]):
        # Find nearest global peak o' to individual x_i
        dis = np.linalg.norm(xopt - repX[k, :], axis=1)
        ind_min = np.argmin(dis)
        minDis = dis[ind_min]

        # Check: |F(x_i) - F(o')| < epsilon_f AND ||x_i - o'|| < epsilon_d
        # Note: fitness check already done above (within_tolf)
        if minDis < Rnich:
            found_set.add(ind_min)

    npf = len(found_set)
    return npf, n_peaks


def postprocess_tsd(filename, tolf=None, Rnich=0.05):
    """
    Convert TSD pickle results to CSV files with PR calculation.

    CEC 2022 Compliant: Uses Formula (13) PR = Sum(NPF) / Sum(Peaks)

    Args:
        filename: Path to TSD result pickle file
        tolf: Target precision levels (default: [0.001, 0.0001, 0.00001])
        Rnich: Distance criterion for detection (default: 0.05 per spec)

    Returns:
        result_dict: Dictionary containing NPF/Peaks data for aggregation
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

    # Calculate PR if we have problem data (CEC 2022 compliant)
    result_dict = None
    if problem and hasattr(problem, 'extProb'):
        maxEnv = problem.extProb.maxEnv
        n_archives = len(archive_history)

        if n_archives == 0:
            print("Warning: No environment archives found!")
            return None

        # Only process environments that have ground truth data
        n_process = min(n_archives, maxEnv, len(problem.extProb.his_o))
        print(f"Processing {n_process} environments (archives={n_archives}, maxEnv={maxEnv}) for PR calculation...")

        # Store NPF and total peaks per environment for each tolerance
        # Shape: (n_tolf, n_envs)
        npf_per_env = np.zeros((len(tolf), n_process), dtype=int)
        peaks_per_env = np.zeros((len(tolf), n_process), dtype=int)

        for t in range(n_process):
            xopt = problem.extProb.his_o[t][2]  # Global optimum positions
            fopt = np.max(problem.extProb.his_of[t][2])  # Global optimum value (positive for maximization)
            rangeFE = problem.extProb.his_o[t][1] + np.array([0, problem.extProb.freq])

            # Get archive for this environment
            if t < len(archive_history):
                entry = archive_history[t]
                if isinstance(entry, dict) and 'archive' in entry:
                    env_archive = entry['archive']
                else:
                    env_archive = entry  # Fallback for legacy format
            else:
                env_archive = None

            for tolFind in range(len(tolf)):
                npf, n_peaks = calc_npf_dmmop(
                    env_archive, xopt, fopt, rangeFE, tolf[tolFind], Rnich
                )
                npf_per_env[tolFind, t] = npf
                peaks_per_env[tolFind, t] = n_peaks

            if (t + 1) % 10 == 0:
                print(f"  Processed {t + 1}/{n_process} environments")

        # Calculate PR using Formula (13): PR = Sum(NPF) / Sum(Peaks)
        total_npf = np.sum(npf_per_env, axis=1)  # Sum across environments
        total_peaks = np.sum(peaks_per_env, axis=1)
        pr = total_npf / total_peaks  # CEC 2022 compliant PR

        print("\nPR (Peak Ratio - CEC 2022 Formula 13) for the selected accuracy:")
        for i, (tol, pr_val) in enumerate(zip(tolf, pr)):
            print(f"  epsilon_f={tol:.5f}: PR={pr_val:.6f} (NPF={total_npf[i]}, Peaks={total_peaks[i]})")

        # Store results for aggregation across runs
        result_dict = {
            'pr': pr,
            'total_npf': total_npf,
            'total_peaks': total_peaks,
            'npf_per_env': npf_per_env,
            'peaks_per_env': peaks_per_env,
            'n_envs': n_process,
        }
    
    # 1. Summary statistics (PR per tolerance - CEC 2022 compliant)
    if result_dict is not None:
        summary_data = {
            'tolerance': tolf.tolist(),
            'PR': result_dict['pr'].tolist(),
            'NPF': result_dict['total_npf'].tolist(),
            'Total_Peaks': result_dict['total_peaks'].tolist(),
        }

        summary_df = pd.DataFrame(summary_data)
        summary_file = output_dir / f"{Path(filename).stem}_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSummary saved to: {summary_file}")

        # 2. Detailed NPF per environment (for analysis)
        npf_per_env = result_dict['npf_per_env']
        peaks_per_env = result_dict['peaks_per_env']
        n_envs = result_dict['n_envs']

        detail_data = {
            'environment': list(range(1, n_envs + 1)),
        }
        for i, tol in enumerate(tolf):
            detail_data[f'NPF_tol_{tol}'] = npf_per_env[i, :].tolist()
            detail_data[f'Peaks_tol_{tol}'] = peaks_per_env[i, :].tolist()

        detail_df = pd.DataFrame(detail_data)
        detail_file = output_dir / f"{Path(filename).stem}_detail.csv"
        detail_df.to_csv(detail_file, index=False)
        print(f"Detailed NPF saved to: {detail_file}")

    # 3. Archive information per environment
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

    return result_dict


def postprocess_batch(result_dir: str, output_file: str = None):
    """
    Postprocess all TSD results in a directory and create CEC 2022 compliant aggregate summary.

    Implements Formula (13): PR = Sum(NPF_ij) / Sum(Peaks_ij) across all runs and environments.

    Args:
        result_dir: Directory containing .pkl result files
        output_file: Output CSV for aggregate summary
    """
    result_path = Path(result_dir)
    pkl_files = sorted(result_path.glob("*.pkl"))

    if not pkl_files:
        print(f"No .pkl files found in {result_dir}")
        return

    print(f"Found {len(pkl_files)} result files in {result_dir}")
    print("=" * 60)

    # Collect results for CEC 2022 compliant aggregation
    tolf = np.array([0.001, 0.0001, 0.00001])
    all_run_results = []  # Store per-run data for aggregation

    for pkl_file in pkl_files:
        print(f"\nProcessing: {pkl_file.name}")
        try:
            result_dict = postprocess_tsd(str(pkl_file), tolf=tolf)

            # Parse filename for metadata (e.g., smoke_test_result_TSD_F1_D5_seed1.pkl)
            name = pkl_file.stem
            parts = name.split('_')

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

            if result_dict is not None:
                all_run_results.append({
                    'filename': pkl_file.name,
                    'function': func_num,
                    'dimension': dim,
                    'seed': seed_num,
                    # Per-run PR (for Best/Worst calculation)
                    'PR_1e-3': result_dict['pr'][0],
                    'PR_1e-4': result_dict['pr'][1],
                    'PR_1e-5': result_dict['pr'][2],
                    # NPF and Peaks totals for this run (for Formula 13 aggregation)
                    'NPF_1e-3': result_dict['total_npf'][0],
                    'NPF_1e-4': result_dict['total_npf'][1],
                    'NPF_1e-5': result_dict['total_npf'][2],
                    'Peaks_1e-3': result_dict['total_peaks'][0],
                    'Peaks_1e-4': result_dict['total_peaks'][1],
                    'Peaks_1e-5': result_dict['total_peaks'][2],
                })
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_run_results:
        print("\nNo results to aggregate!")
        return

    df = pd.DataFrame(all_run_results)

    # Save individual run results
    if output_file is None:
        output_file = result_path / "individual_runs.csv"
    df.to_csv(output_file, index=False)
    print(f"\n{'=' * 60}")
    print(f"Individual run results saved to: {output_file}")

    # =========================================================================
    # CEC 2022 Compliant PR Calculation (Formula 13)
    # PR = Sum(NPF_ij over all runs i and envs j) / Sum(Peaks_ij)
    # =========================================================================
    tolerances = [1e-3, 1e-4, 1e-5]
    tol_names = ['1e-3', '1e-4', '1e-5']

    # Calculate overall PR using Formula (13)
    cec2022_results = []
    for tol, tol_name in zip(tolerances, tol_names):
        total_npf = df[f'NPF_{tol_name}'].sum()
        total_peaks = df[f'Peaks_{tol_name}'].sum()
        overall_pr = total_npf / total_peaks if total_peaks > 0 else 0.0

        # Best and Worst are the best/worst per-run PR values
        per_run_pr = df[f'PR_{tol_name}'].values
        best_pr = np.max(per_run_pr)
        worst_pr = np.min(per_run_pr)

        cec2022_results.append({
            'epsilon_f': f'{tol:.0e}',
            'PR': f'{overall_pr:.6f}',
            'Best': f'{best_pr:.6f}',
            'Worst': f'{worst_pr:.6f}',
        })

    # Create CEC 2022 Table 2 format output
    cec2022_df = pd.DataFrame(cec2022_results)

    # Determine output filename based on function number
    func_nums = df['function'].dropna().unique()
    dim_nums = df['dimension'].dropna().unique()
    if len(func_nums) == 1 and len(dim_nums) == 1:
        agg_filename = f"TSD_F{int(func_nums[0])}_D{int(dim_nums[0])}_CEC2022.csv"
    else:
        agg_filename = "TSD_CEC2022_results.csv"

    agg_file = result_path / agg_filename
    cec2022_df.to_csv(agg_file, index=False)
    print(f"CEC 2022 compliant results saved to: {agg_file}")

    # Print CEC 2022 format summary (Table 2 style)
    print(f"\n{'=' * 60}")
    print("CEC 2022 Competition Results (Table 2 Format):")
    print("-" * 60)
    n_runs = len(df)
    n_envs = 60  # CEC 2022 spec: 60 environments
    print(f"Runs: {n_runs}, Environments per run: {n_envs}")
    print("-" * 60)
    print(f"{'epsilon_f':<12} {'PR':<12} {'Best':<12} {'Worst':<12}")
    for row in cec2022_results:
        print(f"{row['epsilon_f']:<12} {row['PR']:<12} {row['Best']:<12} {row['Worst']:<12}")

    # Additional: Summary statistics across seeds
    print(f"\n{'=' * 60}")
    print("Per-Run PR Statistics (for reference):")
    print("-" * 60)
    for tol_name in tol_names:
        pr_col = f'PR_{tol_name}'
        values = df[pr_col].values
        print(f"  epsilon_f={tol_name}: mean={np.mean(values):.4f}, std={np.std(values):.4f}, "
              f"min={np.min(values):.4f}, max={np.max(values):.4f}")

    # Create summary by function if multiple functions present
    if 'function' in df.columns and df['function'].notna().any() and len(func_nums) > 1:
        print(f"\n{'=' * 60}")
        print("PR by Function (epsilon_f=1e-3):")
        print("-" * 40)
        for func in sorted(func_nums):
            func_df = df[df['function'] == func]
            total_npf = func_df['NPF_1e-3'].sum()
            total_peaks = func_df['Peaks_1e-3'].sum()
            pr = total_npf / total_peaks if total_peaks > 0 else 0.0
            n = len(func_df)
            print(f"  F{int(func):2d}: PR={pr:.4f} (n={n} runs)")


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

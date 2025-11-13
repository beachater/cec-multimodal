"""
Aggregate results from multiple seeds (30 runs) and compute mean statistics.

This script processes all runs for each function and algorithm, computing:
- Mean Peak Ratio (MPR) across all seeds
- Standard deviation of MPR
- Mean best fitness
- Mean archive size
- Other aggregate statistics

Usage:
    python aggregate_results.py --algorithm amlp --function 1
    python aggregate_results.py --algorithm tsd --functions 1-24
    python aggregate_results.py --algorithm both --functions 1-24  # Aggregate both algorithms
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
import sys

# Add dmmops to path
sys.path.insert(0, str(Path(__file__).parent / 'dmmops_py'))


def load_amlp_results(function_num: int, seeds: List[int]) -> Dict:
    """Load and aggregate AMLP results across seeds."""
    result_dir = Path('result')
    
    best_values = []
    archive_sizes = []
    total_evals = []
    mprs = {0.001: [], 0.0001: [], 0.00001: []}
    
    successful_seeds = []
    
    for seed in seeds:
        # Try different dimension patterns (D5 for F1-12, D10 for F13-24 typically)
        pkl_patterns = [
            f'result_F{function_num}_D5_seed{seed}.pkl',
            f'result_F{function_num}_D10_seed{seed}.pkl',
            f'result_F{function_num}_D20_seed{seed}.pkl',
        ]
        
        pkl_path = None
        for pattern in pkl_patterns:
            potential_path = result_dir / pattern
            if potential_path.exists():
                pkl_path = potential_path
                break
        
        if pkl_path is None:
            print(f"  Warning: No result file found for seed {seed}")
            continue
        
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            archive = data.get('archive')
            problem = data.get('problem')
            
            if archive and archive.size > 0:
                best_values.append(np.min(archive.value))
                archive_sizes.append(archive.size)
            
            if problem:
                total_evals.append(problem.numCallF)
            
            # Compute MPR directly without saving individual CSVs
            process = data.get('process')
            if process and problem:
                try:
                    from postprocess_result import calc_pr_dmmop
                    
                    Nenv = problem.extProb.maxEnv
                    pr_seed = {0.001: [], 0.0001: [], 0.00001: []}
                    
                    for t in range(Nenv):
                        xopt = problem.extProb.his_o[t][2]
                        fopt = -np.max(problem.extProb.his_of[t][2])
                        rangeFE = problem.extProb.his_o[t][1] + np.array([0, problem.extProb.freq])
                        archive_t = process.dynamics.endArchive[t]
                        
                        for tol in [0.001, 0.0001, 0.00001]:
                            wasFound, pr = calc_pr_dmmop(
                                archive_t, xopt, fopt, rangeFE, tol, Rnich=0.05
                            )
                            pr_seed[tol].append(pr)
                    
                    # Compute MPR (mean across environments) for this seed
                    for tol in [0.001, 0.0001, 0.00001]:
                        mprs[tol].append(np.mean(pr_seed[tol]))
                    
                except Exception as e:
                    print(f"    Warning: MPR calculation failed for seed {seed}: {e}")
            
            successful_seeds.append(seed)
            
        except Exception as e:
            print(f"  Warning: Error loading seed {seed}: {e}")
            continue
    
    if not successful_seeds:
        return None
    
    # Calculate statistics for MPR at each tolerance
    result = {
        'algorithm': 'AMLP',
        'function': function_num,
        'successful_runs': len(successful_seeds),
        'mean_best_fitness': np.mean(best_values) if best_values else np.nan,
        'std_best_fitness': np.std(best_values) if best_values else np.nan,
        'mean_archive_size': np.mean(archive_sizes) if archive_sizes else np.nan,
        'std_archive_size': np.std(archive_sizes) if archive_sizes else np.nan,
        'mean_evaluations': np.mean(total_evals) if total_evals else np.nan,
        'std_evaluations': np.std(total_evals) if total_evals else np.nan,
    }
    
    # Add worst, best, mean for each tolerance level
    for tol in [0.001, 0.0001, 0.00001]:
        tol_str = f"{tol:.5f}".replace('.', '_')
        if mprs[tol]:
            result[f'worst_mpr_{tol_str}'] = np.min(mprs[tol])
            result[f'best_mpr_{tol_str}'] = np.max(mprs[tol])
            result[f'mean_mpr_{tol_str}'] = np.mean(mprs[tol])
            result[f'std_mpr_{tol_str}'] = np.std(mprs[tol])
        else:
            result[f'worst_mpr_{tol_str}'] = np.nan
            result[f'best_mpr_{tol_str}'] = np.nan
            result[f'mean_mpr_{tol_str}'] = np.nan
            result[f'std_mpr_{tol_str}'] = np.nan
    
    return result


def load_tsd_results(function_num: int, seeds: List[int]) -> Dict:
    """Load and aggregate TSD results across seeds."""
    result_dir = Path('result')
    
    best_values = []
    archive_sizes = []
    total_evals = []
    generations = []
    elapsed_times = []
    
    successful_seeds = []
    
    for seed in seeds:
        # Try different dimension patterns
        pkl_patterns = [
            f'result_TSD_F{function_num}_D5_seed{seed}.pkl',
            f'result_TSD_F{function_num}_D10_seed{seed}.pkl',
            f'result_TSD_F{function_num}_D20_seed{seed}.pkl',
        ]
        
        pkl_path = None
        for pattern in pkl_patterns:
            potential_path = result_dir / pattern
            if potential_path.exists():
                pkl_path = potential_path
                break
        
        if pkl_path is None:
            print(f"  Warning: No TSD result file found for seed {seed}")
            continue
        
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            archive = data.get('archive')
            tsd_info = data.get('tsd_info', {})
            
            if archive and archive.size > 0:
                best_values.append(np.min(archive.value))
                archive_sizes.append(archive.size)
            elif data.get('best_f') is not None:
                best_values.append(data.get('best_f'))
            
            total_evals.append(tsd_info.get('evals_used', 0))
            generations.append(tsd_info.get('generations_run', 0))
            elapsed_times.append(data.get('elapsed_time', 0.0))
            
            successful_seeds.append(seed)
            
        except Exception as e:
            print(f"  Warning: Error loading TSD seed {seed}: {e}")
            continue
    
    if not successful_seeds:
        return None
    
    return {
        'algorithm': 'TSD',
        'function': function_num,
        'successful_runs': len(successful_seeds),
        'mean_best_fitness': np.mean(best_values) if best_values else np.nan,
        'std_best_fitness': np.std(best_values) if best_values else np.nan,
        'mean_archive_size': np.mean(archive_sizes) if archive_sizes else np.nan,
        'std_archive_size': np.std(archive_sizes) if archive_sizes else np.nan,
        'mean_evaluations': np.mean(total_evals) if total_evals else np.nan,
        'std_evaluations': np.std(total_evals) if total_evals else np.nan,
        'mean_generations': np.mean(generations) if generations else np.nan,
        'std_generations': np.std(generations) if generations else np.nan,
        'mean_elapsed_time': np.mean(elapsed_times) if elapsed_times else np.nan,
        'std_elapsed_time': np.std(elapsed_times) if elapsed_times else np.nan,
    }


def parse_range(range_str):
    """Parse range string like '1-5' or '1,3,5' into list of integers."""
    result = []
    for part in range_str.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def main():
    parser = argparse.ArgumentParser(description='Aggregate results from multiple seeds')
    parser.add_argument('--algorithm', type=str, default='both',
                        choices=['amlp', 'tsd', 'both'],
                        help='Which algorithm(s) to aggregate')
    parser.add_argument('--functions', type=str, default='1-24',
                        help='Function range (e.g., "1-24", "1,5,10", "16")')
    parser.add_argument('--seeds', type=str, default='1-30',
                        help='Seed range (e.g., "1-30", "1-10")')
    parser.add_argument('--output', type=str, default='aggregated_results.csv',
                        help='Output CSV filename')
    
    args = parser.parse_args()
    
    # Parse ranges
    function_nums = parse_range(args.functions)
    seeds = parse_range(args.seeds)
    
    # Determine algorithms
    if args.algorithm == 'both':
        algorithms = ['amlp', 'tsd']
    else:
        algorithms = [args.algorithm]
    
    print("=" * 70)
    print("AGGREGATING RESULTS FROM MULTIPLE SEEDS")
    print("=" * 70)
    print(f"Functions: {function_nums}")
    print(f"Seeds: {seeds} ({len(seeds)} runs per function)")
    print(f"Algorithms: {', '.join(alg.upper() for alg in algorithms)}")
    print("=" * 70)
    
    all_results = []
    
    for alg in algorithms:
        print(f"\nProcessing {alg.upper()} results...")
        
        for fn in function_nums:
            print(f"  Function {fn}...", end=' ')
            
            if alg == 'amlp':
                result = load_amlp_results(fn, seeds)
            else:  # tsd
                result = load_tsd_results(fn, seeds)
            
            if result:
                all_results.append(result)
                print(f"✓ ({result['successful_runs']}/{len(seeds)} runs)")
            else:
                print("✗ (no data)")
    
    if not all_results:
        print("\nNo results found!")
        sys.exit(1)
    
    # Create DataFrame and save
    df = pd.DataFrame(all_results)
    
    # Sort by algorithm and function
    df = df.sort_values(['algorithm', 'function'])
    
    # Save to CSV
    output_path = Path('result') / args.output
    df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 70)
    print(f"Aggregated results saved to: {output_path}")
    print(f"Total entries: {len(df)}")
    print("=" * 70)
    
    # Display summary
    print("\nSummary:")
    print(df.groupby('algorithm').agg({
        'function': 'count',
        'successful_runs': 'mean',
        'mean_best_fitness': 'mean',
    }).rename(columns={'function': 'num_functions'}))


if __name__ == '__main__':
    main()

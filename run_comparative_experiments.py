"""
Run comparative experiments for AMLP vs TSD on DMMOP benchmark.

This script runs both algorithms on all 24 DMMOP functions with 30 seeds each.
Total experiments: 24 functions × 30 seeds × 2 algorithms = 1440 runs

Usage:
    python run_comparative_experiments.py              # Run all experiments
    python run_comparative_experiments.py --algorithm amlp  # Only AMLP
    python run_comparative_experiments.py --algorithm tsd   # Only TSD
    python run_comparative_experiments.py --functions 1-5   # Specific functions
    python run_comparative_experiments.py --seeds 1-10      # Specific seeds
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import time
import numpy as np
import pandas as pd
import json
from postprocess_result import postprocess_result
from postprocess_tsd import postprocess_tsd


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


def load_checkpoint(checkpoint_file):
    """Load checkpoint from previous run if exists."""
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        except:
            return None
    return None


def save_checkpoint(checkpoint_file, completed_experiments):
    """Save checkpoint of completed experiments."""
    with open(checkpoint_file, 'w') as f:
        json.dump({
            'completed': completed_experiments,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)


def run_experiment(algorithm, function_num, seed, verbose=True):
    """Run a single experiment using absolute paths for reliability."""
    script_dir = Path(__file__).parent.resolve()

    if algorithm == 'amlp':
        entry = script_dir / 'main.py'
        pkl_name = f'result_F{function_num}_D5_seed{seed}.pkl'  # unused here but kept for reference
    elif algorithm == 'tsd':
        entry = script_dir / 'main_tsd.py'
        pkl_name = f'result_TSD_F{function_num}_D5_seed{seed}.pkl'
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    if not entry.exists():
        return False, f"Entry script not found: {entry}"

    cmd = [
        sys.executable,
        str(entry),
        '--function', str(function_num),
        '--seed', str(seed)
    ]

    if verbose:
        print(f"Running: {' '.join(cmd)} (cwd={script_dir})")

    try:
        # Run with suppressed output (no verbose AMLP messages)
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(script_dir)
        )
        # No per-seed CSV conversion here
        return True, None
    except subprocess.CalledProcessError as e:
        return False, f"Error: {e.stderr or str(e)}"


def main():
    parser = argparse.ArgumentParser(description='Run comparative experiments: AMLP vs TSD')
    parser.add_argument('--algorithm', type=str, default='both',
                        choices=['amlp', 'tsd', 'both'],
                        help='Which algorithm(s) to run')
    parser.add_argument('--functions', type=str, default='1-24',
                        help='Function range (e.g., "1-24", "1,5,10", "16")')
    parser.add_argument('--seeds', type=str, default='1-30',
                        help='Seed range (e.g., "1-30", "1-10", "1,5,10")')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed output from each run')
    parser.add_argument('--continue-on-error', action='store_true',
                        help='Continue running even if some experiments fail')
    
    args = parser.parse_args()
    
    # Parse ranges
    function_nums = parse_range(args.functions)
    seeds = parse_range(args.seeds)
    
    # Determine algorithms to run - TSD FIRST, then AMLP
    if args.algorithm == 'both':
        algorithms = ['tsd', 'amlp']
    else:
        algorithms = [args.algorithm]
    
    # Calculate total experiments
    total_experiments = len(algorithms) * len(function_nums) * len(seeds)
    
    print("=" * 70)
    print("COMPARATIVE EXPERIMENTS: AMLP vs TSD on DMMOP Benchmark")
    print("=" * 70)
    print(f"Algorithms: {', '.join(alg.upper() for alg in algorithms)}")
    print(f"Functions: {function_nums}")
    print(f"Seeds: {seeds}")
    print(f"Total experiments: {total_experiments}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Track results
    completed = 0
    failed = 0
    failed_experiments = []
    
    # Setup checkpoint file for crash recovery
    checkpoint_file = Path('result') / 'experiment_checkpoint.json'
    checkpoint_file.parent.mkdir(exist_ok=True)
    
    # Load checkpoint if exists (resume from brownout/crash)
    checkpoint = load_checkpoint(checkpoint_file)
    completed_set = set()
    
    if checkpoint:
        completed_set = set(checkpoint['completed'])
        print(f"\n📂 RESUMING from checkpoint: {len(completed_set)} experiments already completed")
        print(f"   Last run: {checkpoint['timestamp']}")
        print("-" * 70)
    
    start_time = time.time()
    experiment_times = []  # Track individual experiment durations
    
    # Run experiments
    for alg in algorithms:
        for fn in function_nums:
            function_start_seed = len(completed_set) + failed + 1  # Track where function starts
            
            for seed in seeds:
                experiment_id = f"{alg.upper()}_F{fn}_seed{seed}"
                
                # Skip if already completed (from checkpoint)
                if experiment_id in completed_set:
                    print(f"\n⏭️  [{len(completed_set) + failed + 1}/{total_experiments}] {experiment_id} | SKIPPED (already completed)")
                    continue
                
                exp_start = time.time()
                
                # Calculate ETA
                if len(experiment_times) > 0:
                    avg_time = np.mean(experiment_times)
                    remaining = total_experiments - (len(completed_set) + failed)
                    eta_seconds = avg_time * remaining
                    eta_hours, eta_remainder = divmod(eta_seconds, 3600)
                    eta_minutes, eta_secs = divmod(eta_remainder, 60)
                    eta_str = f"{int(eta_hours)}h {int(eta_minutes)}m {int(eta_secs)}s"
                else:
                    eta_str = "calculating..."
                
                print(f"\n[{len(completed_set) + failed + 1}/{total_experiments}] {experiment_id} | ETA: {eta_str}")
                print("-" * 70)
                
                success, error = run_experiment(alg, fn, seed, verbose=args.verbose)
                
                exp_duration = time.time() - exp_start
                experiment_times.append(exp_duration)
                
                if success:
                    completed += 1
                    completed_set.add(experiment_id)
                    # Save checkpoint after each successful run (brownout protection)
                    save_checkpoint(checkpoint_file, list(completed_set))
                    print(f"✓ {experiment_id} completed in {exp_duration:.1f}s")
                else:
                    failed += 1
                    failed_experiments.append((experiment_id, error))
                    print(f"✗ {experiment_id} FAILED: {error}")
                    
                    if not args.continue_on_error:
                        print("\nStopping due to error (use --continue-on-error to continue)")
                        break
            
            # After completing all seeds for this function, postprocess results
            if completed > 0:
                print("\n" + "=" * 70)
                print(f"Post-processing {alg.upper()} Function {fn} results...")
                print("=" * 70)
                
                # Find all result files for this function
                result_dir = Path('result')
                
                # Get dimension from get_info
                sys.path.insert(0, str(Path(__file__).parent / 'dmmops_py'))
                from dmmops_py.problem.get_info import get_info
                _, _, dim = get_info(fn)
                
                # Find result files based on algorithm
                if alg == 'amlp':
                    pattern = f'result_F{fn}_D{dim}_seed*.pkl'
                else:  # tsd
                    pattern = f'result_TSD_F{fn}_D{dim}_seed*.pkl'
                
                result_files = list(result_dir.glob(pattern))
                
                if result_files:
                    print(f"  Found {len(result_files)} result file(s)")
                    
                    # Process each result file to CSV and collect MPR values
                    mpr_values = []  # Store MPR from each seed
                    success_count = 0
                    
                    for result_file in result_files:
                        try:
                            if alg == 'amlp':
                                mpr, _ = postprocess_result(str(result_file))
                            else:  # tsd
                                mpr, _ = postprocess_tsd(str(result_file))
                            
                            if mpr is not None:
                                mpr_values.append(mpr)
                            success_count += 1
                        except Exception as e:
                            print(f"  ⚠ Warning: Failed to process {result_file.name}: {e}")
                    
                    print(f"  ✓ Successfully processed {success_count}/{len(result_files)} files to CSV")
                    
                    # CEC 2022 Requirement: Aggregate statistics across 30 runs
                    if mpr_values:
                        mpr_array = np.array(mpr_values)  # Shape: (n_seeds, n_tolerances)
                        
                        # Calculate mean, worst, best for each tolerance
                        tolerances = [0.001, 0.0001, 0.00001]
                        aggregated_stats = {
                            'tolerance': tolerances,
                            'mean_MPR': np.mean(mpr_array, axis=0).tolist(),
                            'worst_MPR': np.min(mpr_array, axis=0).tolist(),
                            'best_MPR': np.max(mpr_array, axis=0).tolist(),
                            'std_MPR': np.std(mpr_array, axis=0).tolist(),
                        }
                        
                        # Save aggregated results per function (CEC 2022 format)
                        agg_df = pd.DataFrame(aggregated_stats)
                        agg_file = result_dir / f'{alg.upper()}_F{fn}_D{dim}_aggregated.csv'
                        agg_df.to_csv(agg_file, index=False, float_format='%.6f')
                        
                        print(f"\n  CEC 2022 Aggregated Statistics (across {len(mpr_values)} runs):")
                        print(f"  {'Tolerance':<12} {'Mean MPR':<12} {'Worst MPR':<12} {'Best MPR':<12}")
                        print(f"  {'-'*48}")
                        for i, tol in enumerate(tolerances):
                            print(f"  {tol:<12.5f} {aggregated_stats['mean_MPR'][i]:<12.6f} "
                                  f"{aggregated_stats['worst_MPR'][i]:<12.6f} "
                                  f"{aggregated_stats['best_MPR'][i]:<12.6f}")
                        print(f"  ✓ Saved to: {agg_file}")
                else:
                    print(f"  ⚠ No result files found for {alg.upper()} F{fn}")
                
                print("=" * 70)
            
            if failed > 0 and not args.continue_on_error:
                break
        
        if failed > 0 and not args.continue_on_error:
            break
    
    # Summary
    elapsed = time.time() - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Total experiments: {total_experiments}")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {100 * completed / total_experiments:.1f}%")
    print(f"Time elapsed: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if failed > 0:
        print("\nFailed experiments:")
        for exp_id, error in failed_experiments:
            print(f"  - {exp_id}: {error}")
    
    print("=" * 70)
    
    # Clean up checkpoint file if all completed successfully
    if failed == 0 and len(completed_set) == total_experiments:
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print("\n🗑️  Checkpoint file removed (all experiments completed successfully)")
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()

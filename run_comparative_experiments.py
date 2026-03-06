"""
Run AMLP experiments on DMMOP benchmark with parallel execution.

Usage:
    python run_comparative_experiments.py                          # Run all F1-F24, seeds 1-30
    python run_comparative_experiments.py --functions 16-24        # Functions 16-24 only
    python run_comparative_experiments.py --seeds 1-10             # Seeds 1-10 only
    python run_comparative_experiments.py --seed 1 2 4 6           # Specific seeds
    python run_comparative_experiments.py --workers 8              # Use 8 parallel workers
    # python run_comparative_experiments.py --algorithm tsd        # TSD commented out for now
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import numpy as np
import pandas as pd
import json
from postprocess_result import postprocess_result
# from postprocess_tsd import postprocess_tsd  # TSD commented out


def parse_range(range_input):
    """Parse range input like '1-5', '1,3,5', or ['1', '3', '5']."""
    if isinstance(range_input, list):
        tokens = range_input
    else:
        tokens = [range_input]

    result = []
    for token in tokens:
        for part in str(token).split(','):
            part = part.strip()
            if not part:
                continue

            if '-' in part:
                start, end = map(int, part.split('-', 1))
                if start > end:
                    start, end = end, start
                result.extend(range(start, end + 1))
            else:
                result.append(int(part))
    return sorted(set(result))


def summarize_values(values):
    """Return a concise summary for ordered integer values."""
    if not values:
        return "[] (0)"
    if len(values) == 1:
        return f"{values[0]} (1)"

    is_consecutive = all((b - a) == 1 for a, b in zip(values, values[1:]))
    if is_consecutive:
        return f"{values[0]}-{values[-1]} ({len(values)})"
    return f"{values} ({len(values)})"


def run_single(function_num, seed):
    """Run a single AMLP experiment. Returns (function, seed, status, error)."""
    script_dir = Path(__file__).parent.resolve()
    entry = script_dir / 'main.py'

    # # TSD commented out
    # if algorithm == 'tsd':
    #     entry = script_dir / 'main_tsd.py'

    cmd = [
        sys.executable,
        str(entry),
        '--function', str(function_num),
        '--seed', str(seed),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(script_dir),
        )
        return function_num, seed, "OK", ""
    except subprocess.CalledProcessError as e:
        return function_num, seed, "FAIL", (e.stderr or str(e))[:300]


def postprocess_function(fn):
    """Post-process all AMLP results for a given function number."""
    result_dir = Path('result')
    if not result_dir.exists():
        return fn, None

    sys.path.insert(0, str(Path(__file__).parent / 'dmmops_py'))
    from dmmops_py.problem.get_info import get_info
    _, _, dim = get_info(fn)

    # AMLP pattern
    pattern = f'result_F{fn}_D{dim}_seed*.pkl'
    # # TSD pattern (commented out)
    # pattern = f'result_TSD_F{fn}_D{dim}_seed*.pkl'

    result_files = list(result_dir.glob(pattern))
    if not result_files:
        return fn, None

    mpr_values = []
    success_count = 0

    for result_file in result_files:
        try:
            mpr, _ = postprocess_result(str(result_file))
            # # TSD postprocessing (commented out)
            # mpr, _ = postprocess_tsd(str(result_file))
            if mpr is not None:
                mpr_values.append(mpr)
            success_count += 1
        except Exception as e:
            print(f"  Warning: Failed to process {result_file.name}: {e}")

    print(f"  F{fn}: processed {success_count}/{len(result_files)} files")

    if mpr_values:
        mpr_array = np.array(mpr_values)
        tolerances = [0.001, 0.0001, 0.00001]
        aggregated_stats = {
            'tolerance': tolerances,
            'mean_MPR': np.mean(mpr_array, axis=0).tolist(),
            'worst_MPR': np.min(mpr_array, axis=0).tolist(),
            'best_MPR': np.max(mpr_array, axis=0).tolist(),
            'std_MPR': np.std(mpr_array, axis=0).tolist(),
        }

        agg_df = pd.DataFrame(aggregated_stats)
        agg_file = result_dir / f'AMLP_F{fn}_D{dim}_aggregated.csv'
        agg_df.to_csv(agg_file, index=False, float_format='%.6f')

        print(f"  F{fn}: Mean MPR = {aggregated_stats['mean_MPR']}")
        print(f"         Saved to: {agg_file}")
        return fn, aggregated_stats
    return fn, None


def main():
    parser = argparse.ArgumentParser(description='Run AMLP experiments in parallel on DMMOP benchmark')
    # # TSD option commented out
    # parser.add_argument('--algorithm', type=str, default='amlp',
    #                     choices=['amlp', 'tsd', 'both'],
    #                     help='Which algorithm(s) to run')
    parser.add_argument('--functions', type=str, default='1-24',
                        help='Function range (e.g., "1-24", "16-24", "1,5,10")')
    parser.add_argument('--seeds', '--seed', dest='seeds', nargs='+', default=['1-30'],
                        help='Seed range/list (e.g., "1-30", "1,2,4,6", or "1 2 4 6")')
    parser.add_argument('--workers', type=int, default=5,
                        help='Number of parallel workers (default: 5)')
    parser.add_argument('--continue-on-error', action='store_true', default=True,
                        help='Continue running even if some experiments fail (default: True)')

    args = parser.parse_args()

    function_nums = parse_range(args.functions)
    seeds = parse_range(args.seeds)

    # Build job list
    jobs = []
    for fn in function_nums:
        for seed in seeds:
            jobs.append((fn, seed))

    total = len(jobs)

    print("=" * 70)
    print("AMLP EXPERIMENTS on DMMOP Benchmark (Parallel)")
    print("=" * 70)
    print(f"Algorithm: AMLP")
    # print(f"Algorithm: TSD  # commented out")
    print(f"Functions: {summarize_values(function_nums)}")
    print(f"Seeds: {summarize_values(seeds)}")
    print(f"Total experiments: {total}")
    print(f"Workers: {args.workers}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Ensure result directory exists
    Path('result').mkdir(exist_ok=True)

    completed = 0
    failed = 0
    failed_experiments = []
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_single, fn, seed): (fn, seed)
            for fn, seed in jobs
        }

        for future in as_completed(futures):
            fn, seed, status, err = future.result()
            completed += 1
            if status == "FAIL":
                failed += 1
                failed_experiments.append((fn, seed, err))
                print(f"  [{completed}/{total}] F{fn} seed={seed} FAILED: {err[:200]}")
            else:
                print(f"  [{completed}/{total}] F{fn} seed={seed} done")

    elapsed = time.time() - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)

    print("\n" + "=" * 70)
    print(f"Finished: {completed - failed}/{total} succeeded, {failed} failed")
    print(f"Time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print("=" * 70)

    # Post-process results for each function
    print("\nPost-processing results...")
    for fn in function_nums:
        postprocess_function(fn)

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY: AMLP CEC 2022 Results")
    print("=" * 70)

    result_dir = Path('result')
    sys.path.insert(0, str(Path(__file__).parent / 'dmmops_py'))
    from dmmops_py.problem.get_info import get_info

    for fn in function_nums:
        _, _, dim = get_info(fn)
        agg_file = result_dir / f'AMLP_F{fn}_D{dim}_aggregated.csv'
        if agg_file.exists():
            try:
                df = pd.read_csv(agg_file)
                mean_mpr = df['mean_MPR'].tolist()
                print(f"  F{fn:2d} (D{dim}): Mean MPR = {mean_mpr}")
            except Exception:
                print(f"  F{fn:2d}: see {agg_file}")
        else:
            print(f"  F{fn:2d}: no results yet")

    print("=" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if failed_experiments:
        print(f"\nFailed experiments ({failed}):")
        for fn, seed, err in failed_experiments[:20]:
            print(f"  F{fn} seed={seed}: {err[:100]}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()

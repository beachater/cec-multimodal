"""
Runner for TSD v4 experiments across F1-F24 with parallel execution.

Usage:
    python run_tsd_v4.py                          # Run all F1-F24, seeds 1-30
    python run_tsd_v4.py --functions 1 2 3 4      # Simple D5 only
    python run_tsd_v4.py --functions 5 6 7 8      # Composition D5 only
    python run_tsd_v4.py --functions 17 18 19 20  # Simple D10 only
    python run_tsd_v4.py --seeds 1 5              # Seeds 1 to 5
    python run_tsd_v4.py --workers 8              # Use 8 parallel workers
"""

import subprocess
import sys
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_single(function, seed, result_dir):
    """Run a single TSD v4 experiment."""
    cmd = [
        sys.executable, "main_tsd_v4.py",
        "--function", str(function),
        "--seed", str(seed),
        "--result-dir", result_dir,
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent),
    )
    status = "OK" if result.returncode == 0 else "FAIL"
    return function, seed, status, result.stderr if result.returncode != 0 else ""


def main():
    parser = argparse.ArgumentParser(description="Run TSD v4 experiments in parallel")
    parser.add_argument("--functions", type=int, nargs="+",
                        default=list(range(1, 25)),
                        help="Function numbers to run (default: 1-24)")
    parser.add_argument("--seeds", type=int, nargs=2, default=[1, 30], metavar=("START", "END"),
                        help="Seed range inclusive (default: 1 30)")
    parser.add_argument("--workers", type=int, default=5,
                        help="Number of parallel workers (default: 5)")
    parser.add_argument("--result-prefix", type=str, default="tsd_v4_result",
                        help="Result directory prefix (default: tsd_v4_result)")
    args = parser.parse_args()

    # Build job list
    jobs = []
    for func in args.functions:
        result_dir = f"{args.result_prefix}_function{func}"
        Path(result_dir).mkdir(exist_ok=True)
        for seed in range(args.seeds[0], args.seeds[1] + 1):
            jobs.append((func, seed, result_dir))

    total = len(jobs)
    n_seeds = args.seeds[1] - args.seeds[0] + 1
    print(f"Running {total} experiments ({len(args.functions)} functions x "
          f"{n_seeds} seeds) with {args.workers} workers")
    print(f"Result prefix: {args.result_prefix}_function{{N}}")
    print("=" * 60)

    completed = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single, f, s, d): (f, s) for f, s, d in jobs}

        for future in as_completed(futures):
            func, seed, status, err = future.result()
            completed += 1
            if status == "FAIL":
                failed += 1
                print(f"  [{completed}/{total}] F{func} seed={seed} FAILED: {err[:200]}")
            else:
                print(f"  [{completed}/{total}] F{func} seed={seed} done")

    print("=" * 60)
    print(f"Finished: {completed - failed}/{total} succeeded, {failed} failed")

    # Run postprocessing for each function
    print("\nPostprocessing results...")
    for func in args.functions:
        result_dir = f"{args.result_prefix}_function{func}"
        pkl_files = list(Path(result_dir).glob("*.pkl"))
        if pkl_files:
            print(f"\n--- F{func} ({len(pkl_files)} runs) ---")
            subprocess.run(
                [sys.executable, "postprocess_tsd.py", "--batch", result_dir],
                cwd=str(Path(__file__).parent),
            )

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY: CEC 2022 Results")
    print("=" * 60)
    for func in args.functions:
        result_dir = f"{args.result_prefix}_function{func}"
        csv_files = list(Path(result_dir).glob("TSD_*_CEC2022.csv"))
        if csv_files:
            try:
                import pandas as pd
                df = pd.read_csv(csv_files[0])
                pr_values = df['PR'].tolist()
                print(f"  F{func:2d}: PR = {pr_values}")
            except Exception:
                print(f"  F{func:2d}: see {csv_files[0]}")
        else:
            print(f"  F{func:2d}: no results yet")


if __name__ == "__main__":
    main()

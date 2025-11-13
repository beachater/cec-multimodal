"""
Quick comparison: Run AMLP vs TSD on same function/seed for side-by-side analysis.

Usage:
    python quick_compare.py --function 1 --seed 1
    python quick_compare.py --function 16 --seed 5 --maxeval 50000
"""

import argparse
import subprocess
import sys
from pathlib import Path
import time
import pickle
import pandas as pd
import numpy as np


def run_algorithm(alg, function_num, seed, maxeval=None):
    """Run one algorithm and return results."""
    if alg == 'amlp':
        cmd = [sys.executable, 'main.py', '--function', str(function_num), '--seed', str(seed)]
    else:  # tsd
        cmd = [sys.executable, 'main_tsd.py', '--function', str(function_num), '--seed', str(seed)]
    
    if maxeval:
        cmd.extend(['--maxeval', str(maxeval)])
    
    print(f"\nRunning {alg.upper()}...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 70)
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    
    if result.returncode != 0:
        print(f"ERROR: {alg.upper()} failed!")
        print(result.stderr)
        return None
    
    print(result.stdout)
    print(f"Time: {elapsed:.2f}s")
    
    return elapsed


def load_results(filename):
    """Load pickle results."""
    filepath = Path('result') / filename
    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return None
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def compare_results(amlp_data, tsd_data):
    """Compare AMLP and TSD results."""
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    comparison = []
    
    # Basic info
    comparison.append(("Algorithm", "AMLP", "TSD"))
    comparison.append(("", "", ""))
    
    # Best fitness (extract from archive for AMLP, directly from best_f for TSD)
    if amlp_data and 'archive' in amlp_data:
        archive = amlp_data['archive']
        amlp_best = np.min(archive.value) if archive.size > 0 else float('inf')
    else:
        amlp_best = float('inf')
    
    tsd_best = tsd_data.get('best_f', float('inf')) if tsd_data else float('inf')
    comparison.append(("Best Fitness", f"{amlp_best:.6e}", f"{tsd_best:.6e}"))
    
    # Archive size
    amlp_arch = amlp_data.get('archive').size if amlp_data and 'archive' in amlp_data else 0
    tsd_arch = tsd_data.get('archive').size if tsd_data and 'archive' in tsd_data else 0
    comparison.append(("Archive Size", str(amlp_arch), str(tsd_arch)))
    
    # Environments (AMLP uses dynamics.currentTimeStep, TSD uses archive_history)
    if amlp_data and 'process' in amlp_data:
        amlp_envs = amlp_data['process'].dynamics.currentTimeStep + 1 if hasattr(amlp_data['process'], 'dynamics') else 1
    else:
        amlp_envs = 0
    tsd_envs = len(tsd_data.get('archive_history', [])) if tsd_data else 0
    comparison.append(("Environments", str(amlp_envs), str(tsd_envs)))
    
    # Time (AMLP doesn't store elapsed_time)
    amlp_time = amlp_data.get('elapsed_time', 'N/A') if amlp_data else 'N/A'
    tsd_time = tsd_data.get('elapsed_time', 0) if tsd_data else 0
    if isinstance(amlp_time, float):
        comparison.append(("Time (s)", f"{amlp_time:.2f}", f"{tsd_time:.2f}"))
    else:
        comparison.append(("Time (s)", str(amlp_time), f"{tsd_time:.2f}"))
    
    # TSD-specific info
    if tsd_data and 'tsd_info' in tsd_data:
        info = tsd_data['tsd_info']
        comparison.append(("", "", ""))
        comparison.append(("TSD Details:", "", ""))
        comparison.append(("  Generations", "", str(info.get('generations_run', 'N/A'))))
        comparison.append(("  Substrate Norm", "", f"{info.get('substrate_norm', 0):.6e}"))
    
    # Print table
    for row in comparison:
        if len(row[0]) == 0:
            print()
        else:
            print(f"{row[0]:<25} {row[1]:<20} {row[2]:<20}")
    
    print("=" * 70)
    
    # Winner
    if amlp_best < tsd_best:
        print(f"🏆 AMLP wins (better fitness by {tsd_best - amlp_best:.6e})")
    elif tsd_best < amlp_best:
        print(f"🏆 TSD wins (better fitness by {amlp_best - tsd_best:.6e})")
    else:
        print("🤝 Tie (same fitness)")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Quick comparison: AMLP vs TSD')
    parser.add_argument('--function', type=int, default=1, help='DMMOP function (1-24)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--maxeval', type=int, default=None, help='Max evaluations (optional)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"QUICK COMPARISON: AMLP vs TSD on DMMOP{args.function}")
    print("=" * 70)
    print(f"Function: DMMOP{args.function}")
    print(f"Seed: {args.seed}")
    if args.maxeval:
        print(f"Max Evaluations: {args.maxeval}")
    print("=" * 70)
    
    # Run both algorithms
    amlp_time = run_algorithm('amlp', args.function, args.seed, args.maxeval)
    tsd_time = run_algorithm('tsd', args.function, args.seed, args.maxeval)
    
    # Load results
    from optim_problem_dmmop import OptimProblemDMMOP
    problem = OptimProblemDMMOP(args.function)
    dim = problem.dim
    
    amlp_file = f'result_F{args.function}_D{dim}_seed{args.seed}.pkl'
    tsd_file = f'result_TSD_F{args.function}_D{dim}_seed{args.seed}.pkl'
    
    print(f"\nLoading results...")
    amlp_data = load_results(amlp_file)
    tsd_data = load_results(tsd_file)
    
    # Compare
    compare_results(amlp_data, tsd_data)
    
    # Postprocess both
    print("\nGenerating CSV files...")
    for pkl_file in [amlp_file, tsd_file]:
        pkl_path = Path('result') / pkl_file
        if pkl_path.exists():
            cmd = [sys.executable, 'postprocess_result.py', str(pkl_path)]
            subprocess.run(cmd, capture_output=True)
            print(f"  ✓ {pkl_file} → CSV files generated")
    
    print("\n✓ Comparison complete! Check result/ directory for CSV files.")


if __name__ == '__main__':
    main()

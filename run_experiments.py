"""
Runner script for AMLP-RS-CMSA-ESII experiments on DMMOP benchmark.

This script automates:
1. Running experiments for selected function(s) with seeds 1-30
2. Post-processing results to generate CSV files

Written by Ali Ahrari (aliahrari1983@gmail.com)
Translated to Python from MATLAB
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import time


def run_experiment(function_num, seed):
    """
    Run a single experiment.
    
    Args:
        function_num: DMMOP function number (1-24)
        seed: Random seed (1-30)
    
    Returns:
        True if successful, False otherwise
    """
    cmd = [sys.executable, 'main.py', '--function', str(function_num), '--seed', str(seed)]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ ERROR in F{function_num} seed {seed}")
        print(f"     {e.stderr}")
        return False


def postprocess_results(function_num):
    """
    Post-process all results for a specific function to generate CSV.
    
    Args:
        function_num: DMMOP function number (1-24)
    
    Returns:
        True if successful, False otherwise
    """
    # Find all result files for this function
    result_dir = Path('result')
    if not result_dir.exists():
        print(f"  ⚠️  No result directory found")
        return False
    
    # Get dimension from get_info
    sys.path.insert(0, str(Path(__file__).parent / 'dmmops_py'))
    from dmmops_py.problem.get_info import get_info
    _, _, dim = get_info(function_num)
    
    # Find result files
    pattern = f'result_F{function_num}_D{dim}_seed*.pkl'
    result_files = list(result_dir.glob(pattern))
    
    if not result_files:
        print(f"  ⚠️  No result files found for function {function_num}")
        return False
    
    print(f"  Found {len(result_files)} result file(s)")
    
    # Process each result file to CSV
    success_count = 0
    for result_file in result_files:
        cmd = [sys.executable, 'postprocess_result.py', str(result_file)]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"  ❌ ERROR processing {result_file.name}: {e.stderr}")
    
    print(f"  ✓ Successfully processed {success_count}/{len(result_files)} files to CSV")
    return success_count > 0


def run_single_function(function_num, seeds=range(1, 31)):
    """
    Run all experiments for a single function.
    
    Args:
        function_num: DMMOP function number (1-24)
        seeds: List or range of seeds to run
    """
    print(f"\n{'='*70}")
    print(f"Running DMMOP Function {function_num}")
    print(f"{'='*70}")
    
    total_seeds = len(list(seeds))
    success_count = 0
    failed_seeds = []
    
    start_time = time.time()
    
    for i, seed in enumerate(seeds, 1):
        print(f"[{i}/{total_seeds}] Running F{function_num} with seed {seed}...", end=' ')
        sys.stdout.flush()
        
        if run_experiment(function_num, seed):
            print("✓")
            success_count += 1
        else:
            print("✗")
            failed_seeds.append(seed)
    
    elapsed = time.time() - start_time
    
    print(f"\nCompleted {success_count}/{total_seeds} runs in {elapsed:.1f}s")
    if failed_seeds:
        print(f"Failed seeds: {failed_seeds}")
    
    # Post-process results
    print(f"\nPost-processing results for Function {function_num}...")
    postprocess_results(function_num)
    
    return success_count, failed_seeds


def run_all_functions(seeds=range(1, 31)):
    """
    Run experiments for all 24 DMMOP functions.
    
    Args:
        seeds: List or range of seeds to run
    """
    print(f"\n{'='*70}")
    print(f"Running ALL 24 DMMOP Functions")
    print(f"{'='*70}")
    print(f"Seeds: {list(seeds)}")
    print(f"Total experiments: 24 functions × {len(list(seeds))} seeds = {24 * len(list(seeds))}")
    print(f"{'='*70}\n")
    
    overall_start = time.time()
    all_results = {}
    
    for func_num in range(1, 25):
        success, failed = run_single_function(func_num, seeds)
        all_results[func_num] = {'success': success, 'failed': failed}
    
    overall_elapsed = time.time() - overall_start
    
    # Summary
    print(f"\n{'='*70}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*70}")
    
    total_success = sum(r['success'] for r in all_results.values())
    total_runs = 24 * len(list(seeds))
    
    print(f"Total runs: {total_success}/{total_runs}")
    print(f"Total time: {overall_elapsed/60:.1f} minutes")
    
    # Show any failures
    failures = {f: r['failed'] for f, r in all_results.items() if r['failed']}
    if failures:
        print(f"\n⚠️  Functions with failures:")
        for func, failed_seeds in failures.items():
            print(f"  F{func}: seeds {failed_seeds}")
    else:
        print(f"\n✓ All experiments completed successfully!")
    
    print(f"{'='*70}\n")


def interactive_mode():
    """Interactive mode for user to choose what to run."""
    print("\n" + "="*70)
    print("AMLP-RS-CMSA-ESII Experiment Runner")
    print("="*70)
    print("\nOptions:")
    print("  1. Run a single function (seeds 1-30)")
    print("  2. Run all 24 functions (seeds 1-30)")
    print("  3. Run a single function with custom seed range")
    print("  4. Run all functions with custom seed range")
    print("  5. Post-process existing results only")
    print("  0. Exit")
    print("="*70)
    
    while True:
        try:
            choice = input("\nEnter your choice (0-5): ").strip()
            
            if choice == '0':
                print("Exiting...")
                sys.exit(0)
            
            elif choice == '1':
                func = int(input("Enter function number (1-24): "))
                if 1 <= func <= 24:
                    run_single_function(func)
                else:
                    print("❌ Invalid function number. Must be 1-24.")
            
            elif choice == '2':
                confirm = input("This will run 720 experiments (24 functions × 30 seeds). Continue? (y/n): ")
                if confirm.lower() == 'y':
                    run_all_functions()
                else:
                    print("Cancelled.")
            
            elif choice == '3':
                func = int(input("Enter function number (1-24): "))
                start_seed = int(input("Enter start seed: "))
                end_seed = int(input("Enter end seed: "))
                if 1 <= func <= 24 and start_seed <= end_seed:
                    run_single_function(func, range(start_seed, end_seed + 1))
                else:
                    print("❌ Invalid input.")
            
            elif choice == '4':
                start_seed = int(input("Enter start seed: "))
                end_seed = int(input("Enter end seed: "))
                confirm = input(f"This will run {24 * (end_seed - start_seed + 1)} experiments. Continue? (y/n): ")
                if confirm.lower() == 'y' and start_seed <= end_seed:
                    run_all_functions(range(start_seed, end_seed + 1))
                else:
                    print("Cancelled.")
            
            elif choice == '5':
                print("\nPost-processing options:")
                print("  1. Process a single function")
                print("  2. Process all functions")
                pp_choice = input("Enter choice (1-2): ").strip()
                
                if pp_choice == '1':
                    func = int(input("Enter function number (1-24): "))
                    if 1 <= func <= 24:
                        print(f"\nPost-processing Function {func}...")
                        postprocess_results(func)
                    else:
                        print("❌ Invalid function number.")
                
                elif pp_choice == '2':
                    print("\nPost-processing all functions...")
                    for func in range(1, 25):
                        print(f"\nFunction {func}:")
                        postprocess_results(func)
                    print("\n✓ All post-processing complete!")
            
            else:
                print("❌ Invalid choice. Please enter 0-5.")
        
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            sys.exit(0)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run AMLP-RS-CMSA-ESII experiments on DMMOP benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python run_experiments.py
  
  # Run single function with seeds 1-30
  python run_experiments.py --function 16
  
  # Run all 24 functions with seeds 1-30
  python run_experiments.py --all
  
  # Run specific function with custom seed range
  python run_experiments.py --function 16 --seeds 1 5
  
  # Post-process existing results only
  python run_experiments.py --postprocess-only --function 16
  python run_experiments.py --postprocess-only --all
        """
    )
    
    parser.add_argument('--function', type=int, help='Run specific function (1-24)')
    parser.add_argument('--all', action='store_true', help='Run all 24 functions')
    parser.add_argument('--seeds', nargs=2, type=int, metavar=('START', 'END'),
                        help='Seed range (default: 1 30)')
    parser.add_argument('--postprocess-only', action='store_true',
                        help='Only post-process existing results, do not run experiments')
    
    args = parser.parse_args()
    
    # Determine seed range
    if args.seeds:
        seed_range = range(args.seeds[0], args.seeds[1] + 1)
    else:
        seed_range = range(1, 31)  # Default: seeds 1-30
    
    # Post-process only mode
    if args.postprocess_only:
        if args.all:
            print("Post-processing all functions...")
            for func in range(1, 25):
                print(f"\nFunction {func}:")
                postprocess_results(func)
            print("\n✓ All post-processing complete!")
        elif args.function:
            if 1 <= args.function <= 24:
                print(f"Post-processing Function {args.function}...")
                postprocess_results(args.function)
            else:
                print("❌ Invalid function number. Must be 1-24.")
        else:
            print("❌ Please specify --function or --all with --postprocess-only")
        return
    
    # Run experiments
    if args.all:
        run_all_functions(seed_range)
    elif args.function:
        if 1 <= args.function <= 24:
            run_single_function(args.function, seed_range)
        else:
            print("❌ Invalid function number. Must be 1-24.")
    else:
        # No arguments provided, run interactive mode
        interactive_mode()


if __name__ == '__main__':
    main()

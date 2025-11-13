"""
Main entry point for TSD algorithm on DMMOP benchmark.

Example usage:
    python main_tsd.py --function 16 --seed 1
    python main_tsd.py --function 1 --seed 1 --response partial_reset
"""

import argparse
import sys
from pathlib import Path

# Add dmmops to path
sys.path.insert(0, str(Path(__file__).parent / 'dmmops_py'))

from optim_problem_dmmop import OptimProblemDMMOP
from tsd_driver import tsd_driver


def main():
    """Main function to run TSD on DMMOP benchmark."""
    parser = argparse.ArgumentParser(description='Run TSD algorithm on DMMOP problems')
    parser.add_argument('--function', type=int, default=16, help='DMMOP function number (1-24)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--maxeval', type=int, default=None, help='Max function evaluations (default: auto)')
    parser.add_argument('--output', type=str, default=None, help='Output filename (default: auto)')
    
    # TSD-specific parameters
    parser.add_argument('--N', type=int, default=60, help='Population size')
    parser.add_argument('--budget-per-tick', type=int, default=200, help='Evaluations per generation')
    parser.add_argument('--eta', type=float, default=0.25, help='Substrate learning rate')
    parser.add_argument('--lambda-s', type=float, default=0.5, help='Substrate shift strength')
    parser.add_argument('--rho', type=float, default=0.98, help='Substrate decay factor')
    parser.add_argument('--drift-interval', type=int, default=2000, help='Evals between substrate decay')
    parser.add_argument('--response', type=str, default='partial_reset',
                        choices=['full_reset', 'partial_reset', 'adapt'],
                        help='Response mode for environment changes')
    
    args = parser.parse_args()
    
    # Create problem
    print(f"Creating DMMOP{args.function} problem for TSD")
    problem = OptimProblemDMMOP(args.function)
    
    # Set max evaluations
    if args.maxeval is not None:
        problem.maxEval = args.maxeval
    else:
        problem.maxEval = problem.extProb.evaluation  # = 5000 * D * 60
    
    # Set output filename
    if args.output is None:
        output_filename = f'result_TSD_F{args.function}_D{problem.dim}_seed{args.seed}.pkl'
    else:
        output_filename = args.output
    
    print(f"Dimension: {problem.dim} (from DMMOP specification)")
    print(f"Starting TSD optimization with seed {args.seed}")
    print(f"Max evaluations: {problem.maxEval}")
    print(f"Response mode: {args.response}")
    print(f"Results will be saved to: {output_filename}")
    print("=" * 60)
    
    # Prepare TSD parameters
    tsd_params = {
        'N': args.N,
        'budget_per_tick': args.budget_per_tick,
        'eta': args.eta,
        'lambda_s': args.lambda_s,
        'rho': args.rho,
        'drift_interval': args.drift_interval,
        'response_mode': args.response,
    }
    
    # Run TSD
    tsd_driver(problem, args.seed, output_filename, **tsd_params)
    
    print("=" * 60)
    print("TSD optimization completed!")


if __name__ == '__main__':
    main()

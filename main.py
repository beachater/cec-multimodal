"""
Main entry point for AMLP-RS-CMSA-ESII algorithm.

Written by Ali Ahrari (aliahrari1983@gmail.com)
Translated to Python from MATLAB

Example usage:
    python main.py --function 16 --seed 1
"""

import numpy as np
import argparse
import sys
from pathlib import Path

# Add dmmops to path
sys.path.insert(0, str(Path(__file__).parent / 'dmmops_py'))

from optim_option import OptimOption
from optim_problem_dmmop import OptimProblemDMMOP
from driver import driver


def main():
    """Main function to run AMLP-RS-CMSA-ESII on DMMOP benchmark."""
    parser = argparse.ArgumentParser(description='Run AMLP-RS-CMSA-ESII algorithm on DMMOP problems')
    parser.add_argument('--function', type=int, default=16, help='DMMOP function number (1-24)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--maxeval', type=int, default=None, help='Max function evaluations (default: auto)')
    parser.add_argument('--output', type=str, default=None, help='Output filename (default: auto)')
    
    args = parser.parse_args()
    
    # Create problem (dimension is determined by get_info based on function number)
    print(f"Creating DMMOP{args.function} problem")
    problem = OptimProblemDMMOP(args.function)
    
    # Set max evaluations if specified
    if args.maxeval is not None:
        problem.maxEval = args.maxeval
    else:
        # Default: 60 environments × 5000*D evals per environment (per DMMOP spec)
        # DMMOP already sets this correctly in extProb.evaluation
        problem.maxEval = problem.extProb.evaluation  # = 5000 * D * 60
    
    # Create optimization options
    opt = OptimOption(problem)
    
    # Set output filename
    if args.output is None:
        output_filename = f'result_F{args.function}_D{problem.dim}_seed{args.seed}.pkl'
    else:
        output_filename = args.output
    
    print(f"Dimension: {problem.dim} (from DMMOP specification)")
    print(f"Starting optimization with seed {args.seed}")
    print(f"Max evaluations: {problem.maxEval}")
    print(f"Results will be saved to: {output_filename}")
    print("-" * 60)
    
    # Run the algorithm
    driver(opt, problem, args.seed, output_filename)
    
    print("-" * 60)
    print("Optimization completed!")
    print(f"Results saved to: result/{output_filename}")


if __name__ == '__main__':
    main()

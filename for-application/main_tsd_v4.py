"""
Main entry point for TSD v4 (function-aware) on DMMOP benchmark.

v4 automatically selects parameters based on function type:
- Simple functions (F1-F4 base): tuned for 4-peak landscape
- Composition functions (F5-F8 base): tuned for 6-8 peak landscape
- D=10 problems: larger population and more refinement budget

Usage:
    python main_tsd_v4.py --function 1 --seed 1
    python main_tsd_v4.py --function 8 --seed 1     # Auto-detects composition
    python main_tsd_v4.py --function 17 --seed 1     # Auto-detects D10 simple
    python main_tsd_v4.py --function 21 --seed 1     # D10 composition
"""

import argparse
import sys
from pathlib import Path

# Add dmmops to path
sys.path.insert(0, str(Path(__file__).parent / 'dmmops_py'))

from optim_problem_dmmop import OptimProblemDMMOP
from tsd_driver_v4 import tsd_driver_v4, get_function_preset


def main():
    """Main function to run TSD v4 on DMMOP benchmark."""
    parser = argparse.ArgumentParser(description='Run TSD v4 (function-aware) on DMMOP')
    parser.add_argument('--function', type=int, default=1, help='DMMOP function number (1-24)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--result-dir', type=str, default='result', help='Output directory')

    # Optional overrides (normally auto-detected by preset)
    parser.add_argument('--N', type=int, default=None, help='Override population size')
    parser.add_argument('--eta', type=float, default=0.25, help='Substrate learning rate')
    parser.add_argument('--lambda-s', type=float, default=0.5, help='Substrate shift strength')
    parser.add_argument('--rho', type=float, default=0.98, help='Substrate decay factor')
    parser.add_argument('--drift-interval', type=int, default=2000, help='Evals between substrate decay')
    parser.add_argument('--response', type=str, default='partial_reset',
                        choices=['full_reset', 'partial_reset', 'adapt'],
                        help='Response mode for environment changes')

    args = parser.parse_args()

    # Create problem
    print(f"Creating DMMOP{args.function} problem for TSD v4")
    problem = OptimProblemDMMOP(args.function)
    problem.maxEval = problem.extProb.evaluation  # = 5000 * D * 60

    # Show auto-detected preset
    preset = get_function_preset(args.function, problem.dim)
    func_type = "composition" if preset.is_composition else "simple"
    print(f"Dimension: {problem.dim}")
    print(f"Auto-detected: base=F{preset.base_func}, type={func_type}, "
          f"peaks={preset.n_peaks}, optimal={preset.optimal_value}")
    print(f"Preset: N={preset.N}, niche_r={preset.clearing_niche_radius}, "
          f"archive_dist={preset.archive_dist_threshold}, "
          f"refine_evals={preset.refine_max_evals}")
    print(f"Max evaluations: {problem.maxEval}")
    print(f"Response mode: {args.response}")
    print("=" * 60)

    output_filename = f'result_TSD_v4_F{args.function}_D{problem.dim}_seed{args.seed}.pkl'

    # Build kwargs (only pass overrides)
    tsd_kwargs = {
        'eta': args.eta,
        'lambda_s': args.lambda_s,
        'rho': args.rho,
        'drift_interval': args.drift_interval,
        'response_mode': args.response,
    }
    if args.N is not None:
        tsd_kwargs['N'] = args.N

    # Run
    tsd_driver_v4(problem, args.seed, output_filename,
                  result_dir=args.result_dir, **tsd_kwargs)

    print("=" * 60)
    print("TSD v4 optimization completed!")


if __name__ == '__main__':
    main()

"""
Test TSD integration with DMMOP benchmark.

Quick validation that TSD driver works correctly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'dmmops_py'))

from optim_problem_dmmop import OptimProblemDMMOP
from tsd_driver import tsd_driver
import numpy as np


def test_tsd_basic():
    """Test TSD on a simple DMMOP function."""
    print("=" * 70)
    print("TEST: TSD Integration with DMMOP")
    print("=" * 70)
    
    # Create problem (F1 is simplest)
    print("\nCreating DMMOP1 problem...")
    problem = OptimProblemDMMOP(1)
    
    # Reduce evaluations for quick test
    problem.maxEval = 10000  # Instead of full 5000*D*60
    
    print(f"Problem: DMMOP1")
    print(f"Dimension: {problem.dim}")
    print(f"Max evaluations (reduced for test): {problem.maxEval}")
    
    # Run TSD with small population
    print("\nRunning TSD driver...")
    try:
        tsd_driver(
            problem,
            seedNo=1,
            filename='test_tsd_integration.pkl',
            N=20,  # Small population for quick test
            budget_per_tick=100,
            response_mode='partial_reset'
        )
        print("\n✓ TSD driver completed successfully!")
        
        # Check if output file exists
        output_path = Path('result') / 'test_tsd_integration.pkl'
        if output_path.exists():
            print(f"✓ Output file created: {output_path}")
            
            # Load and verify
            import pickle
            with open(output_path, 'rb') as f:
                results = pickle.load(f)
            
            print(f"\nResults structure:")
            print(f"  - Algorithm: {results.get('algorithm', 'N/A')}")
            print(f"  - Seed: {results.get('seed', 'N/A')}")
            print(f"  - Best fitness: {results.get('best_f', 'N/A'):.6e}")
            print(f"  - Archive size: {results.get('archive').size if 'archive' in results else 'N/A'}")
            print(f"  - Environments: {len(results.get('archive_history', []))}")
            
            print("\n✓ All checks passed!")
            return True
        else:
            print(f"✗ Output file not found: {output_path}")
            return False
            
    except Exception as e:
        print(f"\n✗ TSD driver failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_tsd_basic()
    sys.exit(0 if success else 1)

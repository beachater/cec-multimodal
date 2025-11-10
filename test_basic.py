"""
Basic tests to verify the Python implementation works correctly.

This script tests:
1. DMMOP problem initialization
2. OptimOption creation
3. Basic algorithm components
4. Small run to verify integration
"""

import numpy as np
import sys
from pathlib import Path

# Add dmmops to path
sys.path.insert(0, str(Path(__file__).parent / 'dmmops_py'))

print("=" * 60)
print("AMLP-RS-CMSA-ESII Python Implementation Tests")
print("=" * 60)

# Test 1: Import all modules
print("\n[Test 1] Importing modules...")
try:
    from optim_option import OptimOption
    from optim_problem_dmmop import OptimProblemDMMOP
    from archive import Archive
    from optim_process import OptimProcess
    from restart import Restart
    from data_structures import *
    from mutation_sampling import *
    from core_search import *
    from utility_methods import UtilityMethods
    from prediction import Prediction
    from dynamic_manager import DynamicManager
    from subpopulation import Subpopulation
    from subpopulation_cmsa import SubpopulationCMSA
    from driver import driver
    print("[OK] All modules imported successfully")
except Exception as e:
    print(f"[FAIL] Import failed: {e}")
    sys.exit(1)

# Test 2: Create DMMOP problem
print("\n[Test 2] Creating DMMOP problem...")
try:
    problem = OptimProblemDMMOP(function_number=16, dim=5)
    print(f"[OK] DMMOP16 created with dimension {problem.dim}")
    print(f"  - Search range: [{problem.lowBound[0]}, {problem.upBound[0]}]")
    print(f"  - Max evaluations: {problem.maxEval}")
    print(f"  - Is dynamic: {problem.isDynamic}")
    print(f"  - Suite: {problem.suite}")
except Exception as e:
    print(f"[FAIL] Problem creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Create optimization options
print("\n[Test 3] Creating optimization options...")
try:
    opt = OptimOption(problem)
    print(f"[OK] OptimOption created")
    print(f"  - Core search: {opt.coreSearch.algorithm}")
    print(f"  - Initial subpop size coeff: {opt.coreSearch.iniSubpopSizeCoeff}")
    print(f"  - Initial normalized taboo distance: {opt.archiving.iniNormTabDis}")
    print(f"  - Max prediction level: {opt.dyna.maxPL}")
except Exception as e:
    print(f"[FAIL] Option creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Initialize archive and process
print("\n[Test 4] Initializing archive and process...")
try:
    np.random.seed(42)
    archive = Archive(problem)
    process = OptimProcess(opt, problem)
    print(f"[OK] Archive and process initialized")
    print(f"  - Archive size: {archive.size}")
    print(f"  - Process subpop size: {process.subpopSize}")
    print(f"  - Process mu: {process.mu}")
    print(f"  - Process muEff: {process.muEff:.4f}")
except Exception as e:
    print(f"[FAIL] Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Initialize restart and subpopulation
print("\n[Test 5] Creating restart and subpopulation...")
try:
    restart = Restart(process, opt, problem)
    subpop = restart.initialize_subpop(archive, process, opt, problem)
    print(f"[OK] Restart and subpopulation created")
    print(f"  - Restart stag size: {restart.stagSize}")
    print(f"  - Restart tolHist size: {restart.tolHistSize}")
    print(f"  - Subpop center: {subpop.center[:3]}... (showing first 3 dims)")
    print(f"  - Subpop mutation smean: {subpop.mutProfile.smean:.6f}")
    print(f"  - Subpop best value: {subpop.bestVal}")
except Exception as e:
    print(f"[FAIL] Restart/subpop creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test utility methods
print("\n[Test 6] Testing utility methods...")
try:
    test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    p2p = UtilityMethods.peak2peak(test_data)
    lin_p = UtilityMethods.lin_prctile(test_data, 0.5)
    print(f"[OK] Utility methods working")
    print(f"  - peak2peak([1,2,3,4,5]): {p2p}")
    print(f"  - lin_prctile([1,2,3,4,5], 0.5): {lin_p}")
except Exception as e:
    print(f"[FAIL] Utility methods failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Test prediction
print("\n[Test 7] Testing AMLP prediction...")
try:
    # Create simple trajectory data
    Xhist = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0]
    ])
    maxPL = 2
    xhatNow, estPreErrNowNorm, bestLevel = Prediction.AMLP(Xhist, maxPL)
    print(f"[OK] AMLP prediction working")
    print(f"  - Best level: {bestLevel}")
    print(f"  - Predicted position (best): {xhatNow[bestLevel-1]}")
    print(f"  - Prediction error (best): {estPreErrNowNorm[bestLevel-1]:.6f}")
except Exception as e:
    print(f"[FAIL] Prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Small integration test - one iteration
print("\n[Test 8] Running one evolution iteration...")
try:
    # Sample solutions
    subpop.sample_solutions(restart, archive, process, opt, problem)
    print(f"[OK] Solutions sampled: {subpop.samples.X.shape}")
    
    # Evaluate solutions
    old_numCallF = problem.numCallF
    subpop.eval_solutions(restart, archive, process, opt, problem)
    new_evals = problem.numCallF - old_numCallF
    print(f"[OK] Solutions evaluated: {new_evals} function calls")
    print(f"  - Best value: {np.min(subpop.samples.f):.6f}")
    print(f"  - Median value: {np.median(subpop.samples.f):.6f}")
    
    # Select
    subpop.select(restart, archive, process, opt, problem)
    print(f"[OK] Selection performed")
    
    # Recombine
    subpop.recombine(restart, archive, process, opt, problem)
    print(f"[OK] Recombination performed")
    print(f"  - New center: {subpop.center[:3]}... (showing first 3 dims)")
    print(f"  - New smean: {subpop.mutProfile.smean:.6f}")
    print(f"  - Best solution value: {subpop.bestVal:.6f}")
    
except Exception as e:
    print(f"[FAIL] Evolution iteration failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 9: Test archive operations
print("\n[Test 9] Testing archive operations...")
try:
    if subpop.bestVal < np.inf:
        # Simulate adding a solution to archive
        old_size = archive.size
        archive.append(subpop.bestSol, subpop.bestVal, restart, process, opt, problem)
        print(f"[OK] Solution appended to archive")
        print(f"  - Archive size: {old_size} -> {archive.size}")
        print(f"  - Archived value: {archive.value[-1]:.6f}")
except Exception as e:
    print(f"[FAIL] Archive operation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print("[OK] All basic tests passed!")
print(f"Total function evaluations used: {problem.numCallF}")
print("\nThe implementation is ready to run full experiments.")
print("\nNext steps:")
print("  1. Run a full experiment: python main.py --function 16 --dim 5 --seed 1 --maxeval 50000")
print("  2. Process results: python postprocess_result.py result/result_F16_D5_seed1.pkl")
print("=" * 60)

"""
Analyze and compare results from AMLP and TSD pickle files.
Usage: python analyze_comparison.py result_F1_D5_seed42.pkl result_TSD_F1_D5_seed42.pkl
"""

import sys
import pickle
import numpy as np
from pathlib import Path


def load_result(filepath):
    """Load pickle result file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def analyze_algorithm(data, name):
    """Analyze results from one algorithm."""
    print(f"\n{'='*70}")
    print(f"{name} Analysis")
    print(f"{'='*70}")
    
    if name == "AMLP":
        # AMLP format
        archive = data.get('archive')
        process = data.get('process')
        problem = data.get('problem')
        
        if archive and archive.size > 0:
            best_val = np.min(archive.value)
            worst_val = np.max(archive.value)
            mean_val = np.mean(archive.value)
            best_idx = np.argmin(archive.value)
            best_solution = archive.solution[best_idx, :]
            
            print(f"Archive Size: {archive.size}")
            print(f"Best Value: {best_val:.6e}")
            print(f"Worst Value: {worst_val:.6e}")
            print(f"Mean Value: {mean_val:.6e}")
            print(f"Best Solution: {best_solution}")
            
            if hasattr(archive, 'foundEval'):
                print(f"Found at Evaluation: {archive.foundEval[best_idx]}")
        
        if process:
            if hasattr(process, 'dynamics'):
                env = process.dynamics.currentTimeStep + 1
                print(f"Environments Processed: {env}")
            print(f"Restarts: {process.restartNo}")
            
        if problem:
            print(f"Total Evaluations: {problem.numCallF}")
            
    else:  # TSD
        archive = data.get('archive')
        tsd_info = data.get('tsd_info', {})
        best_f = data.get('best_f')
        best_x = data.get('best_x')
        
        if archive and archive.size > 0:
            best_val = np.min(archive.value)
            worst_val = np.max(archive.value)
            mean_val = np.mean(archive.value)
            
            print(f"Archive Size: {archive.size}")
            print(f"Best Value (archive): {best_val:.6e}")
            print(f"Best Value (tracked): {best_f:.6e}")
            print(f"Worst Value: {worst_val:.6e}")
            print(f"Mean Value: {mean_val:.6e}")
            print(f"Best Solution: {best_x}")
        
        print(f"Generations: {tsd_info.get('generations_run', 'N/A')}")
        print(f"Evaluations Used: {tsd_info.get('evals_used', 'N/A')}")
        print(f"Substrate Norm: {tsd_info.get('substrate_norm', 'N/A'):.6e}")
        
        archive_hist = data.get('archive_history', [])
        print(f"Environments: {len(archive_hist)}")
    
    return data


def compare_algorithms(amlp_data, tsd_data):
    """Side-by-side comparison."""
    print(f"\n{'='*70}")
    print("SIDE-BY-SIDE COMPARISON")
    print(f"{'='*70}")
    
    # Extract best values
    amlp_archive = amlp_data.get('archive')
    tsd_archive = tsd_data.get('archive')
    
    amlp_best = np.min(amlp_archive.value) if amlp_archive and amlp_archive.size > 0 else float('inf')
    tsd_best = tsd_data.get('best_f', float('inf'))
    
    print(f"{'Metric':<30} {'AMLP':<20} {'TSD':<20}")
    print("-" * 70)
    print(f"{'Best Fitness':<30} {amlp_best:<20.6e} {tsd_best:<20.6e}")
    
    # Archive size
    amlp_size = amlp_archive.size if amlp_archive else 0
    tsd_size = tsd_archive.size if tsd_archive else 0
    print(f"{'Archive Size':<30} {amlp_size:<20} {tsd_size:<20}")
    
    # Evaluations
    amlp_evals = amlp_data.get('problem').numCallF if 'problem' in amlp_data else 'N/A'
    tsd_evals = tsd_data.get('tsd_info', {}).get('evals_used', 'N/A')
    print(f"{'Total Evaluations':<30} {amlp_evals:<20} {tsd_evals:<20}")
    
    # Performance comparison
    print("\n" + "="*70)
    if amlp_best < tsd_best:
        diff = tsd_best - amlp_best
        pct = (diff / abs(amlp_best)) * 100 if amlp_best != 0 else 0
        print(f"🏆 AMLP WINS by {diff:.6e} ({pct:.2f}% better)")
    elif tsd_best < amlp_best:
        diff = amlp_best - tsd_best
        pct = (diff / abs(tsd_best)) * 100 if tsd_best != 0 else 0
        print(f"🏆 TSD WINS by {diff:.6e} ({pct:.2f}% better)")
    else:
        print("🤝 TIE - Both achieved same fitness")
    print("="*70)


def main():
    if len(sys.argv) != 3:
        print("Usage: python analyze_comparison.py <amlp_pkl> <tsd_pkl>")
        print("Example: python analyze_comparison.py result_F1_D5_seed42.pkl result_TSD_F1_D5_seed42.pkl")
        sys.exit(1)
    
    amlp_file = Path('result') / sys.argv[1] if not Path(sys.argv[1]).exists() else Path(sys.argv[1])
    tsd_file = Path('result') / sys.argv[2] if not Path(sys.argv[2]).exists() else Path(sys.argv[2])
    
    if not amlp_file.exists():
        print(f"Error: AMLP file not found: {amlp_file}")
        sys.exit(1)
    
    if not tsd_file.exists():
        print(f"Error: TSD file not found: {tsd_file}")
        sys.exit(1)
    
    print("="*70)
    print("FULL-BUDGET COMPARISON: AMLP vs TSD")
    print("="*70)
    print(f"AMLP file: {amlp_file}")
    print(f"TSD file: {tsd_file}")
    
    # Load data
    amlp_data = load_result(amlp_file)
    tsd_data = load_result(tsd_file)
    
    # Analyze each
    analyze_algorithm(amlp_data, "AMLP")
    analyze_algorithm(tsd_data, "TSD")
    
    # Compare
    compare_algorithms(amlp_data, tsd_data)


if __name__ == '__main__':
    main()

# System Ready for CEC 2022 Experiments ✅

## Pre-Flight Checks Completed

### ✅ 1. All Components Error-Free
- `run_comparative_experiments.py` - No errors
- `postprocess_result.py` - No errors  
- `postprocess_tsd.py` - No errors
- `main.py` (AMLP) - No errors
- `main_tsd.py` (TSD) - No errors
- `tsd.py` - No errors
- `tsd_driver.py` - No errors

### ✅ 2. CEC 2022 DMMOP Configuration
- **All 24 functions verified**
- Frequency: 5000 × D per environment
- Environments: 60
- Total evaluations: 5000 × D × 60
- Seeds: 1-30 (fixed)

### ✅ 3. Algorithm Compliance

**AMLP-RS-CMSA-ESII:**
- Uses full evaluation budget (1.5M for D=5, 3M for D=10)
- Generates per-environment archives
- Calculates MPR at 3 tolerance levels

**TSD (Temporal Substrate Drift):**
- `max_gens` dynamically calculated (10,000 for D=5, 16,000 for D=10)
- Uses full evaluation budget (not generation-limited)
- Saves per-environment archives
- Calculates MPR at same tolerances as AMLP

### ✅ 4. Output Format (CEC 2022 Compliant)

**Per-Seed Output:**
- `result_F{fn}_D{dim}_seed{seed}_summary.csv` - MPR per tolerance
- `result_F{fn}_D{dim}_seed{seed}_detail.csv` - PR per environment
- `result_F{fn}_D{dim}_seed{seed}_archive.csv` - Archive solutions

**Per-Function Aggregated (NEW):**
- `{ALG}_F{fn}_D{dim}_aggregated.csv` with columns:
  - tolerance: [0.001, 0.0001, 0.00001]
  - mean_MPR: Average across 30 runs
  - worst_MPR: Minimum across 30 runs
  - best_MPR: Maximum across 30 runs
  - std_MPR: Standard deviation

### ✅ 5. Key Features

**Crash Recovery:**
- Checkpoint system saves progress after each run
- Resume from brownout/interruption with: `--continue-on-error`

**Progress Tracking:**
- ETA calculation based on average experiment time
- Real-time progress updates

**Execution Order:**
- TSD runs first, then AMLP (per user request)

**Suppressed Output:**
- No verbose AMLP timestep messages during batch runs

## Fixed Issues

1. ✅ **phi.mat missing** - Copied from MATLAB dmmops to Python dmmops_py
2. ✅ **TSD max_gens limit** - Now dynamically calculated based on budget
3. ✅ **TSD MPR calculation** - Added Peak Ratio metrics matching AMLP
4. ✅ **Aggregation missing** - Added per-function statistics (mean/worst/best)

## Usage

```bash
# Run all experiments (1440 total: 24 functions × 30 seeds × 2 algorithms)
python run_comparative_experiments.py

# Run specific subset
python run_comparative_experiments.py --functions 1-5 --seeds 1-10

# Run only one algorithm
python run_comparative_experiments.py --algorithm tsd
python run_comparative_experiments.py --algorithm amlp

# Continue after interruption
python run_comparative_experiments.py --continue-on-error
```

## Expected Runtime

- **Per experiment**: ~30-60 minutes (varies by function/dimension)
- **Total (1440 experiments)**: ~720-1440 hours (30-60 days)
- **Recommended**: Run in batches or on HPC cluster

## Output Location

All results saved to `result/` directory:
- Individual seed PKL files
- Individual seed CSV files  
- Aggregated CSV files per function per algorithm

---
**Status**: ✅ READY FOR PRODUCTION
**Date**: November 13, 2025
**Compliance**: CEC 2022 Competition Specifications

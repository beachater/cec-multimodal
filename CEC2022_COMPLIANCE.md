# CEC 2022 Competition Compliance Verification

## DMMOP Benchmark Settings

According to CEC 2022 specifications, the following settings are required:

### 1. ✅ Runs: 30
- **Specification**: The random seeds of all runs should be fixed from 1 to 30.
- **Implementation**: 
  - `run_comparative_experiments.py` default: `--seeds 1-30`
  - Both AMLP and TSD accept `--seed` parameter
  - Seeds are fixed and reproducible

### 2. ✅ Frequency: 5000 × D evaluations per environment
- **Specification**: Each environment contains 5000 × D fitness evaluations.
- **Implementation**:
  - DMMOP initialization: `self.freq = 5000 * D`
  - For D=5: freq = 25,000
  - For D=10: freq = 50,000
  - Verified in `dmmops_py/problem/dmmop.py` line 79

### 3. ✅ Number of Environments: 60
- **Specification**: 60 environments total
- **Implementation**:
  - DMMOP initialization: `self.maxEnv = 60`
  - Verified in `dmmops_py/problem/dmmop.py` line 80

### 4. ✅ Maximum Fitness Evaluations: 5000 × D × 60
- **Specification**: Sum of fitness evaluations in all environments = 5000 × D × 60
- **Implementation**:
  - DMMOP calculation: `self.evaluation = self.maxEnv * self.freq`
  - For D=5: 1,500,000 evaluations
  - For D=10: 3,000,000 evaluations
  - Both algorithms use: `problem.maxEval = problem.extProb.evaluation`
  - Verified in `dmmops_py/problem/dmmop.py` line 81

### 5. ✅ Environmental Change Condition
- **Specification**: Environment changes when there are no more fitness evaluations for current environment
- **Implementation**:
  - AMLP: `CheckChange()` method in DMMOP tracks evaluation count
  - TSD: Same `CheckChange()` wrapper in `tsd_driver.py`
  - Change triggered at multiples of `freq`

### 6. ✅ Termination Condition
- **Specification**: Algorithm terminates when all fitness evaluations are consumed
- **Implementation**:
  - AMLP: Stops when `problem.numCallF >= problem.maxEval`
  - TSD: Stops when `self.evals >= self.max_evals`
  - Both respect the total budget exactly

## Algorithm-Specific Compliance

### AMLP-RS-CMSA-ESII
- Uses DMMOP's maxEval directly
- Runs until budget exhausted
- Generates per-environment archives
- Calculates MPR at tolerances: [0.001, 0.0001, 0.00001]

### TSD (Temporal Substrate Drift)
- **FIXED**: `max_gens` now calculated dynamically based on budget
- Uses same DMMOP maxEval
- Runs until budget exhausted (not generation-limited)
- Saves per-environment archives via `archive_history`
- Calculates MPR at same tolerances as AMLP

## Test Functions

All 24 DMMOP functions are supported:
- **F1-F8**: Base functions with D=5
- **F9-F16**: Various change types with D=5
- **F17-F24**: Same as F1-F8 but with D=10

## Output Format

Both algorithms generate comparable CSV outputs:
- `*_summary.csv`: MPR values per tolerance level
- `*_detail.csv`: PR per environment per tolerance
- `*_archive.csv`: Archive solutions found

## Verification Command

```bash
python -c "from dmmops_py.problem.dmmop import DMMOP; p = DMMOP(1); \
print(f'Freq: {p.freq}, MaxEnv: {p.maxEnv}, Total: {p.evaluation}')"
```

Expected output: `Freq: 25000, MaxEnv: 60, Total: 1500000`

---
**Status**: ✅ Fully compliant with CEC 2022 Competition specifications
**Last Updated**: November 13, 2025

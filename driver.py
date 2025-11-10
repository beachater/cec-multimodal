"""
Driver function for solving multimodal optimization problems.

Written by Ali Ahrari (aliahrari1983@gmail.com)
Translated to Python from MATLAB
"""

import numpy as np
import time
import pickle
from pathlib import Path
from optim_process import OptimProcess
from archive import Archive
from restart import Restart


def driver(opt, problem, seedNo, filename):
    """
    Solve the multimodal optimization problem given the problem and optimization options.
    
    Args:
        opt: Optimization options
        problem: Problem instance
        seedNo: Random seed
        filename: Output filename for saving results
    """
    start_time = time.time()
    np.random.seed(seedNo)
    
    process = OptimProcess(opt, problem)
    archive = Archive(problem)
    
    while True:
        restart = Restart(process, opt, problem)
        subpop = restart.initialize_subpop(archive, process, opt, problem)
        
        # Use predicted centers/step sizes for dynamic problems
        if problem.isDynamic and process.dynamics.recInd <= len(process.dynamics.recStepSize):
            subpop.center = process.dynamics.recCenter[process.dynamics.recInd - 1, :]
            subpop.center = np.minimum(np.maximum(subpop.center, problem.lowBound), problem.upBound)
            subpop.mutProfile.smean = process.dynamics.recStepSize[process.dynamics.recInd - 1]
            process.dynamics.recInd += 1
        
        restart.run_one_restart(subpop, archive, process, opt, problem)
        
        if restart.terminationFlag == -10:  # Evaluation budget finished
            print('inside driver')
            print(f'[{problem.numCallF}]')
            process.dynamics.update(archive, opt, problem)
            archive.usedEvalHist = np.append(archive.usedEvalHist, 0)
            process.update_due_to_change(restart, archive, opt, problem)
            break
        elif restart.terminationFlag == -5:  # Problem changed
            process.dynamics.update(archive, opt, problem)
            archive.usedEvalHist = np.append(archive.usedEvalHist, 0)
            process.update_due_to_change(restart, archive, opt, problem)
            process.reset_static(opt, problem)
            archive = Archive(problem)  # Create new archive
        else:
            archive.update(restart, process, opt, problem)
            process.update(restart, archive, opt, problem)
            print(f'timestep={process.dynamics.currentTimeStep}, restartNo={process.restartNo - 1}, '
                  f'usedEval={process.usedEvalTillRestart}={problem.numCallF} '
                  f'({problem.numCallF / problem.maxEval * 100:.2f}%), '
                  f'archiveSize={archive.size}, bestVal={restart.bestVal}')
    
    # Save results
    result_dir = Path('result')
    result_dir.mkdir(exist_ok=True)
    
    if problem.suite == 'GMPB':
        offErr = problem.extProb.CurrentError
        with open(filename, 'wb') as f:
            pickle.dump({'offErr': offErr}, f)
    else:
        with open(filename, 'wb') as f:
            pickle.dump({
                'problem': problem,
                'process': process,
                'archive': archive
            }, f)
    
    # Move file to result directory
    Path(filename).replace(result_dir / Path(filename).name)

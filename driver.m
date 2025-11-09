function []=driver(opt,problem,seedNo,filename)
    tic
    rng(seedNo)
    % Solve the multimodal optimization problem given the problem and optimization options  
    process=OptimProcess(opt,problem); % create the object process
    archive=Archive(problem); % create the object archive
    while 1
        restart=Restart(process,opt,problem); % create the object restart (restart-specific information)
        subpop=restart.initialize_subpop(archive,process,opt,problem); % initialize the subpopulation
        if problem.isDynamic && process.dynamics.recInd<=numel(process.dynamics.recStepSize)
            subpop.center=process.dynamics.recCenter(process.dynamics.recInd,:);
            subpop.center=min([max([subpop.center;problem.lowBound]);problem.upBound]);
            subpop.mutProfile.smean=process.dynamics.recStepSize(process.dynamics.recInd);
            process.dynamics.recInd=process.dynamics.recInd+1;
        end
        restart.run_one_restart(subpop,archive,process,opt,problem); % evolve the subpopulation until convergence 
        if (restart.terminationFlag==-10) % evaluation budget finished
           disp('inside main_par')
           [problem.numCallF ]
           process.dynamics.update(archive,opt,problem);
           archive.usedEvalHist=[archive.usedEvalHist 0];

           process.update_due_to_change(restart,archive,opt,problem);  


           break  
        elseif (restart.terminationFlag==-5) % problem changed           
            process.dynamics.update(archive,opt,problem);
            archive.usedEvalHist=[archive.usedEvalHist 0];
            process.update_due_to_change(restart,archive,opt,problem);  
            process.reset_static(opt, problem);
            archive=Archive(problem); % create new object archive
        else        
            archive.update(restart,process,opt,problem); % update archive
            process.update(restart,archive,opt,problem); % update process
            disp(['timestep=' num2str(process.dynamics.currentTimeStep) ', restartNo = ',   num2str(process.restartNo-1), ', usedEval = ' num2str(process.usedEvalTillRestart) , '=' , num2str(problem.numCallF), ' (' num2str(problem.numCallF/problem.maxEval*100)  '%), archiveSize = ' num2str(archive.size) ' ,bestVal=' num2str(restart.bestVal)] ) % print optimization progress summary after each restart
        end
    end
    if strcmp(problem.suite,'GMPB')
        offErr=problem.extProb.CurrentError;
        save(filename,'offErr','-v7.3')
    else
        save(filename,'problem','process','archive','-v7.3')
    end
    movefile(filename,'result')
end


   

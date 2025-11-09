% Update The optimization process after perfoming a restart and updating the archive. 
% Data of process will be used by the upcoming restart

% Written by Ali Ahrari (aliahrari1983@gmail.com)
% last update by Ali Ahrari on 13 Jan 2022  
       
classdef OptimProcess < handle
    properties
        restartNo=0 % current restart No
        subpopSize; % subpop size for the restart
        mu;  % parent size for the restart
        recWeights; % weights for recombination 
        muEff;  % effective number of parents
        coreSpecStrPar;  %core search-specific parameters (object)
        defNormTabDis; % default normalized taboo distance (for future archived solutions)
        usedEvalTillRestart=0 % used evaluations until current restart number
        bestValTillRestart=inf % best value found so far until upcoming resest value found untilupcoming restart
        iniR0; % inital benchmark distance for initialization 
        startTime=round(toc*1000) % start time of the optimization process (ms)
        dynamics;
    end
    methods
        function optimProcess=OptimProcess(opt,problem)
            optimProcess.subpopSize=floor(opt.coreSearch.finSubpopSizeCoeff*sqrt(problem.dim)); % subpop size for the restart
            optimProcess.mu=max([1,floor(0.5+optimProcess.subpopSize*opt.coreSearch.muToPopSizeRatio)]); % parent size for the restart
            W=log(1+optimProcess.mu)-log(1:optimProcess.mu); % recombination weights
            optimProcess.recWeights= W/sum(W); 
            optimProcess.muEff=sum(optimProcess.recWeights)^2.0/sum(optimProcess.recWeights.^2);  % effective number of parents
            if strcmp(opt.coreSearch.algorithm,'CMSA')
                optimProcess.coreSpecStrPar=SpecStrParCMSA(optimProcess,opt,problem);  %core search-specific parameters (object)
            elseif strcmp(opt.coreSearch.algorithm,'CMA')
                optimProcess.coreSpecStrPar=SpecStrParCMA(optimProcess,opt,problem);  %core search-specific parameters (object)
            end
            optimProcess.defNormTabDis=opt.archiving.iniNormTabDis; % default normalized taboo distance (for future archived solutions)
            optimProcess.iniR0=sqrt(problem.dim)/2; % inital benchmark distance for initialization 
            optimProcess.dynamics=DynamicManager(opt,problem);
        end
    
        function reset_static(process,opt,problem) 
            process.restartNo=0; % current restart No
            process.subpopSize=floor(opt.coreSearch.finSubpopSizeCoeff*sqrt(problem.dim)); % subpop size for the restart
            process.mu=max(1,floor(0.5+process.subpopSize*opt.coreSearch.muToPopSizeRatio)); % parent size for the restart
            W=log(1+process.mu)-log(1:process.mu); % recombination weights
            process.recWeights= W/sum(W);
            process.muEff=sum(process.recWeights)^2.0/sum(process.recWeights.^2);  % effective number of parents
            process.coreSpecStrPar.update(process,opt,problem);  %core search-specific parameters (object)
            process.defNormTabDis=opt.archiving.iniNormTabDis; % default normalized taboo distance (for future archived solutions)
            process.bestValTillRestart=inf; % best value found so far until upcoming resest value found untilupcoming restart
            process.iniR0=sqrt(problem.dim)/2; % inital benchmark distance for initialization 
        end
        
        function update_due_to_change(optimProcess,restart,archive,opt,problem) 
            usedEvalThisRestart=archive.usedEvalHist(end)+restart.usedEvalEvolve+restart.usedEvalMerge+restart.usedEvalChangeDetect; % The number of function evalautions until the end of this start
            optimProcess.usedEvalTillRestart=optimProcess.usedEvalTillRestart+usedEvalThisRestart; % used eval until now
        end
        function update(optimProcess,restart,archive,opt,problem) 
            % Update process (after updating archive) for the upcoming restart 
            optimProcess.restartNo=optimProcess.restartNo+1; % update restart No
            % Now update subpopSize and numSubpop for the upcoming restart
            usedEvalThisRestart=archive.usedEvalHist(end)+restart.usedEvalEvolve+restart.usedEvalMerge+restart.usedEvalChangeDetect; % The number of function evalautions until the end of this start
            usedEvalSoFar=optimProcess.usedEvalTillRestart+usedEvalThisRestart; % the number of evalautions used so far
            % Now determine the subpop size for the next restart
            subpopSizeCoeff=opt.coreSearch.iniSubpopSizeCoeff*(opt.coreSearch.finSubpopSizeCoeff/opt.coreSearch.iniSubpopSizeCoeff)^(usedEvalSoFar/problem.maxEval); % current subpop size coefficient (geometric interpolation given the inital and final value)
            optimProcess.subpopSize=floor(subpopSizeCoeff*sqrt(problem.dim));% subpop size for the upcoming restart
            optimProcess.mu=max([1,floor(0.5+optimProcess.subpopSize*opt.coreSearch.muToPopSizeRatio)]); % parent size for the upcoming restart
            % calculate recombination weights and muEff
            W=log(1+optimProcess.mu)-log(1:optimProcess.mu); % recombination weights
            optimProcess.recWeights= W/sum(W);
            optimProcess.muEff=sum(optimProcess.recWeights)^2.0/sum(optimProcess.recWeights.^2);  % effective number of parents
            optimProcess.coreSpecStrPar.update(optimProcess,opt,problem); % upate startegy paraeters that depend on the core search algorithm
            optimProcess.defNormTabDis=UtilityMethods.lin_prctile(archive.normTabDis,opt.archiving.newNormTabDisPrc); % default normalize taboo distance for upcoming restart - updated on 9 Feb 2022 to perform linear interpolation
            optimProcess.usedEvalTillRestart=optimProcess.usedEvalTillRestart+usedEvalThisRestart; % used eval until now
            optimProcess.bestValTillRestart=min( [restart.bestVal, optimProcess.bestValTillRestart]); % update best value until now
            optimProcess.iniR0=min([restart.recIniR0*opt.niching.iniR0IncFac,0.5*sqrt(problem.dim)]); % update the inital benchmark distance for initialization 
        end
    end % methods
end % class






% An object from this class is a restart

% Written by Ali Ahrari (aliahrari1983@gmail.com)
% last update by Ali Ahrari on 13 Jan 2022 """
% =============================================================================
classdef Restart < handle 
    % Create the restart object which includes all the required restart-dependent
    % information to perform a restart 
    
    properties
        stagSize % stagnation size for the stagnation check. The size depends on the iteration No
        tolHistSize % Stall size for checking tolHistFun  
        usedEvalEvolve=0 % used evalaution for evolution of the subpopulations
        usedEvalMerge=0 % used evalaution for checking the merge with archive posibility
        usedEvalChangeDetect=0;
        iterNo=0 % iteration number of the restart
        terminationFlag=0 % termination flag of restart
        bestVal=inf % best value of all subpopulation
        bestSol; % best solution of all subpopulation
        recIniR0=nan % recommended intial value for R0 for the next restart
    end
    methods
        function restart=Restart(process,opt,problem) % Constructor
            restart.stagSize=floor(opt.stopCr.stagPar(1)+opt.stopCr.stagPar(3)*problem.dim/process.subpopSize); % stagnation size for the stagnation check. The size depends on the iteration No
            restart.tolHistSize=floor(opt.stopCr.tolHistSizePar(1)+opt.stopCr.tolHistSizePar(2)*problem.dim/process.subpopSize); % Stall size for checking tolHistFun
            restart.bestSol=zeros(0,problem.dim);
        end
        function subpop=initialize_subpop(restart,archive,process,opt,problem)  
            % Initialize subpopulations for the restart 
            % For improved computational complexity, rescale everything into [0, 1]**D, 
            % sample the solutions and then scale back to the actual search range 
            archSolRescaled=zeros(archive.size,problem.dim); % archived solutions in the rescaled space ([0,1]^D) 
            for k=1:archive.size
                archSolRescaled(k,:)=(archive.solution(k,:)-problem.lowBound)./(problem.upBound-problem.lowBound);
            end
            % Generate the center using Maximin startegy 
            numReject=0; % the number of consecutive failed attempts to create a subpopulation
            R0=process.iniR0; % normalized benchmark ditance: R0*d_hat determines the taboo region for sampling

            wasSuccess=false;        
            while ~wasSuccess % try to initialize a subpopulation 
                X=rand(1,problem.dim); % candidate center-random sampling
                % Check different requirements one-after-another. When a requirement
                % is not satiisfied, sample another point"""
                chkDis=true; % it is an acceptable center unless proven otherwise
                % Requirement: center should be outside taboo regions of archived solutions
                if archive.size>0
                    dis2dis=pdist2(X,archSolRescaled); % distance to archives solutions
                    chkDis=all(dis2dis>(archive.normTabDis*R0));
                end
                if chkDis % if true, then this center is acceptable       
                    wasSuccess=true;
                    restart.recIniR0=R0;
                else
                    numReject=numReject+1;  
                end
                if numReject>opt.niching.maxRejectIni % too many successive rejections, reduce R0 an Rsearch
                    numReject=0;
                    R0=R0*opt.niching.redCoeff;
                end
            end % while

            % The center have been generated in [0, 1]^D. Now generate the subpopulations
            center=X.*(problem.upBound-problem.lowBound)+problem.lowBound; % center: scale back to the problem search range
            smean=min(opt.coreSearch.maxIniSigma,R0*opt.coreSearch.iniSigCoeff); % global step sizes of the subpopulations
            stretch=problem.upBound-problem.lowBound;

            % Create the subpopulations and define its properties based on the employed core search
            if strcmp(opt.coreSearch.algorithm,'CMSA')
                subpop=SubpopulationCMSA(center, smean, stretch, process.subpopSize); 
            elseif strcmp(opt.coreSearch.algorithm,'CMA')
                subpop=SubpopulationCMA(center, smean, stretch, process.subpopSize);
            end

        end % function
        

        function run_one_restart(restart,subpop,archive,process,opt,problem) 
            % performs one restart 
            %subpop=restart.initialize_subpop(archive,process,opt,problem); % initialize the subpopulation

            while restart.terminationFlag==0  % if restart has not been terminated   
                restart.iterNo=restart.iterNo+1;
                restart.stagSize=floor(opt.stopCr.stagPar(1)+opt.stopCr.stagPar(2)*restart.iterNo+...
                               +opt.stopCr.stagPar(3)*problem.dim/process.subpopSize);% size for no stagnation check
                % evolve the subpop
                subpop.update_taboo_region(restart,archive,process,opt,problem); % determine the critical taboo region for the subpopulation
                subpop.update_merge_check(restart,archive,opt,problem); % determine the archived solutions that may share the basin with the subpopulation
                subpop.evolve(restart,archive,process,opt,problem); % evolve subpopulation for one generation
                restart.usedEvalEvolve=subpop.usedEvalEvolve;
                %subpop.update_merge_check(restart,archive,opt,problem) % determine the archived solutions that may share the basin with the subpopulation
                subpop.update_term_flag(restart,archive,process,opt,problem); %update the termination flag of the subpopulation
                restart.usedEvalMerge=subpop.usedEvalMerge;
                restart.usedEvalChangeDetect=subpop.usedEvalChangeDetect;
                restart.bestVal=subpop.bestVal; % store the  best value of the subpopultion  
                restart.bestSol=subpop.bestSol; % store the  best value of the subpopultion  
                restart.terminationFlag=subpop.terminationFlag; % store the termination flag of each subpop
                % break if there is not enough evalaution budget left
                remainEvalAfter=problem.maxEval-(process.usedEvalTillRestart+ restart.usedEvalEvolve+restart.usedEvalMerge+process.subpopSize); % remaining evalautions after performing one more evolve
                reqEvalForDetectMult=min(archive.size,opt.archiving.neighborSize)*opt.archiving.hillVallBudget; % maximum required evalautio for analyzing the best solution when updating the archive
                %disp([problem.numCallF, subpop.bestVal])
                if problem.numCallF>=problem.maxEval%reqEvalForDetectMult
                    [problem.numCallF ]
                    restart.terminationFlag=-10; % terminate restart due to shortage of evaaution budget
                    break
                end
            end % while
        end % function
    end % methods
end % class


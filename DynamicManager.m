% This class is used when the problem is dynamic    

classdef DynamicManager < handle
    properties
        endArchive=cell(1,0) % Archives  at the end of each time step
        endSolNorm=cell(1,0) % end archive solutions sorted and normalized wrt search hrange
        chDetectSolXf % candidate solutions and their values for checking occurance of the change by the change detection function
        currentTimeStep=0 % current time step
        recCenter % recommended centers by the employed prediction method for the next subpopulation
        recStepSize=[] % recommended step sizes by the employed prediction method for the next subpopulation
        recInd=1 % the index of the recomanded solution for current restart
        usedEvalChDetect=0 % the number evalautions used by the change detection mechanism
        lastChCheckAtFE=0 % the last time (evalaution) at which the change mechanism was called
        usedPredictLevel=[] % selected prediction level by the code/user
    end
    methods 
        function dynamics=DynamicManager(opt,problem)
            dynamics.chDetectSolXf=zeros(0,problem.dim+1);
            dynamics.recCenter=zeros(0,problem.dim);
        end
        function update(dynamics,archive,opt,problem)  % call this when a change occurs
            dynamics.currentTimeStep=dynamics.currentTimeStep+1;
            dynamics.recInd=1;  %% the index of the recomanded solution for current restart
            [~,ind]=sort(archive.value);
            N=numel(ind);
            tmp=(archive.solution(ind,:)-repmat(problem.lowBound,N,1)) ./ repmat(problem.upBound-problem.lowBound,N,1);
            dynamics.endSolNorm{dynamics.currentTimeStep}=tmp;
            dynamics.endArchive{dynamics.currentTimeStep}=archive;
            dynamics.gen_rec_ini_pop(opt,problem)
            dynamics.chDetectSolXf=zeros(0,problem.dim+1);
        end
        
     
        % change detection mechanism       
        function hasChanged=detect_change(dynamics,opt,problem)
            N=size(dynamics.chDetectSolXf,1); % number of candid solutions for change detection 
            ind=randi(N); % randomly selected solution
            % Now check if a change has occured
            dynamics.usedEvalChDetect=dynamics.usedEvalChDetect+1;
            dynamics.lastChCheckAtFE=problem.numCallF;
            if strcmp(problem.suite,'GMPB')
                
                if problem.extProb.RecentChange
                    problem.extProb.RecentChange=0;
                    hasChanged=true;
                else
                    hasChanged=false;
                end
            elseif strcmp(problem.suite,'DMMOP')  
                hasChanged=problem.extProb.CheckChange(dynamics.chDetectSolXf(ind,1:problem.dim), dynamics.chDetectSolXf(ind,problem.dim+1));
                if (~problem.knownChanges) 
                    problem.func_eval(dynamics.chDetectSolXf(ind,1:problem.dim)); % CheckChange assumes informed changes. if changes are unknown, then counf change detection as one extra evaluation
                end
            else
                testF=problem.func_eval(dynamics.chDetectSolXf(ind,1:problem.dim));
                hasChanged = abs(testF-dynamics.chDetectSolXf(ind,problem.dim+1))>opt.dyna.tolChangeF;% if true, the problem has changed 
            end
            %hasChanged

            
        end

        % finds the time history of one solution given history of multiple solutions
        function histX=track_past_history(dynamics,index,endSolNorm)
            % index: scalar: index of the solution in the current time archive
            % endArchivedSol: cell of matrisses
            maxLevel=numel(endSolNorm);
            [Nsol,D]=size(endSolNorm{end});
            histX=nan*ones(maxLevel,D);
            histX(maxLevel,:)=endSolNorm{end}(index,:);
            for L = (maxLevel):-1:2
                dis2dis=pdist2(endSolNorm{L-1},histX(L,:))';
                [~,ind]=min(dis2dis); % index of closest solution
                histX(L-1,:)=endSolNorm{L-1}(ind,:);
            end
        end
 
        % provides the recommended centers and step sizes for the future subpopulations
        function gen_rec_ini_pop(dynamics,opt,problem)
            dynamics.usedPredictLevel=[];
            maxLevelLimit1=0;
            val1=opt.dyna.maxPL+1;
            val2=numel(dynamics.endSolNorm);         
            for k=1:min(val1,val2)
                if size(dynamics.endSolNorm{end-k+1},1)>0
                    maxLevelLimit1=maxLevelLimit1+1;
                else
                    break
                end
            end

            maxPL= min(opt.dyna.maxPL,maxLevelLimit1);
            maxDataPoint=min(opt.dyna.maxPL+1,maxLevelLimit1);

            %disp([maxLevelLimit1,maxPL,maxDataPoint]);

            Nsol=size(dynamics.endSolNorm{end},1);
            if maxLevelLimit1==0
                disp(  'maxLevelLimit1 is 0' )
                dynamics.recCenter=zeros(0,problem.dim); % scale back to the original search space
                dynamics.recStepSize=[]; % default sigma-maximum
                dynamics.usedPredictLevel=[];
                
            elseif maxLevelLimit1==1
                 % use the last solutions from the previous time steps
                dynamics.recCenter=dynamics.endSolNorm{end}.*repmat(problem.upBound-problem.lowBound,Nsol,1)+repmat(problem.lowBound,Nsol,1); % scale back to the original search space
                dynamics.recStepSize=opt.coreSearch.maxIniSigma*ones(1,size(dynamics.recCenter,1)); % default sigma-maximum
                dynamics.usedPredictLevel=[dynamics.usedPredictLevel 1];

            elseif maxLevelLimit1>1
                % calculate the normalized solutions in [0,1] to calculate the normalized distance between solutions in successive archives
                dynamics.recCenter=nan*ones(Nsol,problem.dim);
                dynamics.recStepSize=nan*ones(1,Nsol);
                for solNo=1:Nsol
                    Xhist=dynamics.track_past_history(solNo,dynamics.endSolNorm(end-maxDataPoint+1:end)); % time history of a solution in the normalized space
                    [xhatNow,estPreErrNowNorm,bestLevel]=Prediction.AMLP(Xhist,maxPL); % best prediction

                    if strcmp(opt.dyna.predictMethod,"FLP")
                        L=numel(estPreErrNowNorm); 
                    elseif strcmp(opt.dyna.predictMethod,"AMLP")
                        L=bestLevel;
                    end
                    dynamics.recCenter(solNo,:)=xhatNow(L,:).*(problem.upBound-problem.lowBound)+problem.lowBound; 
                    dynamics.recStepSize(solNo)=min(max(estPreErrNowNorm(L)*opt.dyna.recStepSizeCoeff,opt.coreSearch.maxIniSigma*1e-6), opt.coreSearch.maxIniSigma);
                    dynamics.usedPredictLevel=[dynamics.usedPredictLevel L];
                end
            end % if 
        end %function
    end %methods
end %class
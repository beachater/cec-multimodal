% An object from this class is a subpopualtion

%Written by Ali Ahrari (aliahrari1983@gmail.com)
%last update by Ali Ahrari on 13 Jan 2022  

 
classdef Subpopulation < handle
    % Create a subpopulation object 
    properties
        center % center of subpopulation
        mutProfile % mutation profile of the subpopulation (object)
        samples % sampled solutions from the subpopulation (object)
        bestSol % best solution of the subpopualtion
        bestVal=inf % value of the best solutions of subpopulation
        tabooRegion % taboo regions for the subpopulation (object)
        mergeCheck % information for checkig the merge termination criteria (object)
        localConvergeCheck % information for checkig the merge termination criteria (object)
        bestValNonEliteHist=[]  %  history of the best value of non-elite solutions at each iteration
        medValNonEliteHist=[]  %  history of the median value of non-elite solutions at each iteration
        maxCriticalityHist=[] % history of the maximum criticality of the archived solution
        terminationFlag=0 %0: not terminated, 1:converged, -1: No improvement, -2: Condition number of C exceeds the limit, 4: Predicted convergence to a local optimum
        usedEvalEvolve=0 % No of evalautions used by evolution of this subpopulation
        usedEvalMerge=0 % No of evalautions used by merge check  
        usedEvalChangeDetect=0;
        iterNo=0
    end
    methods
        function subpop=Subpopulation(center,smean,stretch,popSize) % Constructor  
            D=numel(center);
            subpop.center=center; % center of subpopulation
            subpop.mutProfile=MutationProfile(smean,stretch); % mutation profile of the subpopulation (object)
            subpop.samples=Sampling(popSize,D); % sampled solutions from the subpopulation (object)
            subpop.bestSol=zeros(0,D); % best solution of the subpopualtion
            subpop.tabooRegion=TabooRegion(D); % taboo regions for the subpopulation (object)
            subpop.mergeCheck=MergeCheck(); % information for checkig the merge termination criteria (object)
            subpop.localConvergeCheck=LocalConvergeCheck(); % information for checkig the merge termination criteria (object)
        end
        function normDis=calc_norm_dis(subpop,x1,x2,disMetric)
            % Calculate the normalized distance (Mahalanobis or Euclidean) between
            % points x1, x2 given mutation profile of the subpopulation 
            if strcmp(disMetric,'Mahalanobis') % if the distance metric is Mahalanobis (default)
                normDis=sqrt((x1-x2)*subpop.mutProfile.Cinv*(x1-x2)')/subpop.mutProfile.smean;
            elseif  strcmp(disMetric,'Euclidean') % If the distance meric is Euclidean
                mean_str=geomean(subpop.mutProfile.stretch);
                normDis=pdist2(x1, x2)/(subpop.mutProfile.smean*mean_str);
            else
                error('Distance metric is not valid. Aborting')
                abort
            end
        end  
        
        function tabAccept=is_taboo_acceptable(subpop,sample,tempRedRatio,opt)
            % Check if sample solution is taboo acceptable given the subpopulation and temporary reduction parameter 
            tabAccept=true; % is is acceptable unless proven otherwise
            for tabInd=subpop.tabooRegion.criticInd % check against all critical taboo points one after another
                normDis=subpop.calc_norm_dis(sample,subpop.tabooRegion.center(tabInd,:),'Mahalanobis'); % normalized distances to this critical taboo points
                tabAccept=normDis>=(subpop.tabooRegion.normTabDis(tabInd)*tempRedRatio); % is it taboo acceptable wrt this taboo point?
                if ~tabAccept %  rejected
                    break % do not check for other critical taboo points
                end
            end
        end

        function estimate_taboo_regions_criticality(subpop,opt,problem) 
            % provides an estimate for the criticality of the taboo points (all of them) 
            L=zeros(1,numel(subpop.tabooRegion.normTabDis));
            for k=1:numel(L)
                L(k)=subpop.calc_norm_dis(subpop.tabooRegion.center(k,:),subpop.center,'Mahalanobis');
            end
            intU=(L+subpop.tabooRegion.normTabDis);
            intL=(L-subpop.tabooRegion.normTabDis);
            subpop.tabooRegion.criticality=normcdf(intU)-normcdf(intL);
        end
        
        function update_taboo_region(subpop,restart,archive,process,opt,problem)
            % Determine critical taboo regions for the subpopulation. 
            maxCount=archive.size; % maximum number of taboo regions
            subpop.tabooRegion.center=zeros(maxCount,problem.dim); % Preallocation: centers of the taboo regions
            subpop.tabooRegion.normTabDis=zeros(1,maxCount); % Preallocation: normalized taboo distances of the taboo regions 
            count=0; % counter for the actual number of taboo regions (including non-critical ones)
            for k=1:maxCount % check if the archived solution is better than the subpopulation
                if archive.value(k)<subpop.bestVal 
                    count=count+1;
                    subpop.tabooRegion.center(count,:)=archive.solution(k,:); % that archived solution is a taboo solution
                    subpop.tabooRegion.normTabDis(count)=archive.normTabDis(k); % normalized taboo distance of the archived solution 
                end
            end
            % discard the unused indexes
            subpop.tabooRegion.center=subpop.tabooRegion.center(1:count,:);
            subpop.tabooRegion.normTabDis=subpop.tabooRegion.normTabDis(1:count);

            % Determine which taboo regions are critical   
            if count>0
                subpop.estimate_taboo_regions_criticality(opt,problem); % estimate the criticality of all taboo regions
                [~,crInd]=sort(1-subpop.tabooRegion.criticality); % sorted according to their criticality
                Ncr=sum(subpop.tabooRegion.criticality>opt.niching.criticTabooThresh); % the number of critical taboo regions
                subpop.tabooRegion.criticInd=crInd(1:Ncr); % index of critical taboo regions among all taboo regions
            else
                subpop.tabooRegion.criticInd=[];
            end
            
            
            subpop.maxCriticalityHist=[subpop.maxCriticalityHist,max([subpop.tabooRegion.criticality, 0])]; % history of the maximum criticality 
        end
        
        function update_merge_check(subpop,restart,archive,opt,problem)
            % updates the property 'mergeCheck'. It determines the archived solutions 
            % that are likely to share the basin with the subpopulation.  """
            if archive.size>0 
                % calculate the the normalized taboo distance between the center of the subpopulation and the archived solution
                L=zeros(1,archive.size); % Preaalocation of the normalized distances
                for k=1:numel(L) 
                    L(k)=subpop.calc_norm_dis(archive.solution(k,:),subpop.center,'Mahalanobis');
                end
                Mergeability=(archive.normTabDis+opt.stopCr.merge.addedConst)./L; % the similarity score of each archived solution. 

                % now determine the archived solutions with high similarity score
                [maxMergeability,indMax] = max(Mergeability);  % return the highest Mergeability
                candidArchCount=sum(Mergeability>opt.stopCr.merge.threshold); % the number of higher than threshod similarities
            else
                indMax=-1; % index of archived solutions with high similarity to the subpopulation 
                maxMergeability=0; % similarity of archived solutions with high similarity
                candidArchCount=0; % the number of candidate archived solutions for merge
            end
            % now use the calculated values to update the mergeCheck attributes
            subpop.mergeCheck.bestCandidArchIndHist=[subpop.mergeCheck.bestCandidArchIndHist,indMax];
            subpop.mergeCheck.bestCandidArchMergeabilityHist=[subpop.mergeCheck.bestCandidArchMergeabilityHist,maxMergeability];
            subpop.mergeCheck.candidArchCountHist=[subpop.mergeCheck.candidArchCountHist,candidArchCount];
        end % function
    
        function evolve(subpop,restart,archive,process,opt,problem) 
            % Evolve the subpopulation (mutation, selection, recombination) 
            subpop.sample_solutions(restart,archive,process,opt,problem); % sample the subpopulation members and evaluate them
            subpop.eval_solutions(restart,archive,process,opt,problem); % sample the subpopulation members and evaluate them
            subpop.select(restart,archive,process,opt,problem); % perform selection        
            % Now perform recombination based on the employed core search
            subpop.recombine(restart,archive,process,opt,problem);
        end
        function repair_infeas(subpop,solNo,opt,problem) % repair the infeasible solution solNo
            if opt.coreSearch.repairInfeasMethodCode>0
                if max(problem.upBound<subpop.samples.X(solNo,:)) || max(problem.lowBound>subpop.samples.X(solNo,:)) 
                    subpop.samples.wasRepaired(solNo)= true; % it was repiared           
                    % (element-wise relocation, keeping symmetry around center). 
                    relocItU=find(subpop.samples.X(solNo,:)>problem.upBound); % indexes of elements that are greater than the upper limit
                    if numel(relocItU)>0
                        tmpUp=problem.upBound(relocItU);
                        tmpLow=2*subpop.center(relocItU)-problem.upBound(relocItU); 
                        tmpLow=max([tmpLow;problem.lowBound(relocItU)]);
                        subpop.samples.X(solNo,relocItU)=tmpLow+rand(1,numel(relocItU)).*(tmpUp-tmpLow);
                    end
                    clear relocItU                 
                    % relocate the elements that are smaller than the search range limit
                    relocItL=find(subpop.samples.X(solNo,:)<problem.lowBound); 
                    if numel(relocItL)>0
                        tmpLow=problem.lowBound(relocItL);
                        tmpUp=2*subpop.center(relocItL)-problem.lowBound(relocItL); 
                        tmpUp = min([tmpUp;problem.upBound(relocItL)]);
                        subpop.samples.X(solNo,relocItL)=tmpLow+rand(1,numel(relocItL)).*(tmpUp-tmpLow);
                    end
                    clear relocItL                
                    % update the strategy parameter
                    subpop.samples.Z(solNo,:)=(subpop.samples.X(solNo,:)-subpop.center) / subpop.samples.s(solNo);  
                end % if - infeasible
            end
        end % function

        function eval_solutions(subpop,restart,archive,process,opt,problem)  
            % -------------- perform evalaution ----------------- 
            for solNo=1:process.subpopSize
                % ----------- Calculate bound violation -------------
                penU= subpop.samples.X(solNo,:)-problem.upBound;
                penU= (penU).*(penU>0);
                penL= problem.lowBound-subpop.samples.X(solNo,:);
                penL= (penL).*(penL>0);
                penUL=sum(penU+penL); % overal constraint (bound) violation
                subpop.samples.isFeas(solNo)= ~(penUL>0);
                if ~subpop.samples.isFeas(solNo)
                    subpop.samples.f(solNo)=opt.coreSearch.objValUpLimit*(1+penUL); %death penalty-do not call the objective function
                else
                    oldNumCallF=problem.numCallF;
                    subpop.samples.f(solNo)=problem.func_eval(subpop.samples.X(solNo,:)); % evaluate the solution
                    subpop.usedEvalEvolve=subpop.usedEvalEvolve+(problem.numCallF-oldNumCallF); % used eval by subpop
                    clear oldNumCallF
                end
                % check for petontial change in the problem and update bench solutions
                if problem.isDynamic
                    chk1=(size(process.dynamics.chDetectSolXf,1))>=(opt.dyna.benchSolSizeCoeff*process.subpopSize);
                    %chk2=(problem.numCallF-process.dynamics.lastChCheckAtFE)>=opt.dyna.chCheckFr;
                    chk2=rand<(1/opt.dyna.chCheckFr);
                    if chk1 && chk2
                        oldNumCallF=problem.numCallF;                        
                        hasChanged=process.dynamics.detect_change(opt,problem);
                        subpop.usedEvalChangeDetect=subpop.usedEvalChangeDetect+(problem.numCallF-oldNumCallF);
                        if hasChanged
                            subpop.terminationFlag=-5;
                            break
                        end
                    end
                  
                    if ~chk1
                        tmp=[subpop.samples.X(solNo,:) subpop.samples.f(solNo)];  
                        process.dynamics.chDetectSolXf=[process.dynamics.chDetectSolXf; tmp];
                    end
                end
       
            end % for
            subpop.iterNo=subpop.iterNo+1;

            % ---------- update the indicators for stagnation check ------------
            subpop.bestValNonEliteHist=[subpop.bestValNonEliteHist, min(subpop.samples.f)]; % best non elite value
            subpop.medValNonEliteHist=[subpop.medValNonEliteHist,median(subpop.samples.f)];

            % ----------- Drop old values if the array is too long --------------
            maxSize=max(restart.stagSize,restart.tolHistSize);
            if numel(subpop.bestValNonEliteHist)>maxSize
                subpop.bestValNonEliteHist=subpop.bestValNonEliteHist(end-maxSize+1:end);
                subpop.medValNonEliteHist=subpop.medValNonEliteHist(end-maxSize+1:end);
            end        
        end % function


        function update_term_flag(subpop,restart,archive,process,opt,problem)
            % ---------- Update the termination flag of the subpopulation -----------
            % 0: Not terminated
            % 1: converged: tolHistFun criterion activated
            % 2: converged: the mutation strength along the largest dimension is less than tolX
            % 3: will converge to an archived solution (merge check was positive)
            % 4: will converge to a local minimum (local convergence check was positive)
            % -1: no convergnece: subpopulation has stagnated
            % -2: no convergence: condition number of C has exceeded the limit  """

            % Condition number of C
            condC=(max(subpop.mutProfile.stretch)/min(subpop.mutProfile.stretch))^2; % condition number of the C
            if condC>opt.stopCr.maxCondC % Check condition number of C  
                subpop.terminationFlag=-2;
            end

            % Stagnation 
            if subpop.iterNo>=restart.stagSize && (subpop.terminationFlag==0) % stagnation check
                N0=numel(subpop.bestValNonEliteHist);
                ind1=N0-restart.stagSize+(1:20); % the firts 20 elements 
                ind2=N0-20+(1:20); % the last 20 elements
                minImpBest=median(subpop.bestValNonEliteHist(ind2))-median(subpop.bestValNonEliteHist(ind1));
                minImpMed =median(subpop.medValNonEliteHist(ind2))-median(subpop.medValNonEliteHist(ind1));
                if min(minImpBest,minImpMed)>0 % stagnation check
                    subpop.terminationFlag=-1; % termination du to stagnation
                end
            end

            % tolHistFun convergence criterion        
            if ((subpop.iterNo)>=restart.tolHistSize) && (subpop.terminationFlag==0) % check for tolhistfun
                maxDiff=UtilityMethods.peak2peak(subpop.bestValNonEliteHist(end-restart.tolHistSize+1:end)); % max difference in the latest best values
                if maxDiff<opt.stopCr.tolHistFun
                    subpop.terminationFlag=1; % convergence 
                end
            end

            % step-size criterion
            if ((max(subpop.mutProfile.stretch)*subpop.mutProfile.smean)<opt.stopCr.tolX) && (subpop.terminationFlag==0) % too small a mutation strength?
                subpop.terminationFlag=2; % convergence 
            end

            % -------- Now check for potential merge with an archived solution ---------
            if subpop.terminationFlag==0
                % check some required condifitons for investigating this termination criterion as it has some cost
                chk1=subpop.iterNo>=(opt.stopCr.merge.windowSizeCoeff*restart.tolHistSize); % 1)iteration number is equal or greater than a minimum value (give some time for evolution)
                % 2) one one specific archive has been candidate for merge over the wondow size
                N1=ceil(opt.stopCr.merge.windowSizeCoeff*restart.tolHistSize);
                chk2=false;
                if chk1
                    chk2=all(subpop.mergeCheck.candidArchCountHist(end-N1+1:end)==1); % corrected on 9-feb-2022
                end
                chk3=subpop.bestVal<opt.coreSearch.objValUpLimit; % The best solution is not infeasible
                chk4=subpop.iterNo>subpop.mergeCheck.checkAfterIterNo; % enough evolution from the last check for merge
                chk5=subpop.mergeCheck.mergeAtUsedEval==inf; % has not been flagged by the merge operator before
                %chk6                    % the only candiate remains unchanged (should be added)
                if  chk1 && chk2 && chk3 && chk4 && chk5 % now used detect multimodal method to check if the subpopulation is sharing the basin with the highly similar archive solution
                    usedEval=0; % used eval for checking
                    isNew=false; % by default, they share the same basin unless a solution worse than both is detected in the middle 

                    stp=1/opt.stopCr.merge.maxEval;
                    r=(stp/2):stp:1;
                    % now define the end solutions (endX1, endX2) and sample test solutions between them 
                    endX1=subpop.bestSol; % one end is the best solution of the subpopulation
                    endF1=subpop.bestVal;
                    endX2=archive.solution(subpop.mergeCheck.bestCandidArchIndHist(end),:); % The other end point is the candidate archived point
                    endF2=archive.value(subpop.mergeCheck.bestCandidArchIndHist(end));

                    % ------- Test between points one after another ----------  
                    while usedEval<opt.stopCr.merge.maxEval% this is the budget for detect multimodal heuristic
                        testX=r(usedEval+1)*endX1+(1-r(usedEval+1))*endX2; % solution between the end points
                        testF=problem.func_eval(testX);
                        subpop.usedEvalMerge=subpop.usedEvalMerge+1; % track the extra count in subpop
                        usedEval=usedEval+1; % in merge testing
                        if testF>(max(endF1,endF2)+opt.stopCr.tolHistFun)
                            isNew=true;
                            break % it does not share the basin with this archived solution
                        end
                    end % while

                    if ~isNew % the subpopulation is converging to the archive solution
                        if isnan(subpop.mergeCheck.matchArchInd) % if no archive solution has been matched before 
                            subpop.mergeCheck.matchArchInd=subpop.mergeCheck.bestCandidArchIndHist(end); % the index of the archived solution for merge
                            subpop.mergeCheck.mergeAtUsedEval=subpop.usedEvalMerge+subpop.usedEvalEvolve; % The function evaluations at which this termination flag was activated
                            subpop.terminationFlag=3; % consider this subpopulation has converged to this archived solution
                            subpop.bestSol=archive.solution(subpop.mergeCheck.matchArchInd,:); % the best solution is the archive solution
                            subpop.bestVal=archive.value(subpop.mergeCheck.matchArchInd); % the best value is the archived solution value
                        end
                    else % The subpopulation is not converging to this archived solution-updated on 6 Oct 2020
                        chkInterval=opt.stopCr.merge.chkIntervalCoeff*restart.tolHistSize; % does not need to be int
                        subpop.mergeCheck.checkAfterIterNo=chkInterval+subpop.iterNo; % do not check before this iteration
                    end % if-else it is new
                end % if chk1 && chk1 && chk2 && chk3 && chk4 && chk5
            end % merge check

            % predict if it is likely to converge to a local minimum or not
            if subpop.terminationFlag==0
                ws=floor(2+opt.stopCr.localConverge.windowSizeCoeff*restart.tolHistSize); % windowsize for checking differences-at least 2
                % check multiple requirements for activation of this criterion
                chk1=subpop.iterNo>(2+max(opt.stopCr.localConverge.windowSizeCoeff*restart.tolHistSize,ws));   % sufficient time for evolution has been provided 
                chk2=archive.size>0; % there is an estimate for the global optimum value
                chk3=false;
                if chk1
                    chk3=max(subpop.maxCriticalityHist(end-ws+1:end))<opt.stopCr.localConverge.tabooCriticUpLimit; % The maximum criticality is less than this magic number? 
                end
                chk4=subpop.bestVal<opt.coreSearch.objValUpLimit; % best solution is feasible 
                chk5= subpop.localConvergeCheck.stopAtUsedEval==inf; % the positive local converge in the previous iterations
                if chk1 && chk2 && chk3 && chk4 && chk5
                    meanDiff=mean(abs(diff(subpop.bestValNonEliteHist(end-ws+1:end)))); % fluctuations means lack of convergence-consider the history of the best value
                    willBeLocal= meanDiff<(opt.stopCr.localConverge.tolCoeff*(subpop.bestVal-max(archive.value)-opt.archiving.tolFunArch)); % recent fluctuations is not enough compred to difference with the global minimum value
                    if willBeLocal
                        subpop.localConvergeCheck.stopAtUsedEval=subpop.usedEvalMerge+subpop.usedEvalEvolve; % The function evaluations at which this termination flag was activated
                        subpop.terminationFlag=4; % use this termination criterion to terminate the subpopulation
                    end
                end % chk1 and chk2 and chk3 and chk4 and chk5
            end % if local convergence check
        end % update_term_flag function
    end % methods
end % class

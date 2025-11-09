% =============================================================================
% Object archive stores information about previously identified desirable minima and their normalize taboo distances.
% It has methods to add new solutions and update the normalized taboo distances

% Written by Ali Ahrari (aliahrari1983@gmail.com)
% Last update by Ali Ahrari on 16 Jan 2022
% =============================================================================

classdef Archive < handle
    % Archive of presumably distinct solutions referring to distinct global and desirable minima 
    properties
        solution; % archived solutions
        value=[] % values of the archived solutions
        normTabDis=[] % normalized taboo distance of the archived solutions
        foundEval=[] % When the archived solution was added (evaluation number according to algorithm internal count)
        foundEval2=[]; % according to numcallF
        usedEvalHist=[] % the number of evaluations used by the archive (analyzing final solutions)  
        hitTimesSoFar=[] % the number of times an archived solution was detected in the last restart (zero means no other solution converged to this basin)
        hitTimesThisRestart=[] % the number of times an archived solution was detected from the beginning (zero means no other solution converged to this basin)
        size=0 % number of solutions in the archive
        foundTime=[] % the time (in ms) the global minimum was found  
        dummyArchive; % Dummy archive that stores the reported solution according to CEC'2020 format
        
    end
    methods
        function archive=Archive(problem)
            archive.solution=zeros(0,problem.dim); % archived solutions
            archive.dummyArchive=DummyArchive(problem); % Dummy archive that stores the reported solution according to CEC'2020 format
        end

        function append(archive,sol,val,restart,process,opt,problem)
            % ------- append a new solution to the archive ---------"""
            archive.solution=[archive.solution;sol]; % append the solution
            archive.value=[archive.value, val];  %  append the value
            archive.normTabDis=[archive.normTabDis, process.defNormTabDis]; % append the normalized taboo distance
            archive.foundEval= [archive.foundEval, archive.usedEvalHist(end) + restart.usedEvalMerge + ... 
                                + restart.usedEvalEvolve + process.usedEvalTillRestart]; % when it was appended  (internal count of evaluations)
            archive.foundEval2= [archive.foundEval2,problem.numCallF];
            archive.size=archive.size+1; % increase the archive size
            archive.hitTimesThisRestart=[archive.hitTimesThisRestart, 0]; % the number of repeated times this solution was found in this restart
            archive.hitTimesSoFar=[archive.hitTimesSoFar, 0]; % the number of repeated times this solution has been found so far
            archive.foundTime=[archive.foundTime,round(1000*toc)]; % The time (ms) at which the solution was found
        end
        function update(archive,restart,process,opt,problem) % Update the archive
            archive.usedEvalHist=[archive.usedEvalHist 0]; % used eval by archive after the restart
            archive.hitTimesThisRestart=zeros(1,archive.size); % The number of times each desirable minima is found in the recently performed restart
            bestValSoFar = min([restart.bestVal,process.bestValTillRestart]); % global minimum value so far 
            % find and discard the archived solutions that are not desirable (much better solutions have been found) 
            if archive.size>0
                 keepIt = (archive.value-opt.archiving.tolFunArch)<bestValSoFar; % keep these archived solutions and discard the rest
                 discardInd=find(keepIt==0); % these solutions were wrongly labelled as global minima
                 archive.dummyArchive.append(-1,discardInd,archive,problem); % flag -1 to remove these solutions when calculating dynamic-F1 measure
                 % Discard the solutions in Archive that are not global minima
                 archive.solution=archive.solution(keepIt,:);
                 archive.value=archive.value(keepIt);
                 archive.normTabDis=archive.normTabDis(keepIt);
                 archive.foundEval=archive.foundEval(keepIt);
                 archive.foundEval2=archive.foundEval2(keepIt);
                
                 archive.hitTimesThisRestart=archive.hitTimesThisRestart(keepIt);
                 archive.hitTimesSoFar=archive.hitTimesSoFar(keepIt);
                 archive.foundTime=archive.foundTime(keepIt);
                 archive.size=numel(archive.value);
            end

            % Now check the best solution of the subpopulation.        
            Ndesirable=0; % number of subpopulations that could find a desirable minimum (will be zero or one for RS-CMSA-ESII)
            chkEvolved=restart.iterNo>1; % at least one iteration was performed in the restart
            chkIsGlobal=(restart.bestVal-opt.archiving.tolFunArch)<=bestValSoFar; % the subpopulation best solution is sufficiently good

            % -------- check if it is a new global minimum -----------
            if chkEvolved && chkIsGlobal % it is a global minimum
                Ndesirable=Ndesirable+1; % the subpopulation is good enough for further analysis
                if archive.size==0 % archive is empty, add this solution
                    isNew=true;
                    archive.append(restart.bestSol,restart.bestVal,restart,process,opt,problem);
                    archive.dummyArchive.append(1,archive.size,archive,problem); % code 1: add it to reported solutions
                else % archive is not empty
                    % check if it is a new solution, if not, which archived solution does it share the basin with? 
                    [isNew,matchArchNo,usedEval0Total]=archive.is_new_basin(restart.bestSol,restart.bestVal,restart,opt,problem); 
                    archive.usedEvalHist(end)=archive.usedEvalHist(end)+usedEval0Total;
                    if ~isNew % if it is not a new solution
                        archive.hitTimesSoFar(matchArchNo)=archive.hitTimesSoFar(matchArchNo)+1; 
                        archive.hitTimesThisRestart(matchArchNo)=archive.hitTimesThisRestart(matchArchNo)+1; 
                        % ---- if is better than the already archived solution, replace latter with former --------
                        if restart.bestVal<(archive.value(matchArchNo)-opt.stopCr.tolHistFun) 
                            archive.dummyArchive.append(-1,matchArchNo,archive,problem); % Code -1: report the worse one for removal from the dummy archive
                            % ---- Replace the old solution in the archive with this one since it provides a better approximation of the actual global minimum
                            archive.value(matchArchNo)=restart.bestVal;
                            archive.solution(matchArchNo,:)=restart.bestSol;
                            archive.foundEval(matchArchNo)=archive.usedEvalHist(end) + restart.usedEvalMerge + restart.usedEvalEvolve + ...
                                                          + restart.usedEvalChangeDetect + process.usedEvalTillRestart;
                            archive.foundTime(matchArchNo)=round(1000*toc);
                            % append the better solution for the dummy archive
                            archive.dummyArchive.append(1,matchArchNo,archive,problem);
                        end
                    else % it is a new global basin --> append it to the actual and dummy archive
                        archive.append(restart.bestSol,restart.bestVal,restart,process,opt,problem); % append this solution to the archive
                        archive.dummyArchive.append(1,archive.size-1,archive,problem); % Append this solution to the dummy archive
                    end % if else isNew
                end % if-else archive.size==0
            end % if global minimum

            % now adapt the normalized taboo distances of the archived solutions   
            if Ndesirable==0 % the subpopulation did not converge to a global minimum  
                archive.normTabDis=archive.normTabDis*exp(-opt.archiving.tauNormTabDis*(opt.archiving.targetGlobFr)/archive.size); % reduce the normTabDis of all archived points
            elseif isNew % it has converged to a new global minimum             
                 % don't change the normalized taboo distances
            elseif ~isNew % it has converged to an already detected global minimum  
                repDiff=archive.hitTimesThisRestart;
                if max(archive.hitTimesThisRestart==0) % added on 5 July 2020 to avoid division by zero
                    repDiff(archive.hitTimesThisRestart==0)=(-(1-opt.archiving.targetNewNicheFr))/(archive.size-1);
                end
                % repDiff is 1 for the re-found archived solution but a small negative number for the rest of archived solutions
                archive.normTabDis=archive.normTabDis.*exp(opt.archiving.tauNormTabDis*repDiff); % update the normTabDis of all archived points
            end
        end % end function

        function [isNew,sameArchInd,usedEval0Total]=is_new_basin(archive,x,f,restart,opt,problem)
            % Check if solution (x,f) share the basin with one the archived 
            % solutions (isNew), and if yes, with which on (sameArchInd)?""" 
            sameArchInd=nan; % index of the archive that share basin with x
            % check agains each archived solution from closest to the farthest  
            dis=pdist2(archive.solution, x)'; % Euclidean distance of the subpopulation best member to all the archived solutions
            [~,candidInd]=sort(dis);
            candidInd=candidInd(1:min([opt.archiving.neighborSize,numel(candidInd)])); % check against n-closest archived solutions only
            usedEval0Total=0; % evaluation used by the hill-valley heuristic
            for archNo=candidInd % indices of archive solution in the order of distance to x
                usedEval0=0;
                isNew=false;
                while usedEval0<opt.archiving.hillVallBudget % this is the budget for detect multimodal heuristic
                    r=0.8*rand+0.1; % a andom number between .1 and .9
                    testX=archive.solution(archNo,:)+r*(x-archive.solution(archNo,:)); % test point in between
                    testF=problem.func_eval(testX);
                    usedEval0=usedEval0+1; % counter for current use evaluations
                    if testF>(max([f,archive.value(archNo)])+opt.stopCr.tolHistFun)
                        isNew=true; % it does not share the basin with this archived solution
                        break
                    end
                end % while
                usedEval0Total=usedEval0Total+usedEval0;
                if ~isNew % is shares basin with this archive solution
                    sameArchInd=archNo; % report this archived solution as the one that shares the basin with x
                    break % it was not new, so do not check wrt other archived solutions
                end
            end %for
        end % function
    end % methods
end % class



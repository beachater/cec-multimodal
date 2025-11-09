classdef DummyArchive < handle
    % This archive is used only to report found solutions according to the template 
    % required by CEC'2020 competition. The optimization algorithm does not use information
    % from this archive 
    properties 
        solution; % solutions
        value=[] % values
        foundEval=[] % the number of function evaluation according to the algorithmic count
        foundEval2=[] % the number of function evaluation according to the algorithmic count
        foundTime=[] % time when it was found (ms)
        actionCode=[] % actione code (-1, 0, or 1)
    end
    methods
        function dummyArchive=DummyArchive(problem)
            dummyArchive.solution=zeros(0,problem.dim); 
        end  
        function append(dummyArchive,action,index,archive,problem) 
            % Append solutions with proper action code to the dummy archive 
            dummyArchive.solution= [dummyArchive.solution; archive.solution(index,:)]; % add the solution 
            dummyArchive.value=[dummyArchive.value, archive.value(index)]; % add the value 
            dummyArchive.foundEval=[dummyArchive.foundEval, archive.foundEval(index)]; % add the evaluation until found (internal algorithm count of evaluations)
            dummyArchive.foundEval2=[dummyArchive.foundEval2, problem.numCallF]; % add the evaluation until found (internal algorithm count of evaluations)

            dummyArchive.foundTime=[dummyArchive.foundTime, archive.foundTime(index)]; % time (ms) at which it was found
            dummyArchive.actionCode=[dummyArchive.actionCode, action*ones(1,numel(index)) ]; % action code
        end
    end
end



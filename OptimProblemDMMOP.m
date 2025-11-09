% This class creates the object problem. CEC'2013 test problems have already been defined with ID=1,...,20. It is possible to add custom problems 
% which is shown by one example.
% Written by Ali Ahrari (aliahrari1983@gmail.com)
% last updated by Ali Ahrari on 11 Jan 2021""" 



classdef OptimProblemDMMOP < OptimProblem
    % The main class. It creates an object which determine the problem 
    properties 
        extProb % everything related to the problem subclass (external problem)
    end
    methods
        function problem=OptimProblemDMMOP(pid) % constructor
            problem@OptimProblem(pid);
            problem.suite='DMMOP';
            problem.isDynamic=true;
            problem.extProb = DMMOP(pid);
            problem.PID = pid; % function ID
        end

        function set_problem_data(problem) % Calculate and set the problem information --> must be defined 
            % set the search bounds  
            problem.lowBound=problem.extProb.lower;
            problem.upBound=problem.extProb.upper;
            problem.dim=problem.extProb.D;
            problem.maxEval=problem.extProb.evaluation; 
            problem.globMinData.Ngmin=size(problem.extProb.o,1);
        end
        function f=func_eval(problem,x) 
            % Objective function: Evaluate the solution x 
            % x: a 1-D array representing the solution to be evaluated 
            f = problem.extProb.GetFits(x);
            if numel(f)==0
                f=inf;
            else
                f=-f;
                problem.numCallF=problem.numCallF+1;
            end
            problem.numCallF=problem.extProb.evaluated;
        end
    end
end
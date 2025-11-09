% This class creates the object problem. CEC'2013 test problems have already been defined with ID=1,...,20. It is possible to add custom problems 
% which is shown by one example.
% Written by Ali Ahrari (aliahrari1983@gmail.com)
% last updated by Ali Ahrari on 11 Jan 2021""" 



classdef OptimProblem < handle
    % The main class. It creates an object which determine the problem 
    properties 
        suite % which test suite
        PID
        dim; % Problem dimensionality (No. of variables)
        lowBound; % lower bound of the search range
        upBound; % upper bound of the search range
        maxEval; % evaluation budget 
        globMinData; % information about global minima (only used for performance evaluation after optimization)
        % information about global minima (only used for performance evaluation after optimization)
        numCallF=0; % The number of calls to the objective function (updated inside the objective function)
        isDynamic=false; % if problem is dynamics
        knownChanges=false; % if changes are informed (for dynamic problems only)
    end
    methods
        function problem=OptimProblem(pid,D) % constructor
            problem.suite='internal'; % function ID
            problem.PID=pid;
            if nargin==2
                problem.dim = D; % Problem dimensionality (No. of variables)
                % The rest of the object properties are initialized only, and will be set later using the method "load_problem_data(self):"
                problem.lowBound=nan(1,D);
                problem.upBound=nan(1,D);
            end
            problem.globMinData = GlobalMinimaData(); % information about global minima (only used for performance evaluation after optimization)
        end
        

        
        
        function set_problem_data(problem) % Calculate and set the problem information 
            % set the search bounds  
            if (problem.PID==100) 
                problem.lowBound = 0.25*ones(1,problem.dim);
                problem.upBound = 10*ones(1,problem.dim);
                % set the evaluation budget
                problem.maxEval=200000*(problem.dim-1); 
                % info for global minima (not used for optimization
                problem.globMinData.Ngmin=6^problem.dim; % set the number of global minima (not used by the optimization algorithm)
                problem.globMinData.Rnich=.1; % set the niche radius (not used by the optimization algorithm)
                problem.globMinData.val=-1; % set the global minimum value (Not used by the optimization algorithm)
            elseif problem.PID==101 % rosenbrock function
                problem.lowBound = -10*ones(1,problem.dim);
                problem.upBound = 10*ones(1,problem.dim);
                % set the evaluation budget
                problem.maxEval=200000*(problem.dim-1); 
                % info for global minima (not used for optimization
                problem.globMinData.Ngmin=1; % set the number of global minima (not used by the optimization algorithm)
                problem.globMinData.Rnich=.1; % set the niche radius (not used by the optimization algorithm)
                problem.globMinData.val=0; % set the global minimum value (Not used by the optimization algorithm)


            
            end 
        end
        function f=func_eval(problem,x) 
            % Objective function: Evaluate the solution x 
            % x: a 1-D array representing the solution to be evaluated 
            problem.numCallF=problem.numCallF+size(x,1);
            if (problem.PID==100)  
                f= - mean( sin( 10*log(x) ) );
            elseif problem.PID==101
                f = 100*sum((x(1:end-1).^2 - x(2:end)).^2) + sum((x(1:end-1)-1).^2);
            end
        end
    end
end

    
 
 


%     def __init__(self,D):
%         Rnich=None % the niche radius
%         Ngmin=None % The number of global minima
%         val=None % The global minimum value
%         X=np.zeros((0,D)) % The global minima
%         
%     def __str__(self): % show status of the object
%         output=('\n\t\tRnich = ' + str(Rnich) +
%                 '\n\t\tNgmin = ' + str(Ngmin) +
%                 '\n\t\tval = ' + str(val) +
%                 '\n\t\tX = 2D Array os size ' + str(X.shape)   )
%         return output
% 
% """Import the required packages"""
% import numpy as np
% try: 
%     from cec2013.cec2013 import CEC2013
% except:
%     print('Warning: Package for the CEC2013 test problems was not found.')
% import sys
% 
% if __name__=='__main__': % a simple test of this class
%     problem=OptimProblem(21,2) % creates the problem object
%     problem.set_problem_data() % sets the problem information
%     print(problem) % displays the problem information
% 

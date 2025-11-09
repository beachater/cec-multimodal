classdef CoreSearch < handle
    % Options for all potential core search algorithms 
    properties 
        targetNumSubpo=1; % default number of subpopulations (must be one)
        iniSubpopSizeCoeff=6; % coefficient for the initial population size (popsize=Coeff*sqrt(D)
        finSubpopSizeCoeff=6; % coefficient for the final population size (popsize=Coeff*sqrt(D) 
        iniSigCoeff=2; % ratio of the initial step size to the least pairwise distance 
        muToPopSizeRatio=0.2; % ratio of parent size to offspring size
        maxIniSigma=0.3; % ratio of parent size to offspring size
        objValUpLimit=1e100; % A conservative upper limit for the values of feasible solutions
        repairInfeasMethodCode=1;
    end
    methods
        function coreSearch=CoreSearch(problem)
        end
    end
end

 
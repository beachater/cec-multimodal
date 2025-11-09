classdef Sampling < handle
    %----- information of the sampled solutions from the subpopulation -------
    properties
        s  % 1D array of step sizes
        X % 2D array of taboo acceptable solutions
        Z % matrix of Z_i's
        f  % 1D array of function values
        isFeas % if it is a feasible solution wrt boundaries
        wasRepaired % if it is a feasible solution wrt boundaries
        argsortNoElite=nan % index of the best member of the subpopulation excluding elites from the previous iterations
        argsortWithElite=nan % index of the best member of the subpopulation including elites from the previous iterations
    end
    methods
        function sample=Sampling(popSize,dim) % Constructor
            sample.s=nan(1,popSize);
            sample.X=nan(popSize,dim);
            sample.Z=nan(popSize,dim);
            sample.f=zeros(1,popSize)+inf;
            sample.isFeas=nan(1,popSize);
            sample.wasRepaired=zeros(1,popSize);
        end
    end % methods
end % class
        


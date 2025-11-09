classdef StopCriteria  < handle
    % stopping criterion for termination of a subpopulation 
    properties  
        tolHistFun=1e-6; % value for tolHistFun
        tolHistSizePar=[10 30]; % tolHistSize=x1+x2*D/lambda
        maxCondC=1e14; % upper-limit for condition number of C
        maxIterPar=[100, 50]; % [100, 50] for BI-POP-CMA, maxIter=x0+x1*(D+3)**2/sqrt(popSize) (see Hansen 2009)
        tolX=1e-12; % stopping criterion in the X-space
        stagPar=[120,0.2,30]; % [120,0.2,30] from BI-POP-CMA: sagnation window size= a0 + a1*iterNo + a2*D/popSize  (see Hansen 2009)
        merge; % merge convergence predictor 
        localConverge; % merge convergence predictor 
    end
    methods
        function stopCr=StopCriteria(problem)
            stopCr.merge=MergePredictor(); % merge convergence predictor 
            stopCr.localConverge=LocalConvergencePredictor(); % merge convergence predictor 
        end
    end
end

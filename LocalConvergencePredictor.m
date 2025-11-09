classdef LocalConvergencePredictor < handle
    % Options for local convergence predictor (if a subpopulation will/will not converge to a global minimum 
    properties 
        tolCoeff=.04;  %coefficient c_local for checking convergence to a undesirable minima if the improvement in the objective value is too small in comparison with the difference between the best found solution and subpopulation's best
        windowSizeCoeff=0.5; % this is multiplied by histsize of tolHistFun. The improvement in this window is regarded
        tabooCriticUpLimit=0.01; % no archived solution should be critical accoriding to this criterion in the window size so that this termination criterion may be activated. 
    end
end

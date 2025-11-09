classdef CoreSearchCMSA < CoreSearch     
    % Strategy parameters that are peculiar to CMSA (if it is employed as the core search algorithm 
    properties
        algorithm % name of the core search algorithm
        tauSigmaCoeff=0.5; % coefficient for learning rate of the sigma for CMSA: tau_sigma=sqrt(tauSigmaCoeff/(2D))
        eltRatio=.1; % fraction of elite solutions (for CMSA obly)
        tauCovCoeff=1; % learning rate coefficient for the covariance matrix
        sigmaUpdateBiasImp=1; % importance of compensation for bias for skewness of sampled step sizes when updating the global step size (should be 1 for RS-CMSA)
    end
    methods
        function coreSearch=CoreSearchCMSA(problem)
            coreSearch@CoreSearch(problem);
            coreSearch.algorithm='CMSA';
        end
    end
end
  
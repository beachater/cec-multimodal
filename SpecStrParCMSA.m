classdef SpecStrParCMSA < handle
    % Update the parameters of the core search algorithm: elite CMSA
    properties
        numElt; % No. of elite solutions
        tauCov; % learning rate for the covariance matrix
        tauSigma; % learning rate for step size
    end
    methods
        function specStrParCMSA=SpecStrParCMSA(process,opt,problem)
            specStrParCMSA.numElt=floor(ceil(opt.coreSearch.eltRatio*process.subpopSize));
            specStrParCMSA.tauCov=1+problem.dim*(1.0+problem.dim)/(2.0*process.muEff*opt.coreSearch.tauCovCoeff);
            specStrParCMSA.tauSigma=sqrt(0.5*opt.coreSearch.tauSigmaCoeff/problem.dim);           
        end
        function specStrParCMSA=update(specStrParCMSA,process,opt,problem) 
        	% update the core-search-specific parameters 
            specStrParCMSA.numElt=floor(ceil(opt.coreSearch.eltRatio*process.subpopSize));
            specStrParCMSA.tauCov=1+problem.dim*(1.0+problem.dim)/(2.0*process.muEff*opt.coreSearch.tauCovCoeff); 
            specStrParCMSA.tauSigma=sqrt(0.5*opt.coreSearch.tauSigmaCoeff/problem.dim); 
        end
    end
end

    
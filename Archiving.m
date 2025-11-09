classdef Archiving < handle
    % options for the archiving the best solution of each subpopulation when a restart concludes"""
    properties
        hillVallBudget=10 % detect multimodal budget for determining if a desired minimum is new
        iniNormTabDis=1 % default normalized taboo distance of initialized sub-populations in the zeroth restart
        newNormTabDisPrc=10 % after the first restart, the default normalized taboo distance is based on this percentile of the existing archived solutions
        targetNewNicheFr=.5 % this is the expected fraction of recent global minima (global ones) to be new
        targetGlobFr=.5 % this is the expected fraction of subpopulations to find a global minimum
        tauNormTabDis; % learning rate for adaptation of the normalized taboo distances
        tolFunArch=1e-5 % tolerance for considering a minimum desirable (or considering it as a global minimum)
        neighborSize=5 % how many archived points to check against. Only the closest ones to the candidate solution are checked
    end
    methods
        function archiving=Archiving(problem)
            archiving.tauNormTabDis=(1/problem.dim)^0.5; % learning rate for adaptation of the normalized taboo distances
        end
    end
end

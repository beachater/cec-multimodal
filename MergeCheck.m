classdef MergeCheck < handle
    %------- data for checking which archive point should the subpopulation marge into --------- 
    properties    
        bestCandidArchIndHist=[] % index of candidate archived solutions for merge
        bestCandidArchMergeabilityHist=[] % similiarity measure for each candidate archived solution
        matchArchInd=nan  % the archived solution with which the subpopulation is concluded to share the basin with
        checkAfterIterNo=0 % do not perform merge check until this iteration
        candidArchCountHist=[]% history of the the number of arhived solutions with above the thresho;d similarity metric
        mergeAtUsedEval=inf % merged should happen at this evaluation 
    end
end
 

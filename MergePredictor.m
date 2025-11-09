classdef MergePredictor < handle
    % Options for mergeing a subpopulation to an archived solution 
    properties
        threshold=.5; % mergeability threshold for merging a subpopulation to the archive solution
        windowSizeCoeff=.1; % must be equal or less than histsizeCoeff. for this window, that archive solution must be the only similar archived solution for merge to happen
        maxEval=10; % budget for checking if the canddiate archived solution is sharing the basin with the subpopulation best solution
        chkIntervalCoeff=.1; % how often this condition shpld be checked? The interval is multiplication of this number by 
        addedConst=1;  % fixed value added to the nomalized distance when calculating the mergeability
    end
end

                
% This class specifies the options for the control parameters of the algorithm.
% Written by Ali Ahrari (aliahrari1983@gmail.com)
% last updated by Ali Ahrari on 16 Jan 2022 

classdef OptimOption < handle
    properties
        archiving; % Object archive includes options for updating the archive
        coreSearch; % options for the core search algorithm
        niching; % options for the niching mechanism
        stopCr; %  stopping criteria for termination of a subpopulation
        dyna; % dynamic options
    end
    methods
        function opt=OptimOption(problem,coreSearchName)
            opt.archiving=Archiving(problem); % Object archive includes options for updating the archive
            if strcmp(coreSearchName,'CMSA')
                opt.coreSearch=CoreSearchCMSA(problem);
            elseif strcmp(coreSearchName,'CMA')
                opt.coreSearch=CoreSearchCMA(problem);
            else
                disp('Option for core search is invalid')
                abort
            end
            opt.niching=Niching(problem); % options for the niching mechanism
            opt.stopCr=StopCriteria(problem); % other stopping criteria for termination of a subpopulation
            opt.dyna=DynamicOption();
        end
    end
end
    
    
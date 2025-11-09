classdef Niching <handle
    % options for diversity preservation 
    properties
        criticTabooThresh=.01; % threshold for determining of critical taboo points for each subpopulation
        redCoeff=nan; % For temporary reduction of the taboo region
        iniR0IncFac=1.04; % slight increase in the trail taboo radius for initialization after each restart
        maxRejectIni=100; % maximum number of successive rejction before reduction of niche radius in initialization or temporary shrinkage of taboo regions when sampling
    end
    methods
        function niching=Niching(problem) % Constructor
            niching.redCoeff=0.99^(1.0/problem.dim); 
        end
    end
end

    

        
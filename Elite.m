classdef Elite < handle 
    % ------ Elite solutions of the subpopulations -------- 
    properties
        sol % elite solutions
        val=[] % elite values
        s=[] % elite step sizes
        Z % elite Z 
        wasRepaired=[] % if the elite solution was repaired
    end
    methods
        function elite=Elite(D) % Constructor
            elite.sol=zeros(0,D);
            elite.Z=zeros(0,D);
        end
    end
end
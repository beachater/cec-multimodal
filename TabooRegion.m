classdef TabooRegion 
    % --------- information of taboo regions for the subpopulation ---------- 
    properties 
        center=[]; % center of taboo regions
        normTabDis=[]; % normalized taboo distance of taboo regions
        criticality=[]; % criticality of taboo regions
        criticInd=[]; % index of taboo regions that are critical
    end
    methods
        function tabReg=TabooRegion(D)
            tabReg.center=zeros(0,D);
        end
    end
            
end

 